"""Wound + blood segmentation with body-gated filtering.

Pipeline stages:
    1. ``SegBackend.person_mask(frame, bbox)`` -> binary silhouette of the victim.
    2. ``SegBackend.detect_and_segment(frame, bbox, prompt)`` -> proposed wound /
       blood regions.
    3. The orchestrator (``WoundSegmenter``) intersects every candidate with a
       *dilated* silhouette so red walls, red paint, and unrelated scenery can
       never be labelled as blood.

Backends:
    - ``GdinoSam2Backend`` (default): Grounding DINO (HF) + SAM 2 for masks.
    - ``Sam3Backend`` (optional, ``max`` profile): SAM 3 / 3.1 from
      `facebookresearch/sam3` for text-prompt detection + segmentation in one
      pass. Falls back to GDINO+SAM2 if the repo / gated checkpoint is missing.

Fallback-fallback: HSV color heuristic, still body-gated.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from state.victim import BloodRegion, WoundRegion


# ---------------------------------------------------------------------------
# Label vocabularies — used to keep GDINO hallucinations out of the output.
# ---------------------------------------------------------------------------

# Positive injury vocabulary. A returned label must contain at least one of
# these substrings to be accepted as a wound / blood finding.
INJURY_KEYWORDS = frozenset({
    "wound", "laceration", "cut", "gash", "bleed", "blood", "hemorr",
    "burn", "charred", "soot", "blister", "singed",
    "shrapnel", "fragment", "penetrat", "impaled", "puncture",
    "bone", "exposed", "amputat", "stump",
    "tourniquet", "dressing", "bandage",
    "bruise", "contusion", "abrasion", "avulsion", "crush",
})

# Never treat these tokens as injury labels even if they slip through the
# keyword filter — they are the main source of GDINO false positives
# because the prompt contains nouns like "body" / "skin".
LABEL_BLACKLIST = frozenset({
    "body", "person", "face", "head", "skin", "clothes", "clothing",
    "shirt", "pants", "jacket", "hair", "eye", "mouth", "nose", "ear",
    "limb", "arm", "leg", "hand", "foot", "torso",
})

# Canonical label mapping — keeps display labels human-readable and stable.
_CANON_MAP = (
    ("blood", "bleeding"),
    ("bleed", "bleeding"),
    ("hemorr", "bleeding"),
    ("laceration", "laceration"),
    ("cut", "laceration"),
    ("gash", "laceration"),
    ("avulsion", "laceration"),
    ("puncture", "penetrating wound"),
    ("penetrat", "penetrating wound"),
    ("impaled", "impaled object"),
    ("shrapnel", "shrapnel"),
    ("fragment", "shrapnel"),
    ("bone", "exposed bone"),
    ("amputat", "amputation"),
    ("stump", "amputation"),
    ("tourniquet", "tourniquet"),
    ("dressing", "dressing"),
    ("bandage", "dressing"),
    ("burn", "burn"),
    ("charred", "burn"),
    ("soot", "soot"),
    ("blister", "blister"),
    ("singed", "burn"),
    ("bruise", "bruise"),
    ("contusion", "bruise"),
    ("road rash", "abrasion"),
    ("abrasion", "abrasion"),
    ("crush", "crush injury"),
    ("wound", "wound"),
)


def _label_is_injury(lbl: str) -> bool:
    """Reject prompt tokens like 'body' / 'person' / 'face' that GDINO often
    returns because the prompt contained them.
    """
    s = (lbl or "").lower().strip()
    if not s:
        return False
    # Exact or isolated-token blacklist: "body", "person", "face", ...
    tokens = set(s.replace("_", " ").split())
    if tokens and tokens.issubset(LABEL_BLACKLIST):
        return False
    return any(k in s for k in INJURY_KEYWORDS)


def canonicalize_label(lbl: str) -> str:
    s = (lbl or "").lower().strip()
    for key, canon in _CANON_MAP:
        if key in s:
            return canon
    return s or "wound"


def split_prompt_phrases(prompt: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\s*\.\s*", str(prompt or "")) if p.strip()]
    out: List[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(part)
    return out


def join_prompt_phrases(parts: List[str]) -> str:
    clean = [p.strip() for p in parts if str(p or "").strip()]
    return " . ".join(clean)


# ---------------------------------------------------------------------------
# Severity helpers (used by every backend)
# ---------------------------------------------------------------------------

# Injury classes that are inherently high-severity *if* they are real.
_CRIT_KEYWORDS = ("amputat", "exposed bone", "stump")
_SERIOUS_KEYWORDS = ("tourniquet", "gsw", "gunshot", "penetrat", "shrapnel", "impaled")
_MOD_KEYWORDS = ("burn", "laceration", "crush", "abrasion", "bruise")


def estimate_wound_severity(
    label: str,
    wound_area_px: int,
    person_bbox_area_px: int,
    confidence: float = 0.0,
    evidence: Optional[Dict[str, float]] = None,
) -> str:
    """Severity now requires **corroboration**, not just size.

    The rule is "evidence AND confidence AND area" — each term is required
    to upgrade past ``minor``. This dramatically reduces the "critical
    shrapnel on face" failure mode seen in field tests.
    """
    ratio = wound_area_px / max(person_bbox_area_px, 1)
    lbl = (label or "").lower().replace("_", " ")
    ev = evidence or {}
    blood = float(ev.get("blood_frac", 0.0))
    dark = float(ev.get("dark_frac", 0.0))
    visual = max(blood, dark)

    looks_critical = any(k in lbl for k in _CRIT_KEYWORDS)
    looks_serious = any(k in lbl for k in _SERIOUS_KEYWORDS)
    looks_moderate = any(k in lbl for k in _MOD_KEYWORDS)

    # Critical requires: critical-label OR (serious-label + large area + high confidence + strong visual).
    if looks_critical and confidence >= 0.45 and visual >= 0.08:
        return "critical"
    if (
        looks_serious
        and ratio >= 0.03
        and confidence >= 0.55
        and visual >= 0.10
    ):
        return "critical"
    if looks_serious and confidence >= 0.45 and visual >= 0.06:
        return "serious"
    if (
        ratio >= 0.04
        and confidence >= 0.55
        and visual >= 0.12
    ):
        return "serious"
    if looks_moderate and confidence >= 0.40 and visual >= 0.04:
        return "moderate"
    if ratio >= 0.015 and confidence >= 0.45 and visual >= 0.03:
        return "moderate"
    if ratio > 0.001 and confidence >= 0.40 and visual >= 0.02:
        return "minor"
    # Low-confidence / no-evidence fallback: mark explicitly as possible.
    return "possible"


def reinterpret_diffuse_trauma_label(
    label: str,
    confidence: float,
    evidence: Optional[Dict[str, float]] = None,
    *,
    demo_mode: bool = False,
) -> str:
    """Downgrade implausible high-risk labels into diffuse trauma classes.

    Open-vocab grounding can map obvious abrasions or burn-like trauma to
    ``amputation`` or ``tourniquet`` when the region is large and textured.
    If the region looks like broad surface trauma instead of a discrete device
    or limb-loss pattern, reinterpret it before the evidence gate.
    """
    lbl = (label or "").lower().strip()
    if not lbl:
        return label
    if not any(k in lbl for k in ("amput", "stump", "tourniquet", "dressing", "bandage")):
        return label

    ev = evidence or {}
    blood = float(ev.get("blood_frac", 0.0))
    dark = float(ev.get("dark_frac", 0.0))
    skin = float(ev.get("skin_frac", 0.0))
    edge = float(ev.get("edge_density", 0.0))
    area_ratio = float(ev.get("area_ratio", 0.0))

    if demo_mode:
        max_conf = 0.62
        min_edge = 30.0
        min_area = 0.012
    else:
        max_conf = 0.56
        min_edge = 40.0
        min_area = 0.018

    diffuse_surface_trauma = (
        confidence <= max_conf
        and blood < 0.03
        and edge >= min_edge
        and area_ratio >= min_area
    )
    if not diffuse_surface_trauma:
        return label

    # Burn-like regions skew darker / less skin-like; abrasion-like regions are
    # lighter textured trauma with strong edges and larger exposed-skin area.
    if dark >= 0.04 or skin <= 0.30:
        return "burn"
    return "abrasion"


def _px_area_to_volume_ml(area_px: float, person_height_px: float, assumed_height_cm: float = 170.0) -> float:
    if person_height_px < 1:
        return 0.0
    px_per_cm = person_height_px / assumed_height_cm
    area_cm2 = area_px / max(px_per_cm * px_per_cm, 1e-6)
    depth_cm = 0.3
    return float(area_cm2 * depth_cm)


def _volume_bucket(ml: float) -> str:
    if ml < 500:
        return "<500ml"
    if ml < 1500:
        return "500-1500ml"
    return ">1500ml"


# ---------------------------------------------------------------------------
# Body-gate primitives
# ---------------------------------------------------------------------------

def _dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Dilate a binary mask by `px` pixels using a round kernel."""
    import cv2
    if px <= 0 or mask is None or mask.size == 0:
        return mask
    k = max(3, (px * 2) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.dilate(mask, kernel, iterations=1)


def gate_by_body(candidate_mask: np.ndarray,
                 person_mask: np.ndarray,
                 dilation_px: int = 18) -> np.ndarray:
    """Keep only candidate pixels inside (or within ``dilation_px`` of) ``person_mask``."""
    if candidate_mask is None or candidate_mask.size == 0:
        return candidate_mask
    if person_mask is None or person_mask.size == 0:
        return np.zeros_like(candidate_mask)
    dilated = _dilate_mask(person_mask, dilation_px)
    # Both must be uint8 {0,255} for bitwise.
    cm = (candidate_mask > 0).astype(np.uint8) * 255
    pm = (dilated > 0).astype(np.uint8) * 255
    import cv2
    return cv2.bitwise_and(cm, pm)


def _connected_to_body(candidate_mask: np.ndarray,
                       person_mask: np.ndarray,
                       dilation_px: int = 24) -> np.ndarray:
    """Keep only connected components of ``candidate_mask`` that touch the body.

    Used for floor blood pools: any red region not linked (via dilated body
    silhouette) to the victim is discarded as scenery.
    """
    import cv2
    if candidate_mask is None or candidate_mask.size == 0 or person_mask is None:
        return candidate_mask
    dilated = _dilate_mask(person_mask, dilation_px)
    cand = (candidate_mask > 0).astype(np.uint8)
    n, labels, _, _ = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(cand)
    for cid in range(1, n):
        cc_mask = (labels == cid).astype(np.uint8)
        if cv2.countNonZero(cv2.bitwise_and(cc_mask, (dilated > 0).astype(np.uint8))) > 0:
            out |= cc_mask
    return (out * 255).astype(np.uint8)


def _is_flat_paint(region_hsv: np.ndarray) -> bool:
    """Reject candidates that look like uniform wall paint (low saturation
    variance, tight hue)."""
    if region_hsv is None or region_hsv.size == 0:
        return False
    h = region_hsv[..., 0].flatten()
    s = region_hsv[..., 1].flatten()
    if h.size < 32:
        return False
    return bool(np.median(s) < 60 or np.std(h) < 3)


def visual_evidence(frame_bgr: np.ndarray, retained_mask: np.ndarray) -> Dict[str, float]:
    """Per-detection evidence fingerprint.

    We compute the fraction of pixels inside the *body-gated* candidate mask
    that look like blood, dark tissue, or plain skin, plus an edge-density
    estimate via Laplacian variance. These are the primary discriminators
    between a real wound and a healthy-skin false positive.
    """
    import cv2
    if retained_mask is None or retained_mask.size == 0:
        return {"px": 0.0, "blood_frac": 0.0, "dark_frac": 0.0,
                "skin_frac": 0.0, "edge_density": 0.0, "red_sat_mean": 0.0}
    mask = (retained_mask > 0)
    n = int(mask.sum())
    if n < 32:
        return {"px": float(n), "blood_frac": 0.0, "dark_frac": 0.0,
                "skin_frac": 0.0, "edge_density": 0.0, "red_sat_mean": 0.0}
    pixels = frame_bgr[mask].reshape(-1, 1, 3)
    hsv = cv2.cvtColor(pixels, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    H, S, V = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    red = ((H <= 10) | (H >= 170)) & (S >= 90) & (V >= 50) & (V <= 220)
    dark = (V <= 65) & (S >= 40)  # dark tissue / clot; avoid pure-black shadows
    skin = ((H <= 28) | (H >= 170)) & (S >= 25) & (S <= 170) & (V >= 65)

    # Edge density on the bounding box of the retained region.
    ys, xs = np.where(mask)
    if ys.size > 0:
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1 = max(0, y1); x1 = max(0, x1)
        y2 = min(frame_bgr.shape[0], y2); x2 = min(frame_bgr.shape[1], x2)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            edge = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        else:
            edge = 0.0
    else:
        edge = 0.0
    return {
        "px": float(n),
        "blood_frac": float(red.mean()),
        "dark_frac": float(dark.mean()),
        "skin_frac": float(skin.mean()),
        "edge_density": edge,
        "red_sat_mean": float(S[red].mean()) if red.any() else 0.0,
    }


# ---------------------------------------------------------------------------
# SegBackend protocol + implementations
# ---------------------------------------------------------------------------

class SegBackend(Protocol):
    """Common interface for any detection + segmentation backend."""

    name: str

    def person_mask(self, frame_bgr: np.ndarray,
                    bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Return a full-frame uint8 mask of the person silhouette, or None."""
        ...

    def detect_and_segment(self, frame_bgr: np.ndarray,
                           bbox: Tuple[int, int, int, int],
                           prompt: str) -> List[Dict[str, Any]]:
        """Return raw proposals: list of dicts with keys
        ``{label, score, bbox, mask}`` (bbox in frame coords, mask full-frame
        uint8 or None)."""
        ...


class GdinoSam2Backend:
    """Grounding DINO (HF transformers) + SAM 2 (from ``sam2`` package).

    This is the default, works on HF-only installs.
    """

    name = "gdino_sam2"

    def __init__(self,
                 gdino_model: str = "IDEA-Research/grounding-dino-base",
                 sam_model: str = "facebook/sam2-hiera-small",
                 use_grounding_dino: bool = True,
                 use_sam: bool = True,
                 box_threshold: float = 0.40,
                 text_threshold: float = 0.30) -> None:
        self.gdino_model = gdino_model
        self.sam_model = sam_model
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._gdino = None
        self._gdino_processor = None
        self._gdino_device = "cpu"
        self._sam_predictor = None
        self._torch = None
        if use_grounding_dino:
            self._try_load_gdino()
        if use_sam:
            self._try_load_sam()

    def _try_load_gdino(self) -> None:
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # type: ignore
            self._torch = torch
            self._gdino_processor = AutoProcessor.from_pretrained(self.gdino_model)
            self._gdino = AutoModelForZeroShotObjectDetection.from_pretrained(self.gdino_model)
            dev = "cuda" if torch.cuda.is_available() else (
                "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"
            )
            self._gdino = self._gdino.to(dev).eval()
            self._gdino_device = dev
            print(f"[wound] Grounding DINO ({self.gdino_model}) loaded on {dev}.")
        except Exception as exc:
            print(f"[wound] Grounding DINO unavailable ({exc}); using color fallback.")
            self._gdino = None

    def _try_load_sam(self) -> None:
        try:
            import torch  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._sam_predictor = SAM2ImagePredictor.from_pretrained(self.sam_model, device=device)
            print(f"[wound] SAM 2.1 ({self.sam_model}) loaded on {device}.")
        except Exception as exc:
            print(f"[wound] SAM 2.1 unavailable ({exc}); boxes only.")
            self._sam_predictor = None

    # ------------------------------------------------------------------
    def person_mask(self, frame_bgr, bbox):
        """SAM-on-box prompt for the victim silhouette. Returns full-frame
        uint8 mask or None."""
        if self._sam_predictor is None:
            return None
        import cv2
        x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
        if x2 <= x1 or y2 <= y1:
            return None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        try:
            self._sam_predictor.set_image(rgb)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            masks, scores, _ = self._sam_predictor.predict(
                point_coords=np.array([[cx, cy]], dtype=np.float32),
                point_labels=np.array([1], dtype=np.int32),
                box=np.array([x1, y1, x2, y2], dtype=np.float32),
                multimask_output=False,
            )
            m = _first_mask(masks)
            if m is None:
                return None
            return (m > 0).astype(np.uint8) * 255
        except Exception as exc:
            print(f"[wound] SAM person-mask failed ({exc})")
            return None

    def _sam_mask_for_box(self, rgb, box_xyxy) -> Optional[np.ndarray]:
        if self._sam_predictor is None:
            return None
        try:
            self._sam_predictor.set_image(rgb)
            masks, _s, _l = self._sam_predictor.predict(
                box=np.array(box_xyxy, dtype=np.float32),
                multimask_output=False,
            )
            m = _first_mask(masks)
            if m is None:
                return None
            return (m > 0).astype(np.uint8) * 255
        except Exception:
            return None

    # ------------------------------------------------------------------
    def detect_and_segment(self, frame_bgr, bbox, prompt,
                           box_threshold: Optional[float] = None,
                           text_threshold: Optional[float] = None,
                           ) -> List[Dict[str, Any]]:
        """Run GDINO + SAM2 on ``bbox``.

        ``box_threshold`` / ``text_threshold`` override the instance values
        so callers can run an ensemble / strict-threshold second pass
        without having to rebuild the backend.
        """
        if self._gdino is None:
            return []
        import cv2
        box_th = self.box_threshold if box_threshold is None else float(box_threshold)
        text_th = self.text_threshold if text_threshold is None else float(text_threshold)
        x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
        if x2 <= x1 or y2 <= y1:
            return []
        crop = frame_bgr[y1:y2, x1:x2]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        inputs = self._gdino_processor(images=rgb_crop, text=prompt,
                                       return_tensors="pt").to(self._gdino_device)
        with self._torch.no_grad():
            outputs = self._gdino(**inputs)
        try:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                threshold=box_th,
                text_threshold=text_th,
                target_sizes=[rgb_crop.shape[:2]],
            )[0]
        except TypeError:
            results = self._gdino_processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_th,
                text_threshold=text_th,
                target_sizes=[rgb_crop.shape[:2]],
            )[0]

        label_key = "text_labels" if "text_labels" in results else "labels"
        out: List[Dict[str, Any]] = []
        for score, label, box in zip(results["scores"], results[label_key], results["boxes"]):
            bx1, by1, bx2, by2 = box.tolist()
            gbox = (int(x1 + bx1), int(y1 + by1), int(x1 + bx2), int(y1 + by2))
            raw_label = str(label).strip().lower()
            # Drop hallucinated labels that are just prompt tokens ("body",
            # "person", "face", ...). These are the #1 source of the
            # "CRITICAL SHRAPNEL BODY" false positive on clean faces.
            if not _label_is_injury(raw_label):
                continue
            lbl = canonicalize_label(raw_label)
            mask = self._sam_mask_for_box(full_rgb, gbox)
            out.append({
                "label": lbl,
                "raw_label": raw_label,
                "score": float(score),
                "bbox": gbox,
                "mask": mask,
            })
        return out


def _sam3_load_error_hint(exc: Exception) -> str:
    """One-line reason + fix for SAM3 loader failures (avoid20-line HF traces)."""
    msg = str(exc).strip()
    first = msg.split("\n")[0][:140]
    low = msg.lower()
    if "401" in msg or "gated" in low or "restricted" in low or "cannot access" in low:
        return (
            "gated HF repo — accept the license, then `huggingface-cli login` "
            "or set HF_TOKEN"
        )
    if "no module named 'sam3'" in low or "no module named \"sam3\"" in low:
        return "sam3 package not installed"
    return first


class Sam3Backend:
    """SAM 3 / 3.1 text-grounded backend.

    Tries multiple loaders in order:

    1. ``from sam3 import build_sam3_predictor`` — Meta's upstream repo.
    2. HuggingFace ``transformers`` ``AutoModelForZeroShotObjectDetection``
       (or ``AutoModelForMaskGeneration``) with ``facebook/sam3`` checkpoint.

    If both fail we fall back silently to the ``delegate`` backend so the
    pipeline keeps working on machines that haven't downloaded the SAM 3
    weights yet.  The ensemble gate in
    :meth:`WoundSegmenter.strict_survivors` still works because it walks
    through to the delegate.

    SAM 3 returns text-grounded per-instance masks in a single pass and,
    empirically, produces far fewer phrase-grounding artifacts than
    GDINO+SAM2 (no "body" / "face" hallucination).
    """

    name = "sam3"

    def __init__(self,
                 version: str = "sam3.1",
                 delegate: Optional[SegBackend] = None,
                 hf_checkpoint: Optional[str] = None) -> None:
        self._delegate = delegate
        self._predictor = None
        self._hf_processor = None
        self._hf_model = None
        self._hf_device = None
        self._torch = None
        self._active = False
        self._mode: str = "off"  # "upstream" | "hf" | "off"
        self._checkpoint = hf_checkpoint or (
            "facebook/sam3" if version.startswith("sam3") else version
        )

        # --- loader 1: upstream `sam3` python package ----------------------
        try:
            from sam3 import build_sam3_predictor  # type: ignore
            self._predictor = build_sam3_predictor(version=version)
            self._active = True
            self._mode = "upstream"
            print(f"[wound] SAM {version} backend active (upstream).")
            return
        except Exception as exc_upstream:
            upstream_err = exc_upstream

        # --- loader 2: HuggingFace transformers ----------------------------
        try:
            import torch  # type: ignore
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # type: ignore
            self._torch = torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._hf_processor = AutoProcessor.from_pretrained(self._checkpoint)
            self._hf_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self._checkpoint,
            ).to(device).eval()
            self._hf_device = device
            self._active = True
            self._mode = "hf"
            print(f"[wound] SAM {version} backend active via HuggingFace "
                  f"({self._checkpoint} on {device}).")
            return
        except Exception as exc_hf:
            u_h = _sam3_load_error_hint(upstream_err) if isinstance(upstream_err, Exception) else str(upstream_err).split("\n")[0][:80]
            h_h = _sam3_load_error_hint(exc_hf)
            print(
                f"[wound] SAM3 unavailable (upstream: {u_h}; HF: {h_h}) — "
                f"using {delegate.name if delegate else 'none'}."
            )

    def person_mask(self, frame_bgr, bbox):
        if self._mode == "upstream":
            try:
                return _sam3_person_mask(self._predictor, frame_bgr, bbox)
            except Exception as exc:
                print(f"[wound] SAM3 person_mask failed ({exc})")
        # HF path currently lacks a usable mask head; rely on the delegate
        # for person silhouette (GDINO+SAM2 gives a great one).
        return self._delegate.person_mask(frame_bgr, bbox) if self._delegate else None

    def detect_and_segment(self, frame_bgr, bbox, prompt,
                           box_threshold: Optional[float] = None,
                           text_threshold: Optional[float] = None):
        if self._mode == "upstream":
            try:
                return _sam3_text_detect(self._predictor, frame_bgr, bbox, prompt)
            except Exception as exc:
                print(f"[wound] SAM3 detect failed ({exc})")
        if self._mode == "hf":
            try:
                return self._hf_detect(
                    frame_bgr, bbox, prompt,
                    box_threshold=box_threshold or 0.30,
                    text_threshold=text_threshold or 0.22,
                )
            except Exception as exc:
                print(f"[wound] SAM3 HF detect failed ({exc})")
        if self._delegate is not None:
            try:
                return self._delegate.detect_and_segment(
                    frame_bgr, bbox, prompt,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                )
            except TypeError:
                return self._delegate.detect_and_segment(frame_bgr, bbox, prompt)
        return []

    # ------------------------------------------------------------------
    def _hf_detect(self, frame_bgr, bbox, prompt,
                   box_threshold: float = 0.30,
                   text_threshold: float = 0.22) -> List[Dict[str, Any]]:
        """HuggingFace text-grounded detection path.

        Follows the same post-processing path as GDINO: detection boxes get
        SAM2 masks from the delegate (if one exists) so the downstream gate
        still sees a proper instance mask.
        """
        import cv2
        torch = self._torch
        x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
        if x2 <= x1 or y2 <= y1:
            return []
        crop = frame_bgr[y1:y2, x1:x2]
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        inputs = self._hf_processor(
            images=rgb_crop, text=prompt, return_tensors="pt"
        ).to(self._hf_device)
        with torch.no_grad():
            outputs = self._hf_model(**inputs)
        try:
            results = self._hf_processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs.input_ids,
                threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[rgb_crop.shape[:2]],
            )[0]
        except Exception:
            return []
        label_key = "text_labels" if "text_labels" in results else "labels"
        out: List[Dict[str, Any]] = []
        for score, label, box in zip(results["scores"], results[label_key], results["boxes"]):
            bx1, by1, bx2, by2 = box.tolist()
            gbox = (int(x1 + bx1), int(y1 + by1), int(x1 + bx2), int(y1 + by2))
            raw_label = str(label).strip().lower()
            if not _label_is_injury(raw_label):
                continue
            lbl = canonicalize_label(raw_label)
            mask = None
            if self._delegate is not None and hasattr(self._delegate, "_sam_mask_for_box"):
                try:
                    full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    mask = self._delegate._sam_mask_for_box(full_rgb, gbox)
                except Exception:
                    mask = None
            out.append({
                "label": lbl,
                "raw_label": raw_label,
                "score": float(score),
                "bbox": gbox,
                "mask": mask,
            })
        return out


# Thin adapters for the sam3 library. Kept as module-level helpers so the
# public Sam3Backend class stays readable and so we can mock them out in tests
# once we add them. The exact sam3 API may differ between releases; these are
# wrapped in try/except at the call site.

def _sam3_person_mask(predictor, frame_bgr, bbox):
    x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
    result = predictor.predict(
        image=frame_bgr,
        text_prompt="person",
        boxes=[[x1, y1, x2, y2]],
    )
    masks = result.get("masks") if isinstance(result, dict) else None
    if not masks:
        return None
    m = np.asarray(masks[0])
    return (m > 0).astype(np.uint8) * 255


def _sam3_text_detect(predictor, frame_bgr, bbox, prompt):
    x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
    result = predictor.predict(
        image=frame_bgr,
        text_prompt=prompt,
        boxes=[[x1, y1, x2, y2]],
    )
    out: List[Dict[str, Any]] = []
    labels = result.get("labels", [])
    scores = result.get("scores", [])
    boxes = result.get("boxes", [])
    masks = result.get("masks", [])
    for i, lbl in enumerate(labels):
        box = boxes[i] if i < len(boxes) else None
        mask = masks[i] if i < len(masks) else None
        if box is None:
            continue
        out.append({
            "label": str(lbl).strip().lower(),
            "score": float(scores[i]) if i < len(scores) else 0.5,
            "bbox": tuple(int(v) for v in box),
            "mask": (np.asarray(mask) > 0).astype(np.uint8) * 255 if mask is not None else None,
        })
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class WoundSegmenter:
    """Body-gated wound/blood detector.

    Backends are swappable via ``WoundSegmenter(backend=...)``; the default
    keeps the previous GDINO+SAM2 behaviour for backwards compatibility.
    """

    def __init__(self,
                 gdino_prompt: str,
                 box_threshold: float = 0.40,
                 text_threshold: float = 0.30,
                 use_grounding_dino: bool = True,
                 use_sam: bool = True,
                 gdino_model: str = "IDEA-Research/grounding-dino-base",
                 sam_model: str = "facebook/sam2-hiera-base-plus",
                 backend: Optional[SegBackend] = None,
                 body_dilation_px: int = 12,
                 min_body_overlap: float = 0.55,
                 debug_enabled: bool = False,
                 debug_dir: str = "logs/wound_debug",
                 demo_mode: bool = False) -> None:
        self.gdino_prompt = gdino_prompt
        self.body_dilation_px = body_dilation_px
        self.min_body_overlap = min_body_overlap
        self.debug_enabled = bool(debug_enabled)
        self.debug_dir = str(debug_dir)
        self.demo_mode = bool(demo_mode)
        self._last_debug: Dict[str, Any] = {}
        self.backend: SegBackend = backend or GdinoSam2Backend(
            gdino_model=gdino_model,
            sam_model=sam_model,
            use_grounding_dino=use_grounding_dino,
            use_sam=use_sam,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
        )

    # ------------------------------------------------------------------
    def set_prompt(self, prompt: str) -> None:
        self.gdino_prompt = prompt

    def set_backend(self, backend: SegBackend) -> None:
        self.backend = backend

    def set_demo_mode(self, enabled: bool) -> None:
        self.demo_mode = bool(enabled)

    @property
    def last_debug(self) -> Dict[str, Any]:
        return dict(self._last_debug)

    # ------------------------------------------------------------------
    def person_mask(self, frame_bgr, bbox) -> Optional[np.ndarray]:
        """Public helper so the scan engine can reuse the silhouette."""
        return self.backend.person_mask(frame_bgr, bbox)

    # ------------------------------------------------------------------
    def _prompt_fallback_passes(self) -> List[Dict[str, Any]]:
        phrases = split_prompt_phrases(self.gdino_prompt)
        if not phrases:
            return []

        passes: List[Dict[str, Any]] = []
        passes.append({
            "name": "primary",
            "prompt": join_prompt_phrases(phrases),
            "box_threshold": None,
            "text_threshold": None,
        })

        groups = {
            "surface": [],
            "bleed": [],
            "penetrating": [],
            "support": [],
        }
        for phrase in phrases:
            low = phrase.lower()
            if any(k in low for k in ("burn", "abrasion", "road rash", "blister", "charred", "singed", "bruise", "contusion")):
                groups["surface"].append(phrase)
            elif any(k in low for k in ("blood", "bleed", "laceration", "wound", "cut", "gash", "avulsion")):
                groups["bleed"].append(phrase)
            elif any(k in low for k in ("fragment", "penetrat", "impaled", "bone", "mangled", "crush", "amput")):
                groups["penetrating"].append(phrase)
            elif any(k in low for k in ("tourniquet", "dressing", "bandage")):
                groups["support"].append(phrase)

        fallback_box = 0.22 if self.demo_mode else 0.26
        fallback_text = 0.18 if self.demo_mode else 0.20
        for name in ("surface", "bleed", "penetrating", "support"):
            if groups[name]:
                passes.append({
                    "name": name,
                    "prompt": join_prompt_phrases(groups[name]),
                    "box_threshold": fallback_box,
                    "text_threshold": fallback_text,
                })

        if self.demo_mode:
            for phrase in phrases:
                passes.append({
                    "name": f"phrase:{phrase[:28]}",
                    "prompt": phrase,
                    "box_threshold": 0.16,
                    "text_threshold": 0.14,
                })
        return passes

    def _merge_detection_proposals(self, proposals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        for proposal in sorted(proposals, key=lambda p: float(p.get("score", 0.0)), reverse=True):
            box = proposal.get("bbox")
            if not box:
                continue
            label = canonicalize_label(str(proposal.get("label", "")).strip().lower())
            proposal["label"] = label
            found = None
            for existing in merged:
                if existing.get("label") != label:
                    continue
                ebox = existing.get("bbox")
                if not ebox:
                    continue
                if _iou(tuple(ebox), tuple(box)) >= 0.55:
                    found = existing
                    break
            if found is None:
                merged.append(proposal)
                continue
            if float(proposal.get("score", 0.0)) > float(found.get("score", 0.0)):
                found.update({k: v for k, v in proposal.items() if k != "pass_name"})
            if not found.get("mask") and proposal.get("mask") is not None:
                found["mask"] = proposal.get("mask")
            source_passes = set(found.get("source_passes", []))
            source_passes.update(proposal.get("source_passes", []))
            if proposal.get("pass_name"):
                source_passes.add(proposal["pass_name"])
            found["source_passes"] = sorted(source_passes)
        return merged

    def _detect_with_fallbacks(
        self,
        frame_bgr: np.ndarray,
        victim_bbox: Tuple[int, int, int, int],
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        passes_meta: List[Dict[str, Any]] = []
        all_props: List[Dict[str, Any]] = []
        for idx, spec in enumerate(self._prompt_fallback_passes()):
            try:
                props = self.backend.detect_and_segment(
                    frame_bgr,
                    victim_bbox,
                    spec["prompt"],
                    box_threshold=spec.get("box_threshold"),
                    text_threshold=spec.get("text_threshold"),
                )
            except TypeError:
                props = self.backend.detect_and_segment(frame_bgr, victim_bbox, spec["prompt"])
            except Exception as exc:
                passes_meta.append({
                    "name": spec["name"],
                    "prompt": spec["prompt"],
                    "error": str(exc),
                    "proposal_count": 0,
                })
                continue

            for proposal in props:
                proposal["pass_name"] = spec["name"]
                proposal["source_passes"] = [spec["name"]]
            passes_meta.append({
                "name": spec["name"],
                "prompt": spec["prompt"],
                "box_threshold": spec.get("box_threshold"),
                "text_threshold": spec.get("text_threshold"),
                "proposal_count": len(props),
            })
            all_props.extend(props)
            if idx == 0 and props:
                break
            if props and spec["name"] in ("surface", "bleed", "penetrating"):
                break

        return self._merge_detection_proposals(all_props), passes_meta

    # ------------------------------------------------------------------
    def strict_survivors(self,
                         frame_bgr: np.ndarray,
                         bbox: Tuple[int, int, int, int],
                         box_threshold: float = 0.55,
                         text_threshold: float = 0.45,
                         ) -> List[Dict[str, Any]]:
        """Return the *raw* proposals that survive a strict-threshold
        GDINO pass. Used as an ensemble gate — if a critical-class
        detection from the primary pass does **not** appear here we
        demote it to ``serious`` to avoid over-promoting a shaky call.
        """
        backend = self.backend
        # Only supported on the GDINO backend (and Sam3Backend which
        # delegates to it when SAM3 weights are missing).
        primary = getattr(backend, "_delegate", backend)
        if not hasattr(primary, "detect_and_segment"):
            return []
        try:
            return primary.detect_and_segment(
                frame_bgr, bbox, self.gdino_prompt,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
            )
        except TypeError:
            # Backend doesn't accept override thresholds.
            return []
        except Exception as exc:
            print(f"[wound] strict pass failed ({exc})")
            return []

    # ------------------------------------------------------------------
    def process(self,
                frame_bgr: np.ndarray,
                victim_bbox: Tuple[int, int, int, int],
                person_mask: Optional[np.ndarray] = None,
                face_bbox: Optional[Tuple[int, int, int, int]] = None,
                session_negatives: Optional[List[Tuple[str, str]]] = None,
                debug_scan_id: Optional[str] = None,
                ) -> Tuple[List[WoundRegion], List[BloodRegion]]:
        """Return (wound_regions, blood_regions) inside ``victim_bbox``.

        Parameters
        ----------
        person_mask:
            Optional pre-computed silhouette so the caller avoids a second
            SAM call (the ScanEngine reuses its own person_mask).
        face_bbox:
            Full-frame face bbox from InsightFace. When supplied, detections
            whose centre falls inside the face must show strong visual
            evidence (blood/dark) to survive — this is the face-aware
            suppression gate.  Kills "shrapnel on the cheek" false positives.
        session_negatives:
            List of ``(label, body_region)`` tuples the medic has already
            rejected for this victim.  We use the wound's body_location
            *post-classification*, so here we only filter by label prefix
            match (body_location is assigned later by the caller).
        """
        mask = person_mask
        if mask is None:
            mask = self.backend.person_mask(frame_bgr, victim_bbox)

        passes_meta: List[Dict[str, Any]] = []
        try:
            props, passes_meta = self._detect_with_fallbacks(frame_bgr, victim_bbox)
            wounds, bloods = self._wrap_proposals(
                frame_bgr, victim_bbox, props, mask,
                face_bbox=face_bbox,
                session_negatives=session_negatives,
                debug_scan_id=debug_scan_id,
            )
            self._last_debug["detection_passes"] = passes_meta
        except Exception as exc:
            print(f"[wound] backend '{self.backend.name}' failed ({exc}); HSV fallback.")
            wounds, bloods = [], []

        if not wounds and not bloods:
            wounds = self._process_surface_trauma_heuristic(
                frame_bgr,
                victim_bbox,
                mask,
                face_bbox=face_bbox,
                debug_scan_id=debug_scan_id,
            )
        if not wounds and not bloods:
            bloods = self._process_color_heuristic(frame_bgr, victim_bbox, mask)

        self._finalize_debug(frame_bgr, victim_bbox, mask, debug_scan_id, wounds, bloods)

        self._annotate_blood_volumes(frame_bgr, victim_bbox, bloods)
        floor_blood = self._scan_floor_blood_pools(frame_bgr, victim_bbox, mask)
        bloods = list(bloods) + floor_blood
        return wounds, bloods

    # ------------------------------------------------------------------
    def _wrap_proposals(self,
                        frame_bgr: np.ndarray,
                        victim_bbox: Tuple[int, int, int, int],
                        proposals: List[Dict[str, Any]],
                        person_mask: Optional[np.ndarray],
                        face_bbox: Optional[Tuple[int, int, int, int]] = None,
                        session_negatives: Optional[List[Tuple[str, str]]] = None,
                        debug_scan_id: Optional[str] = None,
                        ) -> Tuple[List[WoundRegion], List[BloodRegion]]:
        import cv2
        # Normalise session negatives into a lowercase set of label tokens.
        neg_labels = set()
        if session_negatives:
            for lbl, _region in session_negatives:
                if lbl:
                    neg_labels.add(str(lbl).strip().lower())
        wounds: List[WoundRegion] = []
        bloods: List[BloodRegion] = []
        now = time.time()
        x1, y1, x2, y2 = _clip_bbox(victim_bbox, frame_bgr.shape)
        bbox_area = max(1, (x2 - x1) * (y2 - y1))

        dilated_body: Optional[np.ndarray] = None
        if person_mask is not None:
            dilated_body = _dilate_mask(person_mask, self.body_dilation_px)

        debug_records: List[Dict[str, Any]] = []
        debug_base = self._prepare_debug_dir(debug_scan_id) if self.debug_enabled and debug_scan_id else None

        for idx, p in enumerate(proposals):
            lbl = p.get("label", "")
            score = float(p.get("score", 0.0))
            box = p.get("bbox")
            dbg: Dict[str, Any] = {
                "index": idx,
                "label": lbl,
                "score": score,
                "bbox": list(box) if box else None,
                "accepted": False,
                "reason": "",
            }
            if not box:
                dbg["reason"] = "missing_bbox"
                debug_records.append(dbg)
                continue
            bx1, by1, bx2, by2 = box
            raw_area = max(0, bx2 - bx1) * max(0, by2 - by1)
            if raw_area <= 0:
                dbg["reason"] = "empty_bbox"
                debug_records.append(dbg)
                continue
            # Session-level negatives: medic already rejected this label for
            # this victim. Skip before doing any expensive work.
            if neg_labels and str(lbl).strip().lower() in neg_labels:
                dbg["reason"] = "session_negative"
                debug_records.append(dbg)
                continue
            cand_mask = p.get("mask")
            if cand_mask is None:
                # Synthesize a box-shaped mask so body gating still applies.
                cand_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
                cand_mask[by1:by2, bx1:bx2] = 255
                dbg["mask_synthesized"] = True

            is_blood_label = ("bleed" in lbl) or ("blood" in lbl)

            retained = cand_mask
            cand_area = int(cv2.countNonZero((cand_mask > 0).astype(np.uint8)))
            kept = cand_area
            ratio = 1.0 if cand_area > 0 else 0.0
            if dilated_body is not None and dilated_body.size > 0:
                retained = cv2.bitwise_and(
                    (cand_mask > 0).astype(np.uint8) * 255,
                    (dilated_body > 0).astype(np.uint8) * 255,
                )
                kept = int(cv2.countNonZero(retained))
                ratio = kept / max(cand_area, 1)
                dbg["body_overlap_ratio"] = float(ratio)
                # Body-overlap gate (slightly looser for blood drips).
                min_overlap = self.min_body_overlap
                if is_blood_label:
                    min_overlap = max(0.10, self.min_body_overlap * 0.5)
                if self.demo_mode and not is_blood_label:
                    min_overlap = max(0.06, min_overlap * 0.55)
                dbg["body_overlap_min"] = float(min_overlap)
                if ratio < min_overlap:
                    dbg["reason"] = "body_overlap_gate"
                    self._save_debug_masks(debug_base, idx, lbl, cand_mask, retained)
                    debug_records.append(dbg)
                    continue
                # Flat-paint rejection using the intersection region HSV.
                ys, xs = np.where(retained > 0)
                if len(ys) > 64:
                    ry1, ry2 = ys.min(), ys.max() + 1
                    rx1, rx2 = xs.min(), xs.max() + 1
                    region_hsv = cv2.cvtColor(frame_bgr[ry1:ry2, rx1:rx2], cv2.COLOR_BGR2HSV)
                    if is_blood_label and _is_flat_paint(region_hsv):
                        dbg["reason"] = "flat_paint_gate"
                        self._save_debug_masks(debug_base, idx, lbl, cand_mask, retained)
                        debug_records.append(dbg)
                        continue

            # --- VISUAL EVIDENCE GATE --------------------------------------
            # Compute per-detection evidence (how bloody / dark / skin-like /
            # edgy the region actually is). Reject detections whose visual
            # signal looks like plain skin with no injury artifacts — this
            # is the main discriminator that stops "critical shrapnel" from
            # firing on healthy faces and forearms.
            evidence = visual_evidence(frame_bgr, retained)
            bf, df, sf, ed = (
                evidence["blood_frac"], evidence["dark_frac"],
                evidence["skin_frac"], evidence["edge_density"],
            )
            original_lbl = lbl
            relabeled = reinterpret_diffuse_trauma_label(
                lbl,
                score,
                {
                    "blood_frac": float(bf),
                    "dark_frac": float(df),
                    "skin_frac": float(sf),
                    "edge_density": float(ed),
                    "area_ratio": float(area_ratio),
                },
                demo_mode=self.demo_mode,
            )
            if relabeled != lbl:
                lbl = canonicalize_label(relabeled)
                dbg["relabeled_from"] = original_lbl
                dbg["relabeled_to"] = lbl

            lower = lbl.lower()
            is_burn = any(k in lower for k in ("burn", "charred", "soot", "blister", "singed"))
            is_penetrate = any(k in lower for k in ("shrapnel", "fragment", "penetrat", "impaled", "puncture", "bone"))
            is_laceration = any(k in lower for k in ("laceration", "wound", "cut", "gash", "avulsion"))
            is_bruise = any(k in lower for k in ("bruise", "contusion"))
            is_abrasion = "abrasion" in lower
            is_tourniquet = "tourniquet" in lower or "dressing" in lower or "bandage" in lower
            is_amput = "amput" in lower or "stump" in lower

            # Class-specific minimum evidence. Each rule is "has at least one
            # positive signal" — a clean-skin region with high skin_frac and
            # no blood / no dark / low edges will be dropped.
            ok_evidence = False
            kept_area = kept or raw_area
            area_ratio = kept_area / max(bbox_area, 1)
            dbg["evidence"] = {
                "blood_frac": float(bf),
                "dark_frac": float(df),
                "skin_frac": float(sf),
                "edge_density": float(ed),
                "area_ratio": float(area_ratio),
            }
            if is_blood_label:
                ok_evidence = (bf >= 0.08) or (bf >= 0.04 and score >= 0.55)
            elif is_penetrate or is_laceration:
                ok_evidence = (
                    (bf >= 0.03)
                    or (df >= 0.08 and ed >= 60)
                    or (score >= 0.48 and ed >= 80)
                    or (area_ratio >= 0.020 and score >= 0.45 and ed >= 45)
                )
            elif is_burn:
                ok_evidence = (
                    (df >= 0.07)
                    or (ed >= 45 and score >= 0.45)
                    or (sf < 0.55 and score >= 0.48)
                    or (area_ratio >= 0.030 and score >= 0.42)
                )
            elif is_abrasion:
                ok_evidence = (
                    (ed >= 45 and area_ratio >= 0.015 and score >= 0.36)
                    or (sf <= 0.60 and area_ratio >= 0.020 and score >= 0.38)
                    or (df >= 0.03 and ed >= 35 and score >= 0.36)
                )
            elif is_bruise:
                ok_evidence = (
                    (df >= 0.06 and score >= 0.45)
                    or (bf >= 0.02 and score >= 0.45)
                    or (area_ratio >= 0.020 and ed >= 40 and score >= 0.42)
                )
            elif is_tourniquet or is_amput:
                # These are visually distinctive; we trust GDINO more but
                # still require confidence and at least *some* edge structure.
                ok_evidence = (
                    (score >= 0.50 and ed >= 50)
                    or (score >= 0.44 and ed >= 45 and area_ratio >= 0.025)
                )
            else:
                # Generic "wound" — require strong confidence + some color/edge evidence.
                ok_evidence = score >= 0.45 and (bf >= 0.02 or df >= 0.06 or ed >= 70 or area_ratio >= 0.025)

            if self.demo_mode:
                if is_penetrate or is_laceration:
                    ok_evidence = ok_evidence or (score >= 0.38 and (df >= 0.05 or ed >= 35 or area_ratio >= 0.015))
                elif is_burn or is_bruise or is_abrasion:
                    ok_evidence = ok_evidence or (score >= 0.36 and (df >= 0.04 or area_ratio >= 0.020))
                elif is_tourniquet or is_amput:
                    ok_evidence = ok_evidence or (score >= 0.40 and ed >= 35 and area_ratio >= 0.015)
                else:
                    ok_evidence = ok_evidence or (score >= 0.40 and area_ratio >= 0.020)

            # Final override: if the retained region is overwhelmingly plain
            # skin with no blood / dark / structure, refuse regardless of
            # what GDINO said. This kills the "critical shrapnel body — head"
            # false positive on a clean face.
            if sf >= 0.75 and bf < 0.02 and df < 0.05 and ed < 40:
                dbg["reason"] = "plain_skin_override"
                self._save_debug_masks(debug_base, idx, lbl, cand_mask, retained)
                debug_records.append(dbg)
                continue
            if not ok_evidence:
                dbg["reason"] = "visual_evidence_gate"
                self._save_debug_masks(debug_base, idx, lbl, cand_mask, retained)
                debug_records.append(dbg)
                continue

            # --- FACE-AWARE SUPPRESSION -----------------------------------
            # If InsightFace found a face and this detection's centre falls
            # inside that bbox, require strong visual evidence. Faces are
            # the highest-risk region for phantom "shrapnel / laceration"
            # detections because GDINO tends to ground generic injury
            # nouns on facial features (lips, eyes, nostrils).
            if face_bbox is not None:
                fbx1, fby1, fbx2, fby2 = face_bbox
                cx = 0.5 * (bx1 + bx2)
                cy = 0.5 * (by1 + by2)
                if fbx1 <= cx <= fbx2 and fby1 <= cy <= fby2:
                    # Accept only if we have real bleeding evidence or a
                    # pronounced darkening (burns) or an extreme-confidence
                    # penetrate.  Bruises and plain lacerations on the face
                    # need visible blood — otherwise they are almost always
                    # phantom groundings of lips / nostrils.
                    face_ok = False
                    if is_blood_label:
                        face_ok = bf >= 0.12
                    elif is_burn:
                        face_ok = df >= 0.18 or (sf < 0.30 and ed >= 90)
                    elif is_tourniquet or is_amput:
                        face_ok = False  # TQ / amputation on the face is nonsensical
                    elif is_penetrate:
                        face_ok = (bf >= 0.08 and score >= 0.55) or (df >= 0.15 and ed >= 120 and score >= 0.55)
                    elif is_laceration or is_bruise:
                        face_ok = bf >= 0.10 or (df >= 0.14 and ed >= 100 and score >= 0.55)
                    else:
                        face_ok = bf >= 0.12 and score >= 0.55
                    if not face_ok:
                        dbg["reason"] = "face_gate"
                        self._save_debug_masks(debug_base, idx, lbl, cand_mask, retained)
                        debug_records.append(dbg)
                        continue

            # Recompute bbox from the retained (gated) mask when we have one.
            gated_bbox = _mask_bbox(retained) or (bx1, by1, bx2, by2)
            kept_area = int(np.count_nonzero(retained > 0)) or raw_area

            if is_blood_label:
                bloods.append(BloodRegion(
                    area_px=kept_area,
                    bbox=gated_bbox,
                    fractional_coverage=kept_area / bbox_area,
                    last_seen=now,
                ))
            else:
                sev = estimate_wound_severity(
                    lbl, kept_area, bbox_area,
                    confidence=score, evidence=evidence,
                )
                wounds.append(WoundRegion(
                    label=lbl,
                    confidence=float(score),
                    bbox=gated_bbox,
                    mask_area_px=kept_area,
                    first_seen=now, last_seen=now,
                    severity=sev,
                    evidence={
                        "blood_frac": float(evidence.get("blood_frac", 0.0)),
                        "dark_frac": float(evidence.get("dark_frac", 0.0)),
                        "skin_frac": float(evidence.get("skin_frac", 0.0)),
                        "edge_density": float(evidence.get("edge_density", 0.0)),
                    },
                ))
            dbg["accepted"] = True
            dbg["reason"] = "accepted"
            dbg["gated_bbox"] = list(gated_bbox)
            dbg["kept_area"] = int(kept_area)
            self._save_debug_masks(debug_base, idx, lbl, cand_mask, retained)
            debug_records.append(dbg)
        self._last_debug = {
            "proposal_count": len(proposals),
            "accepted_count": sum(1 for r in debug_records if r.get("accepted")),
            "rejected_count": sum(1 for r in debug_records if not r.get("accepted")),
            "records": debug_records,
            "debug_dir": str(debug_base) if debug_base is not None else None,
            "demo_mode": bool(self.demo_mode),
        }
        self._write_debug_summary(debug_base, self._last_debug)
        for rec in debug_records:
            state = "ACCEPT" if rec.get("accepted") else "REJECT"
            print(
                f"[wound-debug] {state} label={rec.get('label')} score={rec.get('score', 0.0):.2f} "
                f"reason={rec.get('reason')} evidence={rec.get('evidence', {})}"
            )
        return wounds, bloods

    def _prepare_debug_dir(self, debug_scan_id: Optional[str]) -> Optional[Path]:
        if not self.debug_enabled or not debug_scan_id:
            return None
        path = Path(self.debug_dir) / str(debug_scan_id)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _save_debug_masks(
        self,
        debug_base: Optional[Path],
        idx: int,
        label: str,
        raw_mask: np.ndarray,
        gated_mask: np.ndarray,
    ) -> None:
        if debug_base is None:
            return
        import cv2
        safe = re.sub(r"[^a-z0-9_-]+", "_", (label or "candidate").lower()).strip("_") or "candidate"
        cv2.imwrite(str(debug_base / f"{idx:02d}_{safe}_raw.png"), (raw_mask > 0).astype(np.uint8) * 255)
        cv2.imwrite(str(debug_base / f"{idx:02d}_{safe}_gated.png"), (gated_mask > 0).astype(np.uint8) * 255)

    def _write_debug_summary(self, debug_base: Optional[Path], payload: Dict[str, Any]) -> None:
        if debug_base is None:
            return
        with open(debug_base / "summary.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _finalize_debug(
        self,
        frame_bgr: np.ndarray,
        victim_bbox: Tuple[int, int, int, int],
        person_mask: Optional[np.ndarray],
        debug_scan_id: Optional[str],
        wounds: List[WoundRegion],
        bloods: List[BloodRegion],
    ) -> None:
        if not self.debug_enabled or not debug_scan_id:
            return
        import cv2
        debug_base = self._prepare_debug_dir(debug_scan_id)
        if debug_base is None:
            return
        x1, y1, x2, y2 = _clip_bbox(victim_bbox, frame_bgr.shape)
        cv2.imwrite(str(debug_base / "frame.jpg"), frame_bgr)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size > 0:
            cv2.imwrite(str(debug_base / "victim_crop.jpg"), crop)
        if person_mask is not None and person_mask.size > 0:
            cv2.imwrite(str(debug_base / "person_mask.png"), (person_mask > 0).astype(np.uint8) * 255)
        self._last_debug["result"] = {
            "wounds": [w.label for w in wounds],
            "blood_regions": len(bloods),
            "victim_bbox": [x1, y1, x2, y2],
        }
        self._write_debug_summary(debug_base, self._last_debug)

    # ------------------------------------------------------------------
    def _process_surface_trauma_heuristic(
        self,
        frame_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        person_mask: Optional[np.ndarray],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        debug_scan_id: Optional[str] = None,
    ) -> List[WoundRegion]:
        import cv2

        x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
        if x2 <= x1 or y2 <= y1:
            return []

        h = y2 - y1
        w = x2 - x1
        bbox_area = max(1, h * w)
        crop = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if person_mask is not None and person_mask.size > 0:
            body = (person_mask[y1:y2, x1:x2] > 0).astype(np.uint8)
        else:
            body = np.ones((h, w), dtype=np.uint8)

        # Ignore the very bottom of the crop where monitor bezels / keyboards
        # often bleed into the victim silhouette during demo scans.
        body[int(h * 0.88):, :] = 0

        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = _clip_bbox(face_bbox, frame_bgr.shape)
            fx1 = max(0, fx1 - x1); fy1 = max(0, fy1 - y1)
            fx2 = min(w, fx2 - x1); fy2 = min(h, fy2 - y1)
            if fx2 > fx1 and fy2 > fy1:
                pad_x = max(6, int((fx2 - fx1) * 0.10))
                pad_y = max(6, int((fy2 - fy1) * 0.10))
                body[max(0, fy1 - pad_y):min(h, fy2 + pad_y), max(0, fx1 - pad_x):min(w, fx2 + pad_x)] = 0

        if int(body.sum()) < 128:
            return []

        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        skin_like = (((H <= 28) | (H >= 170)) & (S >= 20) & (S <= 190) & (V >= 55))
        warm = (((H <= 25) | (H >= 170)) & (S >= 35) & (V >= 45))
        darkish = ((V <= 170) & (S >= 30))
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        texture = np.abs(lap)

        body_vals = texture[body > 0]
        if body_vals.size == 0:
            return []
        texture_threshold = max(12.0, float(np.percentile(body_vals, 78)))
        candidate = (
            (body > 0)
            & (texture >= texture_threshold)
            & (skin_like | warm | darkish)
        ).astype(np.uint8) * 255
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        contours, _ = cv2.findContours(candidate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        wounds: List[WoundRegion] = []
        now = time.time()
        debug_base = self._prepare_debug_dir(debug_scan_id) if self.debug_enabled and debug_scan_id else None
        best_mask = None
        min_area_ratio = 0.009 if self.demo_mode else 0.010
        min_kept_ratio = 0.007 if self.demo_mode else 0.009
        min_density = 0.14 if self.demo_mode else 0.18

        for contour in contours:
            area = int(cv2.contourArea(contour))
            if area < max(450, int(bbox_area * min_area_ratio)):
                continue
            bx, by, bw, bh = cv2.boundingRect(contour)
            area_ratio = area / max(1, bbox_area)
            if area_ratio > 0.35:
                continue
            density = area / max(1, bw * bh)
            if density < min_density:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 7.5:
                continue

            full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
            full_mask[y1:y2, x1:x2] = cv2.drawContours(
                np.zeros((h, w), dtype=np.uint8),
                [contour],
                -1,
                255,
                thickness=-1,
            )
            if person_mask is not None and person_mask.size > 0:
                full_mask = gate_by_body(full_mask, person_mask, dilation_px=self.body_dilation_px)
            kept_area = int(np.count_nonzero(full_mask > 0))
            if kept_area < max(350, int(bbox_area * min_kept_ratio)):
                continue
            evidence = visual_evidence(frame_bgr, full_mask)
            if evidence["edge_density"] < 28:
                continue
            label = "burn" if evidence["dark_frac"] >= 0.045 or evidence["skin_frac"] <= 0.28 else "abrasion"
            confidence = 0.46 if self.demo_mode else 0.40
            sev = estimate_wound_severity(label, kept_area, bbox_area, confidence=confidence, evidence=evidence)
            wound_bbox = _mask_bbox(full_mask) or (x1 + bx, y1 + by, x1 + bx + bw, y1 + by + bh)
            wounds.append(WoundRegion(
                label=label,
                confidence=confidence,
                bbox=wound_bbox,
                mask_area_px=kept_area,
                first_seen=now,
                last_seen=now,
                severity=sev,
                evidence={
                    "blood_frac": float(evidence.get("blood_frac", 0.0)),
                    "dark_frac": float(evidence.get("dark_frac", 0.0)),
                    "skin_frac": float(evidence.get("skin_frac", 0.0)),
                    "edge_density": float(evidence.get("edge_density", 0.0)),
                },
            ))
            if best_mask is None or kept_area > int(np.count_nonzero(best_mask > 0)):
                best_mask = full_mask

        wounds.sort(key=lambda w: (w.mask_area_px, w.confidence), reverse=True)
        if debug_base is not None and best_mask is not None:
            self._save_debug_masks(debug_base, 90, "surface_trauma_fallback", best_mask, best_mask)
        if wounds:
            self._last_debug.setdefault("fallbacks", []).append({
                "name": "surface_trauma_heuristic",
                "accepted_count": len(wounds),
                "labels": [w.label for w in wounds],
            })
        return wounds[:2]

    # ------------------------------------------------------------------
    def _annotate_blood_volumes(self, frame_bgr, victim_bbox, bloods: List[BloodRegion]) -> None:
        x1, y1, x2, y2 = _clip_bbox(victim_bbox, frame_bgr.shape)
        person_h = max(1, y2 - y1)
        for b in bloods:
            ml = _px_area_to_volume_ml(float(b.area_px), float(person_h))
            b.estimated_volume_ml = ml
            b.volume_bucket = _volume_bucket(ml)

    # ------------------------------------------------------------------
    def _scan_floor_blood_pools(self,
                                frame_bgr: np.ndarray,
                                victim_bbox: Tuple[int, int, int, int],
                                person_mask: Optional[np.ndarray],
                                ) -> List[BloodRegion]:
        """Floor blood pools, **connected** to the victim silhouette.

        Anything not reachable from the dilated body mask (random red walls,
        paint, chairs) is discarded.
        """
        import cv2

        fh, fw = frame_bgr.shape[:2]
        x1, y1, x2, y2 = _clip_bbox(victim_bbox, frame_bgr.shape)
        band_start = min(fh, max(0, y2 + max(10, int((y2 - y1) * 0.03))))
        if band_start >= fh - 20:
            band_start = max(0, fh - int(fh * 0.35))
        roi = frame_bgr[band_start:fh, 0:fw]
        if roi.size == 0:
            return []

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Tighter red HSV — require higher saturation AND value floor. This
        # rejects dim indoor reds (cushions, picture frames, signage) that
        # were lighting up the "bleeding pool" chip even on clean scenes.
        m1 = cv2.inRange(hsv, np.array([0, 140, 70]), np.array([8, 255, 220]))
        m2 = cv2.inRange(hsv, np.array([172, 140, 70]), np.array([180, 255, 220]))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # Lift ROI mask into frame coordinates so we can test connectivity
        # against the body silhouette.
        full_mask = np.zeros((fh, fw), dtype=np.uint8)
        full_mask[band_start:fh, :] = mask
        if person_mask is not None and person_mask.size > 0:
            full_mask = _connected_to_body(full_mask, person_mask, dilation_px=22)
        else:
            # Without a silhouette we refuse to emit floor pools — too many
            # false positives (this is the root cause of the red-wall bug).
            return []

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        now = time.time()
        person_area = max(1, (x2 - x1) * (y2 - y1))
        person_h = max(1, y2 - y1)
        out: List[BloodRegion] = []

        # A real pool is (a) large, (b) solid (contour vs hull), (c) not a
        # tight line (posters/tape), and (d) has red pixels that aren't
        # flat paint. Each pool must satisfy *all* four to be emitted.
        for c in contours:
            area = int(cv2.contourArea(c))
            if area < 2000:  # was 400 — far too permissive
                continue
            hull = cv2.convexHull(c)
            hull_area = max(1, int(cv2.contourArea(hull)))
            solidity = area / hull_area
            if solidity < 0.55:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # Aspect-ratio sanity — skinny rectangles are tape, cords, paint lines.
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 6.0:
                continue
            # Drop detections that overlap heavily with the victim bbox — those
            # are already covered by on-body wound/blood entries.
            bbox_g = (bx, by, bx + bw, by + bh)
            if _iou(bbox_g, (x1, y1, x2, y2)) > 0.25:
                continue
            # Flat-paint rejection on the contour crop.
            crop_hsv = hsv_for(frame_bgr, bbox_g)
            if crop_hsv is not None and _is_flat_paint(crop_hsv):
                continue
            # Density check: at least 60 % of the bbox area should actually
            # be red pixels, otherwise it's mostly background noise.
            density = area / max(1, bw * bh)
            if density < 0.35:
                continue
            ml = _px_area_to_volume_ml(float(area), float(person_h))
            out.append(BloodRegion(
                area_px=area,
                bbox=bbox_g,
                fractional_coverage=area / person_area,
                last_seen=now,
                estimated_volume_ml=ml,
                volume_bucket=_volume_bucket(ml),
                is_floor_pool=True,
            ))
        return out

    # ------------------------------------------------------------------
    # Color heuristic fallback (no ML, still body-gated)
    # ------------------------------------------------------------------
    def _process_color_heuristic(self,
                                 frame_bgr: np.ndarray,
                                 bbox: Tuple[int, int, int, int],
                                 person_mask: Optional[np.ndarray],
                                 ) -> List[BloodRegion]:
        import cv2
        x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
        if x2 <= x1 or y2 <= y1:
            return []

        crop = frame_bgr[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # Tighter HSV (matches the floor-pool detector). Without this the
        # heuristic paints every skin shadow as blood.
        m1 = cv2.inRange(hsv, np.array([0, 140, 70]), np.array([8, 255, 220]))
        m2 = cv2.inRange(hsv, np.array([172, 140, 70]), np.array([180, 255, 220]))
        mask = cv2.bitwise_or(m1, m2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # Promote to full-frame coords so body gating works uniformly.
        full_mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask
        if person_mask is not None and person_mask.size > 0:
            full_mask = gate_by_body(full_mask, person_mask, dilation_px=self.body_dilation_px)
        else:
            # Without a silhouette, stay extremely strict: only allow tight
            # regions in the centre 60% of the bbox where a body should sit.
            cy1 = y1 + int((y2 - y1) * 0.1)
            cy2 = y1 + int((y2 - y1) * 0.9)
            cx1 = x1 + int((x2 - x1) * 0.2)
            cx2 = x1 + int((x2 - x1) * 0.8)
            centre = np.zeros_like(full_mask)
            centre[cy1:cy2, cx1:cx2] = 255
            full_mask = cv2.bitwise_and(full_mask, centre)

        contours, _ = cv2.findContours(full_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes: List[BloodRegion] = []
        bbox_area = max(1, (x2 - x1) * (y2 - y1))
        now = time.time()
        for c in contours:
            area = int(cv2.contourArea(c))
            # Require a meaningful chunk of red — not a single shadow pixel.
            if area < 600:
                continue
            bx, by, bw, bh = cv2.boundingRect(c)
            # Density + solidity check — same idea as floor-blood: a real
            # on-body bleed is a solid shape, not a scattering.
            density = area / max(1, bw * bh)
            if density < 0.4:
                continue
            aspect = max(bw, bh) / max(1, min(bw, bh))
            if aspect > 6.0:
                continue
            bboxes.append(BloodRegion(
                area_px=area,
                bbox=(bx, by, bx + bw, by + bh),
                fractional_coverage=area / bbox_area,
                last_seen=now,
            ))
        return bboxes


# ---------------------------------------------------------------------------
# misc helpers
# ---------------------------------------------------------------------------

def _clip_bbox(bbox: Tuple[int, int, int, int], shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    h, w = shape[:2]
    x1, y1, x2, y2 = bbox
    return (max(0, int(x1)), max(0, int(y1)), min(int(w), int(x2)), min(int(h), int(y2)))


def _first_mask(masks) -> Optional[np.ndarray]:
    if masks is None:
        return None
    arr = np.asarray(masks)
    if arr.size == 0:
        return None
    if arr.ndim == 3:
        return arr[0]
    return arr


def _mask_bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    if mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def hsv_for(frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    import cv2
    x1, y1, x2, y2 = _clip_bbox(bbox, frame_bgr.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    return cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
