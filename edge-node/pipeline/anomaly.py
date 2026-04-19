"""DINOv3 patch-feature anomaly prior.

Idea: for each candidate wound, compare its DINOv3 patch embeddings against
a *victim-specific* baseline sampled from presumed-healthy skin patches on
the same person.  Detections whose patch features look indistinguishable
from the baseline skin are likely phantom groundings and can be downgraded
(or rejected outright).

Why on the same victim? Because skin color, clothing, lighting and
camera-specific noise dominate most global "skin vs. wound" classifiers.
By anchoring against the **same individual's** healthy patches we cancel
those nuisance variables and isolate the actual visual anomaly.

This is intentionally conservative: we only *downgrade* or *reject*,
never upgrade — the primary decision still comes from GDINO/SAM3 with
the visual-evidence gate.  The prior adds an extra safety layer that only
lights up when the candidate region is visually boring compared to the
rest of the victim.

The module degrades gracefully: if ``transformers`` or the preferred
checkpoint isn't available it tries **public fallbacks** (e.g. DINOv2
base) when configured; if nothing loads, :py:meth:`score_candidates`
returns an empty dict.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


def _one_line_exc(exc: Exception, limit: int = 160) -> str:
    """Strip HF stack traces — keep the first line for console readability."""
    msg = str(exc).strip().replace("\r", " ")
    line = msg.split("\n")[0].strip()
    if len(line) > limit:
        line = line[: limit - 3] + "..."
    low = line.lower()
    if "401" in line or "gated" in low or "restricted" in low:
        return (
            "Hugging Face gated / unauthenticated "
            "(accept the model license, then `huggingface-cli login` or set HF_TOKEN)"
        )
    return line


# Keypoint indices (COCO 17-pt) that usually sit on clean skin.
_BASELINE_KPTS = [5, 6, 7, 8, 11, 12]   # shoulders, elbows, hips


@dataclass
class AnomalyScore:
    """Per-candidate anomaly signal.

    ``score`` is the mean cosine distance between the candidate's patch
    embeddings and the baseline; ``z`` is the z-score against an
    intra-baseline distance distribution (how many sigmas the candidate is
    away from the "boring skin" manifold).  Decisions rely on z rather
    than raw distance so a victim with patterned clothing doesn't flip
    the threshold.
    """

    score: float
    z: float
    baseline_samples: int


class DinoV3AnomalyPrior:
    """Patch-feature anomaly detector (DINOv3-style, DINOv2-compatible).

    Tries the primary HF checkpoint first, then optional **fallback** ids
    (public ``facebook/dinov2-base`` works without a gated login). Check
    ``ready`` after construction.
    """

    def __init__(self,
                 checkpoint: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
                 fallbacks: Optional[Union[str, Sequence[str]]] = None,
                 patch_size: int = 16,
                 device: Optional[str] = None) -> None:
        self.checkpoint = checkpoint
        self._primary_checkpoint = checkpoint
        self.patch_size = patch_size
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None
        self.ready = False

        fb_list: List[str] = []
        if isinstance(fallbacks, str):
            fb_list = [fallbacks]
        elif fallbacks:
            fb_list = [str(x) for x in fallbacks if x]
        candidates: List[str] = [checkpoint]
        for fid in fb_list:
            if fid and fid not in candidates:
                candidates.append(fid)

        last_exc: Optional[BaseException] = None
        try:
            import torch  # type: ignore
            from transformers import AutoImageProcessor, AutoModel  # type: ignore
            self._torch = torch
            self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:
            print(f"[anomaly] patch prior disabled (torch/transformers: {_one_line_exc(exc)}).")
            return

        for ckpt in candidates:
            try:
                proc = AutoImageProcessor.from_pretrained(ckpt)
                model = AutoModel.from_pretrained(ckpt).to(self._device).eval()
                self._processor = proc
                self._model = model
                self.checkpoint = ckpt
                self.ready = True
                if ckpt == self._primary_checkpoint:
                    print(f"[anomaly] patch prior active ({ckpt} on {self._device}).")
                else:
                    print(
                        f"[anomaly] patch prior active ({ckpt} on {self._device}) "
                        f"[fallback — primary `{self._primary_checkpoint}` unavailable]."
                    )
                return
            except Exception as exc:
                last_exc = exc

        tried = ", ".join(candidates)
        print(
            f"[anomaly] patch prior disabled (last error: {_one_line_exc(last_exc) if last_exc else 'unknown'}). "
            f"Tried: {tried}"
        )

    # ------------------------------------------------------------------
    def _embed_grid(self, img_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Return per-patch token embeddings (H_p x W_p x D).

        We use the last hidden state, stripping CLS/register tokens. This
        gives us a spatial grid we can index by pixel bbox later.
        """
        if not self.ready:
            return None
        import cv2
        torch = self._torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        hs = outputs.last_hidden_state  # (1, T, D)
        feats = hs[0].detach().float().cpu().numpy()  # (T, D)
        # DINOv3 adds 1 CLS + usually 4 register tokens; strip any token count
        # that isn't a perfect square.
        h, w = rgb.shape[:2]
        # Infer the patch grid size from the actual model config when possible.
        try:
            target_size = int(getattr(self._model.config, "image_size", 224))
            patch_size = int(getattr(self._model.config, "patch_size", self.patch_size))
            grid = target_size // patch_size
            expected = grid * grid
        except Exception:
            grid = int(np.sqrt(feats.shape[0]))
            expected = grid * grid
        prefix = feats.shape[0] - expected
        if prefix < 0:
            return None
        grid_feats = feats[prefix:]
        return grid_feats.reshape(grid, grid, -1), (h, w)

    # ------------------------------------------------------------------
    def score_candidates(self,
                         frame_bgr: np.ndarray,
                         victim_bbox: Tuple[int, int, int, int],
                         keypoints: Optional[List[Tuple[float, float, float]]],
                         wound_bboxes: List[Tuple[int, int, int, int]],
                         ) -> Dict[int, AnomalyScore]:
        """Return ``{wound_idx: AnomalyScore}`` for a set of wound bboxes.

        Empty dict on any failure or when the prior is inactive.
        """
        if not self.ready or not wound_bboxes:
            return {}

        x1, y1, x2, y2 = victim_bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(frame_bgr.shape[1], int(x2)), min(frame_bgr.shape[0], int(y2))
        if x2 <= x1 or y2 <= y1:
            return {}
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return {}

        try:
            grid_feats, (ch, cw) = self._embed_grid(crop)
        except Exception as exc:
            print(f"[anomaly] embed failed ({exc})")
            return {}
        if grid_feats is None:
            return {}

        gh, gw, _ = grid_feats.shape
        ch = max(1, ch)
        cw = max(1, cw)
        sy = gh / ch
        sx = gw / cw

        def _crop_box_to_grid(px1: int, py1: int, px2: int, py2: int) -> Tuple[int, int, int, int]:
            # Convert frame-coords bbox to crop-coords, then to grid-coords.
            cx1 = max(0, int(px1) - x1)
            cy1 = max(0, int(py1) - y1)
            cx2 = min(cw, int(px2) - x1)
            cy2 = min(ch, int(py2) - y1)
            if cx2 <= cx1 or cy2 <= cy1:
                return 0, 0, 0, 0
            gx1 = int(np.floor(cx1 * sx))
            gy1 = int(np.floor(cy1 * sy))
            gx2 = int(np.ceil(cx2 * sx))
            gy2 = int(np.ceil(cy2 * sy))
            return (
                max(0, min(gw - 1, gx1)),
                max(0, min(gh - 1, gy1)),
                max(1, min(gw, gx2)),
                max(1, min(gh, gy2)),
            )

        # --- Build baseline from presumed-healthy keypoint neighborhoods ---
        baseline_vecs: List[np.ndarray] = []
        body_w = max(1, x2 - x1)
        body_h = max(1, y2 - y1)
        radius = max(8, int(0.08 * max(body_w, body_h)))
        for idx in _BASELINE_KPTS:
            if not keypoints or idx >= len(keypoints):
                continue
            kx, ky, kconf = keypoints[idx]
            if kconf < 0.3:
                continue
            px1, py1 = int(kx) - radius, int(ky) - radius
            px2, py2 = int(kx) + radius, int(ky) + radius
            gx1, gy1, gx2, gy2 = _crop_box_to_grid(px1, py1, px2, py2)
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            patch_block = grid_feats[gy1:gy2, gx1:gx2].reshape(-1, grid_feats.shape[-1])
            if patch_block.size == 0:
                continue
            baseline_vecs.append(patch_block)

        if not baseline_vecs:
            # Fall back to a grid of "torso" patches if keypoints missing.
            cx = gw // 2
            cy = int(gh * 0.55)
            r = max(1, min(gh, gw) // 6)
            patch_block = grid_feats[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
            patch_block = patch_block.reshape(-1, grid_feats.shape[-1])
            if patch_block.size:
                baseline_vecs.append(patch_block)

        if not baseline_vecs:
            return {}

        baseline = np.concatenate(baseline_vecs, axis=0)
        baseline = baseline / (np.linalg.norm(baseline, axis=1, keepdims=True) + 1e-8)
        baseline_mean = baseline.mean(axis=0)
        baseline_mean /= (np.linalg.norm(baseline_mean) + 1e-8)

        # Intra-baseline distance distribution (cosine distance from mean).
        intra = 1.0 - baseline @ baseline_mean
        intra_mean = float(intra.mean())
        intra_std = float(intra.std() + 1e-6)

        # --- Score each wound candidate -----------------------------------
        out: Dict[int, AnomalyScore] = {}
        for i, wbb in enumerate(wound_bboxes):
            gx1, gy1, gx2, gy2 = _crop_box_to_grid(*wbb)
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            patches = grid_feats[gy1:gy2, gx1:gx2].reshape(-1, grid_feats.shape[-1])
            if patches.size == 0:
                continue
            patches_n = patches / (np.linalg.norm(patches, axis=1, keepdims=True) + 1e-8)
            dists = 1.0 - patches_n @ baseline_mean
            score = float(dists.mean())
            z = (score - intra_mean) / intra_std
            out[i] = AnomalyScore(score=score, z=float(z),
                                  baseline_samples=int(baseline.shape[0]))
        return out
