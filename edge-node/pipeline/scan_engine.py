"""Scan engine — capture a frozen per-victim assessment.

A ``ScanEngine.capture(frame, victim_bbox, keypoints)`` call runs the full
perception stack once, wraps the result in a :class:`ScanRecord`, and stores
the original frame plus the victim crop in :mod:`pipeline.scan_store` so the
dashboard can fetch them over HTTP.

The engine is intentionally stateless between scans: it doesn't update
tracking or rPPG buffers. The caller passes in the already-running
``WoundSegmenter`` / ``FaceReID`` / ``BodyLocator`` / ``RppgEstimator`` so
model weights load only once per process.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from state.victim import ScanRecord, Vitals, WoundRegion
from state.priority import derive_priority
from state.tccc import scan_transcript
from .scan_store import store as scan_store


# Human-readable body-region map (shared with body_location labels).
_REGION_PRETTY = {
    "head": "head", "face": "face", "neck": "neck",
    "left_shoulder": "left shoulder", "right_shoulder": "right shoulder",
    "left_arm": "left arm", "right_arm": "right arm",
    "left_forearm": "left forearm", "right_forearm": "right forearm",
    "left_hand": "left hand", "right_hand": "right hand",
    "chest": "chest", "abdomen": "abdomen", "pelvis": "pelvis",
    "left_torso": "left torso", "right_torso": "right torso",
    "left_thigh": "left thigh", "right_thigh": "right thigh",
    "left_leg": "left leg", "right_leg": "right leg",
    "left_foot": "left foot", "right_foot": "right foot",
    "unknown": "body",
}


def _pretty_region(loc: str) -> str:
    return _REGION_PRETTY.get(loc or "unknown", (loc or "body").replace("_", " "))


def _wound_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1)) * max(1, (ay2 - ay1))
    area_b = max(1, (bx2 - bx1)) * max(1, (by2 - by1))
    return inter / float(area_a + area_b - inter)


_SEVERITY_RANK = {"critical": 4, "serious": 3, "moderate": 2, "minor": 1, "possible": 0, "unknown": 0}


def _two_shot_consensus(
    per_frame: List[List[WoundRegion]],
    iou_thresh: float = 0.3,
) -> Dict[str, Any]:
    """Keep detections that appear in >=2 frames, mark singletons possible.

    Returns ``{"wounds": [...], "agreement": {label: count}}``.
    Used by the ``max`` profile to give the medic the strongest possible
    corroboration signal: a wound GDINO stabilises across multiple frames
    is far more likely to be real than a one-shot positive.
    """
    if not per_frame:
        return {"wounds": [], "agreement": {}}
    # Everything survives if only 1 frame was processed.
    if len(per_frame) < 2:
        return {"wounds": list(per_frame[0]), "agreement": {}}

    primary = per_frame[0]
    others = per_frame[1:]
    kept: List[WoundRegion] = []
    agreement: Dict[str, int] = {}
    for w in primary:
        matches = 1
        for frame_list in others:
            for other in frame_list:
                if other.label != w.label:
                    continue
                if _wound_iou(w.bbox, other.bbox) >= iou_thresh:
                    matches += 1
                    # Also take the max confidence as the consensus confidence.
                    w.confidence = max(float(w.confidence or 0.0),
                                       float(other.confidence or 0.0))
                    break
        agreement[w.label] = max(agreement.get(w.label, 0), matches)
        if matches >= 2:
            kept.append(w)
        else:
            # Singleton: downgrade to "possible" rather than drop entirely —
            # the medic can still decide via thumbs up/down.
            w.severity = "possible"
            kept.append(w)
    # Also add wounds that only appeared in non-primary frames (secondary
    # consensus across those frames), downgraded to possible.
    for i, frame_list in enumerate(others):
        for other in frame_list:
            already = any(
                k.label == other.label and _wound_iou(k.bbox, other.bbox) >= iou_thresh
                for k in kept
            )
            if already:
                continue
            other.severity = "possible"
            kept.append(other)
            agreement[other.label] = max(agreement.get(other.label, 0), 1)
    return {"wounds": kept, "agreement": agreement}


def _merge_wound_frames(
    per_frame: List[List[WoundRegion]],
    iou_thresh: float = 0.4,
) -> List[WoundRegion]:
    """Dedup wound detections across a sweep using IoU + label match.

    Keeps the variant with the highest severity, breaking ties by confidence.
    """
    merged: List[WoundRegion] = []
    for frame_wounds in per_frame:
        for w in frame_wounds:
            paired = None
            for m in merged:
                if m.label != w.label:
                    continue
                if _wound_iou(m.bbox, w.bbox) >= iou_thresh:
                    paired = m
                    break
            if paired is None:
                merged.append(w)
                continue
            # Keep the more severe / higher-confidence detection.
            if (
                _SEVERITY_RANK.get(w.severity, 0) > _SEVERITY_RANK.get(paired.severity, 0)
                or (
                    _SEVERITY_RANK.get(w.severity, 0) == _SEVERITY_RANK.get(paired.severity, 0)
                    and float(w.confidence or 0) > float(paired.confidence or 0)
                )
            ):
                idx = merged.index(paired)
                merged[idx] = w
    return merged


def _encode_jpeg(img: np.ndarray, quality: int = 85) -> bytes:
    import cv2
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def _wound_to_scan_dict(w: WoundRegion, frame_shape: Tuple[int, int]) -> Dict[str, Any]:
    h, w_ = frame_shape
    x1, y1, x2, y2 = w.bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    # Margin label goes to the nearer vertical edge, a bit above the wound.
    margin_x = 12 if cx < w_ / 2 else max(12, w_ - 12)
    margin_y = max(24, min(h - 12, int(cy)))
    return {
        "label": w.label,
        "severity": w.severity,
        "body_region": w.body_location,
        "bbox": list(w.bbox),
        "arrow_anchor": [int(cx), int(cy)],
        "label_anchor": [int(margin_x), int(margin_y)],
        "confidence": float(w.confidence or 0.0),
        "confirmation": getattr(w, "confirmation", "pending"),
        "evidence": dict(getattr(w, "evidence", {}) or {}),
    }


class ScanEngine:
    def __init__(self,
                 wound_segmenter,
                 body_locator,
                 rppg=None,
                 face_reid=None,
                 llm=None,
                 scenarios: Optional[Dict[str, Any]] = None,
                 burn_estimator=None,
                 anomaly_prior=None,
                 consensus_enabled: bool = False,
                 ensemble_enabled: bool = False) -> None:
        self.wound_segmenter = wound_segmenter
        self.body = body_locator
        self.rppg = rppg
        self.face = face_reid
        self.llm = llm
        self.scenarios = scenarios or {}
        self.burn_estimator = burn_estimator
        # Optional DINOv3 patch-anomaly prior (see pipeline.anomaly).
        self.anomaly_prior = anomaly_prior
        # max-profile flags: two-shot frame consensus + strict second-pass.
        self.consensus_enabled = bool(consensus_enabled)
        self.ensemble_enabled = bool(ensemble_enabled)
        # Per-victim session negatives the medic has rejected.
        self._session_negatives: Dict[str, List[Tuple[str, str]]] = {}

    # ------------------------------------------------------------------
    def add_session_negative(self, victim_id: str, label: str, body_region: str) -> None:
        """Remember that the medic rejected (label, body_region) for victim.

        The tuple is consulted on the next scan and propagated to
        :meth:`WoundSegmenter.process` so the same phantom detection is
        suppressed at the detection layer rather than just hidden in UI.
        """
        if not victim_id or not label:
            return
        key = (str(label).strip().lower(), str(body_region or "").strip().lower())
        bucket = self._session_negatives.setdefault(victim_id, [])
        if key not in bucket:
            bucket.append(key)

    # ------------------------------------------------------------------
    def capture(self,
                frame_bgr: np.ndarray,
                victim_bbox: Tuple[int, int, int, int],
                keypoints: Optional[List[Tuple[float, float, float]]] = None,
                victim_id: Optional[str] = None,
                scenario_id: str = "combat_blast",
                transcript_snippet: str = "",
                frame_provider: Optional[Callable[[], Optional[np.ndarray]]] = None,
                sweep_duration_sec: float = 0.0,
                sweep_samples: int = 5,
                progress_cb: Optional[Callable[[int, int, str], None]] = None,
                ) -> Tuple[ScanRecord, Dict[str, Any]]:
        """Run a per-victim assessment, optionally sweeping across multiple
        frames to emulate a head-to-toe body scan.

        Parameters
        ----------
        frame_provider:
            Callable returning the latest BGR frame. If supplied together with
            ``sweep_duration_sec > 0`` the engine samples ``sweep_samples``
            frames evenly across the window, unions wound detections via IoU
            dedup, and picks the frame with the richest signal as the
            canonical crop/face reference.
        progress_cb:
            Optional ``fn(step, total, phase)`` hook the caller can use to
            emit ``scan_progress`` WS events during the sweep.

        Returns ``(scan_record, extras)``.
        """
        import cv2

        def _emit(step: int, total: int, phase: str) -> None:
            if progress_cb is not None:
                try:
                    progress_cb(step, total, phase)
                except Exception:
                    pass

        # Build the list of frames that make up the sweep. We only *sample*
        # frames here — the heavy wound pipeline (GDINO + SAM) runs exactly
        # once on the sharpest frame below. Additional frames are cheap
        # references used for face re-ID and vitals.
        sweep_frames: List[np.ndarray] = [frame_bgr]
        if frame_provider is not None and sweep_duration_sec > 0 and sweep_samples > 1:
            extra = max(0, int(sweep_samples) - 1)
            interval = float(sweep_duration_sec) / max(1, extra)
            for i in range(extra):
                _emit(i + 1, sweep_samples, "sweeping")
                time.sleep(max(0.0, interval))
                try:
                    nxt = frame_provider()
                except Exception:
                    nxt = None
                if nxt is not None and getattr(nxt, "size", 0) > 0:
                    sweep_frames.append(nxt)
        _emit(len(sweep_frames), max(1, sweep_samples), "analyzing")

        # Pick the sharpest frame to run the expensive vision models on.
        # Laplacian variance is a cheap proxy for focus — this is standard
        # practice for keyframe selection.
        def _sharpness(img: np.ndarray) -> float:
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Subsample to keep the cost negligible even on 1080p.
                small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5) if gray.size > 640 * 360 else gray
                return float(cv2.Laplacian(small, cv2.CV_32F).var())
            except Exception:
                return 0.0

        canonical = max(sweep_frames, key=_sharpness) if len(sweep_frames) > 1 else sweep_frames[0]
        h, w = canonical.shape[:2]
        x1, y1, x2, y2 = victim_bbox
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))

        # --- Face bbox (for face-aware wound suppression) -------------------
        # We resolve the face *before* running wound segmentation so the
        # gate knows to reject GDINO detections that ground on lips/eyes.
        face_bbox_frame: Optional[Tuple[int, int, int, int]] = None
        if self.face is not None and getattr(self.face, "ready", False):
            try:
                _emb, _crop, fbb = self.face.embed_person(canonical, (x1, y1, x2, y2))
                if fbb is not None:
                    face_bbox_frame = fbb
            except Exception as exc:
                print(f"[scan] face pre-detect failed ({exc})")

        # --- Session negatives from the medic's prior rejections ------------
        neg_pairs = list(self._session_negatives.get(victim_id or "", []))

        # --- Person silhouette (used for body-gating wounds + blood) --------
        person_mask = self.wound_segmenter.person_mask(canonical, (x1, y1, x2, y2))

        # --- Wound + blood segmentation (runs ONCE on the canonical frame) --
        # Running GDINO+SAM per sweep frame was ~5× slower without meaningful
        # recall gains, because the subject barely moves between samples at
        # 30fps over 0.8s. If the best frame finds nothing, we retry once
        # more on the next-sharpest fallback frame.
        wounds: List[WoundRegion] = []
        bloods: List[Any] = []
        candidates = sorted(sweep_frames, key=_sharpness, reverse=True)
        debug_token = uuid.uuid4().hex[:8]
        # In `max` profile with two-shot consensus we run the heavy
        # pipeline on the TOP-2 sharpest frames and require detections to
        # appear in both (IoU >= 0.3, same label).  Otherwise we stop on
        # the first frame that yields findings (fast path).
        per_frame_wounds: List[List[WoundRegion]] = []
        per_frame_bloods: List[List[Any]] = []
        per_frame_refs: List[np.ndarray] = []
        max_heavy_passes = 2 if self.consensus_enabled else 2
        for attempt, sf in enumerate(candidates[:max_heavy_passes]):
            try:
                pm = (
                    person_mask
                    if sf is canonical
                    else self.wound_segmenter.person_mask(sf, (x1, y1, x2, y2))
                )
                # Face bbox for this frame (mostly the same; ~no motion).
                fbb_for_sf = face_bbox_frame
                if sf is not canonical and self.face is not None and getattr(self.face, "ready", False):
                    try:
                        _e, _c, fbb2 = self.face.embed_person(sf, (x1, y1, x2, y2))
                        if fbb2 is not None:
                            fbb_for_sf = fbb2
                    except Exception:
                        pass
                w_list, b_list = self.wound_segmenter.process(
                    sf, (x1, y1, x2, y2),
                    person_mask=pm,
                    face_bbox=fbb_for_sf,
                    session_negatives=neg_pairs,
                    debug_scan_id=f"{victim_id or 'scan'}-{debug_token}-a{attempt}",
                )
            except Exception as exc:
                print(f"[scan] wound segmentation failed on attempt {attempt} ({exc})")
                w_list, b_list = [], []

            per_frame_wounds.append(w_list)
            per_frame_bloods.append(b_list)
            per_frame_refs.append(sf)

            if not self.consensus_enabled:
                # Fast path: first non-empty frame wins, same as before.
                if w_list or b_list:
                    wounds, bloods = w_list, b_list
                    canonical = sf
                    h, w = canonical.shape[:2]
                    if sf is not candidates[0]:
                        person_mask = pm
                    break
                if attempt == 0 and len(candidates) > 1:
                    continue

        consensus_meta: Dict[str, Any] = {}
        ensemble_meta: Dict[str, Any] = {}
        if self.consensus_enabled and per_frame_wounds:
            # Two-shot consensus: keep detections that appear in at least
            # two of the scanned frames with IoU >= 0.3 and label match.
            # Singletons are downgraded to "possible" rather than dropped.
            consensus_meta = _two_shot_consensus(per_frame_wounds)
            wounds = consensus_meta.get("wounds", [])
            # Blood regions come from the sharpest frame because consensus
            # for pooling is rarely meaningful (blood moves).
            bloods = per_frame_bloods[0] if per_frame_bloods else []
            canonical = per_frame_refs[0]
            h, w = canonical.shape[:2]

        # --- Ensemble gate: strict-threshold second pass on criticals ------
        # In `max` profile we re-run GDINO at (0.55 / 0.45) on the same
        # frame. Critical-class findings that DON'T survive the strict pass
        # are demoted to 'serious' — this protects against single-shot
        # over-confident detections on noisy / blurry frames.
        if self.ensemble_enabled and wounds and hasattr(self.wound_segmenter, "strict_survivors"):
            try:
                strict = self.wound_segmenter.strict_survivors(
                    canonical, (x1, y1, x2, y2),
                    box_threshold=0.55, text_threshold=0.45,
                )
            except Exception as exc:
                print(f"[scan] strict ensemble pass failed ({exc})")
                strict = []
            def _survives_strict(wr: WoundRegion) -> bool:
                wr_label = (wr.label or "").lower()
                for s in strict:
                    sl = str(s.get("label") or "").lower()
                    if sl != wr_label:
                        continue
                    if _wound_iou(wr.bbox, tuple(s.get("bbox", (0, 0, 0, 0)))) >= 0.3:
                        return True
                return False
            demoted = 0
            for wr in wounds:
                if (wr.severity or "").lower() in ("critical",):
                    if not _survives_strict(wr):
                        wr.severity = "serious"
                        demoted += 1
            ensemble_meta = {"strict_pass_hits": len(strict), "demoted_to_serious": demoted}

        # --- DINOv3 patch anomaly prior (victim-specific baseline) ---------
        # If a candidate's patch features are *statistically identical* to
        # the baseline healthy-skin patches on the same victim (z < 1.5),
        # there's no visual anomaly to support the detection — downgrade
        # to "possible" or drop outright if GDINO was already borderline.
        anomaly_meta: Dict[str, Any] = {}
        if self.anomaly_prior is not None and getattr(self.anomaly_prior, "ready", False) and wounds:
            try:
                scores = self.anomaly_prior.score_candidates(
                    canonical, (x1, y1, x2, y2),
                    keypoints or [],
                    [tuple(wr.bbox) for wr in wounds],
                )
            except Exception as exc:
                print(f"[scan] anomaly prior failed ({exc})")
                scores = {}
            downgraded = 0
            dropped = 0
            survivors: List[WoundRegion] = []
            for i, wr in enumerate(wounds):
                s = scores.get(i)
                if s is None:
                    survivors.append(wr)
                    continue
                wr.evidence = dict(wr.evidence or {})
                wr.evidence["anomaly_z"] = float(s.z)
                wr.evidence["anomaly_score"] = float(s.score)
                # Strongly anomalous: let it pass (even promote weak ones).
                if s.z >= 2.0:
                    survivors.append(wr)
                    continue
                # Mildly anomalous: keep but note.
                if s.z >= 1.2:
                    survivors.append(wr)
                    continue
                # Looks like normal skin for this individual — demote / drop.
                if (wr.severity or "").lower() in ("critical", "serious"):
                    wr.severity = "possible"
                    survivors.append(wr)
                    downgraded += 1
                else:
                    # Already weak and also non-anomalous — drop.
                    dropped += 1
            wounds = survivors
            anomaly_meta = {
                "scored": len(scores),
                "downgraded": downgraded,
                "dropped": dropped,
            }

        # Attach body locations using the existing BodyLocator.
        for wound in wounds:
            try:
                wound.body_location = self.body.locate(wound.bbox, (x1, y1, x2, y2), keypoints or [])
            except Exception:
                wound.body_location = "unknown"

        _emit(len(sweep_frames), max(1, sweep_samples), "face")

        # --- Face embedding + thumbnail (iterate sweep frames until hit) ----
        face_embedding: Optional[List[float]] = None
        face_thumb_jpeg: Optional[bytes] = None
        if self.face is not None and getattr(self.face, "ready", False):
            for sf in [canonical] + [f for f in sweep_frames if f is not canonical]:
                try:
                    emb, face_crop, _face_bbox = self.face.embed_person(sf, (x1, y1, x2, y2))
                except Exception as exc:
                    print(f"[scan] face embed failed ({exc})")
                    emb, face_crop = None, None
                if emb is not None:
                    face_embedding = [float(v) for v in emb.tolist()]
                if face_crop is not None and face_crop.size > 0:
                    face_thumb_jpeg = _encode_jpeg(face_crop, quality=80)
                if emb is not None and face_crop is not None:
                    break

        # --- rPPG: take the last estimate for this victim if available ------
        vitals_dict: Dict[str, Any] = {"hr": None, "rr": None, "spo2": None}
        if self.rppg is not None and victim_id:
            try:
                # rppg.process is called each frame with the face bbox. If we
                # don't have a fresh sample we still surface the buffered one.
                face_h = int((y2 - y1) * 0.35)
                face_box = (x1, y1, x2, y1 + max(40, face_h))
                latest = self.rppg.process(canonical, [(victim_id, face_box)])
                est = latest.get(victim_id)
                if est:
                    vitals_dict.update({
                        "hr": est.hr, "rr": est.rr,
                        "hr_confidence": est.hr_confidence,
                        "rr_confidence": est.rr_confidence,
                        "last_updated": est.timestamp or time.time(),
                    })
            except Exception as exc:
                print(f"[scan] rPPG failed ({exc})")

        # --- MARCH + SALT (reuse existing state modules) --------------------
        from state.march import derive_march
        from state.salt import suggest_salt
        from state.victim import Victim, SaltTag

        temp_victim = Victim(id=victim_id or f"scan-{uuid.uuid4().hex[:6]}")
        temp_victim.bbox = (x1, y1, x2, y2)
        temp_victim.keypoints = keypoints or []
        temp_victim.wound_regions = wounds
        temp_victim.blood_regions = bloods
        temp_victim.vitals = Vitals(
            hr=vitals_dict.get("hr"),
            rr=vitals_dict.get("rr"),
            hr_confidence=vitals_dict.get("hr_confidence", 0.0),
            rr_confidence=vitals_dict.get("rr_confidence", 0.0),
        )
        temp_victim.transcript = transcript_snippet or ""

        scen = self.scenarios.get(scenario_id, {})
        if scen.get("estimate_burn_percent") and self.burn_estimator is not None:
            try:
                temp_victim.tbsa_burn_percent = self.burn_estimator(wounds)
            except Exception:
                temp_victim.tbsa_burn_percent = None

        march_state = derive_march(temp_victim, scen)
        march_dict = march_state.to_dict()
        suggestion = suggest_salt(temp_victim, march_state)
        tag = suggestion.tag
        if tag in (SaltTag.GREY, SaltTag.BLACK):
            tag = SaltTag.UNTAGGED

        priority = derive_priority(tag.value, wounds, march_dict)
        codewords = scan_transcript(transcript_snippet or "")

        # --- Persist JPEG artifacts ----------------------------------------
        _emit(len(sweep_frames), max(1, sweep_samples), "finalizing")
        scan_id = uuid.uuid4().hex[:12]
        frame_jpeg = _encode_jpeg(canonical, quality=82)
        crop_img = canonical[y1:y2, x1:x2]
        crop_jpeg = _encode_jpeg(crop_img, quality=85) if crop_img.size > 0 else b""
        scan_store.put(scan_id, frame_jpeg, crop_jpeg, face_thumb_jpeg)

        frame_url = f"/api/scans/{scan_id}/frame.jpg"
        crop_url = f"/api/scans/{scan_id}/crop.jpg"
        face_url = f"/api/scans/{scan_id}/face.jpg" if face_thumb_jpeg else None

        # --- Keywords (injury shorthand + codewords + blood buckets) --------
        # Keywords double as the chip labels in the UI. Low-confidence /
        # "possible" findings are prefixed so the medic immediately knows
        # they are probabilistic rather than confirmed — a critical UX
        # safeguard in a MASCAL context where a wrong "CRITICAL" label
        # could redirect resources.
        keywords: List[str] = []
        seen = set()
        for wd in wounds:
            sev = (wd.severity or "").strip().lower()
            conf = float(getattr(wd, "confidence", 0.0) or 0.0)
            region = _pretty_region(wd.body_location)
            prefix = ""
            if sev == "possible" or conf < 0.55:
                prefix = "possible "
                sev_token = ""
            else:
                sev_token = sev + " " if sev and sev not in ("unknown", "possible") else ""
            chip = f"{prefix}{sev_token}{wd.label} — {region}".strip()
            key = chip.lower()
            if key not in seen:
                seen.add(key)
                keywords.append(chip)
        for b in bloods:
            bucket = getattr(b, "volume_bucket", "") or ""
            label = f"bleeding pool — {bucket}" if bucket else "bleeding pool"
            if label.lower() not in seen:
                seen.add(label.lower())
                keywords.append(label)
        if temp_victim.tbsa_burn_percent and temp_victim.tbsa_burn_percent > 0:
            kw = f"burns ~{int(round(temp_victim.tbsa_burn_percent))}% TBSA"
            if kw.lower() not in seen:
                seen.add(kw.lower())
                keywords.append(kw)
        for c in codewords:
            cw = c["codeword"]
            if cw.lower() not in seen:
                seen.add(cw.lower())
                keywords.append(cw)

        # --- Build the ScanRecord ------------------------------------------
        record = ScanRecord(
            scan_id=scan_id,
            victim_id=victim_id or temp_victim.id,
            timestamp=time.time(),
            bbox=(x1, y1, x2, y2),
            frame_shape=(h, w),
            wounds=[_wound_to_scan_dict(wd, (h, w)) for wd in wounds],
            blood=[asdict(b) for b in bloods],
            vitals=vitals_dict,
            march=march_dict,
            salt_tag=tag.value,
            salt_reason=suggestion.reason,
            priority=priority,
            tbsa_burn_percent=temp_victim.tbsa_burn_percent,
            transcript_snippet=transcript_snippet[-800:],
            frame_url=frame_url,
            crop_url=crop_url,
            face_crop_url=face_url,
            keywords=keywords,
            sweep_frames=len(sweep_frames),
            sweep_duration_sec=float(sweep_duration_sec),
            detector_meta={
                "consensus": consensus_meta or {},
                "ensemble": ensemble_meta or {},
                "anomaly": anomaly_meta or {},
                "profile_flags": {
                    "consensus": bool(self.consensus_enabled),
                    "ensemble": bool(self.ensemble_enabled),
                    "anomaly": bool(self.anomaly_prior is not None and getattr(self.anomaly_prior, "ready", False)),
                },
                "face_bbox": list(face_bbox_frame) if face_bbox_frame else None,
            },
        )

        extras: Dict[str, Any] = {
            "face_embedding": face_embedding,
            "face_thumb_url": face_url,
            "priority": priority,
            "tccc_codewords": [c["codeword"] for c in codewords],
            "wound_regions": wounds,
            "blood_regions": bloods,
            "vitals": vitals_dict,
            "march": march_dict,
            "salt_tag": tag,
            "salt_reason": suggestion.reason,
            "tbsa_burn_percent": temp_victim.tbsa_burn_percent,
            "keywords": keywords,
        }
        return record, extras
