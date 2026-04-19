"""Person detection + tracking.

Primary: YOLOv8-pose via ultralytics (detection + 17 pose keypoints in one pass).
Fallback: MediaPipe Pose for single-person demos, or a stub that emits a synthetic
box covering the frame when no detector is available.

Tracker: a tiny IoU-greedy tracker. DeepSORT is overkill for our 3-5 victim demo.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class TrackedPerson:
    track_id: str
    bbox: Tuple[int, int, int, int]          # x1, y1, x2, y2
    confidence: float
    keypoints: List[Tuple[float, float, float]] = field(default_factory=list)  # x, y, vis
    last_seen: float = field(default_factory=time.time)


def _area(b: Tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = b
    return float(max(0, x2 - x1) * max(0, y2 - y1))


def _containment_fraction(inner: Tuple[int, int, int, int],
 outer: Tuple[int, int, int, int]) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    a = _area(inner)
    return inter / a if a > 0 else 0.0


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class PersonDetector:
    """Detect people + keypoints with graceful degradation.

    Order of preference:
      1. ultralytics YOLOv8-pose (accurate, fast, 17 COCO keypoints)
      2. mediapipe Pose (CPU-friendly, single subject)
      3. Stub: whole-frame box, no keypoints.
    """

    def __init__(self,
                 model_name: str = "yolov8n-pose.pt",
                 confidence: float = 0.5,
                 iou_threshold: float = 0.3,
                 max_age_frames: int = 30,
                 min_area_fraction: float = 0.02,
                 min_visible_keypoints: int = 5,
                 keypoint_conf_threshold: float = 0.3,
                 # --- Human-plausibility gates (filters posters, lights, fragments) ---
                 min_bbox_short_edge_px: int = 56,
                 max_aspect_width_over_height: float = 0.92,
                 min_keypoint_span_w: float = 0.22,
                 min_keypoint_span_h: float = 0.28,
                 min_shoulder_separation_frac_diag: float = 0.12,
                 suppress_contained_iou: float = 0.88,
                 reject_ceiling_shorties: bool = True,
                 ceiling_top_frac: float = 0.10,
                 ceiling_max_height_frac: float = 0.11,
                 # When several people are in frame, drop tiny isolated
                 # low-confidence boxes (posters / prints on walls behind them).
                 bg_suppress_enabled: bool = True,
                 bg_min_population: int = 2,
                 bg_area_ratio_vs_max: float = 0.32,
                 bg_max_confidence: float = 0.71,
                 bg_isolated_iou_cap: float = 0.03,
                 bg_isolated_area_ratio: float = 0.45,
                 bg_isolated_max_confidence: float = 0.68,
                 ) -> None:
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.min_area_fraction = min_area_fraction
        self.min_visible_keypoints = min_visible_keypoints
        self.keypoint_conf_threshold = keypoint_conf_threshold
        self.min_bbox_short_edge_px = int(min_bbox_short_edge_px)
        self.max_aspect_width_over_height = float(max_aspect_width_over_height)
        self.min_keypoint_span_w = float(min_keypoint_span_w)
        self.min_keypoint_span_h = float(min_keypoint_span_h)
        self.min_shoulder_separation_frac_diag = float(min_shoulder_separation_frac_diag)
        self.suppress_contained_iou = float(suppress_contained_iou)
        self.reject_ceiling_shorties = bool(reject_ceiling_shorties)
        self.ceiling_top_frac = float(ceiling_top_frac)
        self.ceiling_max_height_frac = float(ceiling_max_height_frac)
        self.bg_suppress_enabled = bool(bg_suppress_enabled)
        self.bg_min_population = int(bg_min_population)
        self.bg_area_ratio_vs_max = float(bg_area_ratio_vs_max)
        self.bg_max_confidence = float(bg_max_confidence)
        self.bg_isolated_iou_cap = float(bg_isolated_iou_cap)
        self.bg_isolated_area_ratio = float(bg_isolated_area_ratio)
        self.bg_isolated_max_confidence = float(bg_isolated_max_confidence)
        self._next_id = 1
        self._tracks: Dict[str, TrackedPerson] = {}
        self._frames_since_seen: Dict[str, int] = {}

        self._backend = None
        self._yolo = None
        self._mp_pose = None

        # Try YOLO first
        try:
            from ultralytics import YOLO  # type: ignore

            self._yolo = YOLO(model_name)
            self._backend = "yolov8-pose"
            print(f"[person] YOLOv8-pose loaded ({model_name}).")
        except Exception as exc:
            print(f"[person] YOLO unavailable ({exc}); trying MediaPipe.")

        if self._backend is None:
            try:
                import mediapipe as mp  # type: ignore

                if not hasattr(mp, "solutions"):
                    raise ImportError("mediapipe.solutions not available")

                self._mp_pose = mp.solutions.pose.Pose(model_complexity=1,
                                                      min_detection_confidence=confidence)
                self._backend = "mediapipe-pose"
                print("[person] MediaPipe Pose loaded (single subject).")
            except Exception as exc:
                print(f"[person] MediaPipe pose unavailable ({exc}); using stub detector.")
                self._backend = "stub"

    # ------------------------------------------------------------------
    def process(self, frame_bgr: np.ndarray) -> List[TrackedPerson]:
        detections = self._detect(frame_bgr)
        tracked = self._track(detections)
        return tracked

    # ------------------------------------------------------------------
    def _suppress_nested_detections(
        self,
        dets: List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]],
    ) -> List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]]:
        """Drop small boxes that are almost fully inside a higher-confidence box.

        Kills duplicate fragments (ear crop + torso) and many poster cells
        that YOLO splits incorrectly.
        """
        dets = sorted(dets, key=lambda x: -x[1])
        kept: List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]] = []
        for bbox, conf, kp in dets:
            area = _area(bbox)
            drop = False
            for obox, _oconf, _ in kept:
                frac_in = _containment_fraction(bbox, obox)
                if frac_in >= self.suppress_contained_iou and area < _area(obox) * 0.72:
                    drop = True
                    break
            if not drop:
                kept.append((bbox, conf, kp))
        return kept

    def _filter_likely_background_prints(
        self,
        dets: List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]],
    ) -> List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]]:
        """Drop wall-poster / distant FPs when at least one strong person exists."""
        if not self.bg_suppress_enabled or len(dets) < self.bg_min_population:
            return dets
        max_a = max(_area(b[0]) for b in dets)
        if max_a <= 1.0:
            return dets
        out: List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]] = []
        for i, (bbox, conf, kp) in enumerate(dets):
            a = _area(bbox)
            drop = False
            if a < self.bg_area_ratio_vs_max * max_a and conf < self.bg_max_confidence:
                drop = True
            if not drop:
                others = [dets[j][0] for j in range(len(dets)) if j != i]
                max_iou = max((_iou(bbox, ob) for ob in others), default=0.0)
                if (
                    max_iou < self.bg_isolated_iou_cap
                    and a < self.bg_isolated_area_ratio * max_a
                    and conf < self.bg_isolated_max_confidence
                ):
                    drop = True
            if not drop:
                out.append((bbox, conf, kp))
        return out if out else dets

    def _plausible_human_pose(
        self,
        bbox: Tuple[int, int, int, int],
        kp_list: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, int, int],
        *,
        coco_layout: bool = True,
    ) -> bool:
        """Geometry checks beyond raw keypoint count — rejects non-human FPs.

        ``coco_layout=True`` expects17 COCO pose points (YOLOv8-pose). For
        MediaPipe's 33-landmark layout set ``coco_layout=False`` — we still
        run size / aspect / spread checks but skip shoulder / nose-hip
        indices that don't line up with COCO numbering.
        """
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        thr = self.keypoint_conf_threshold

        if min(bw, bh) < self.min_bbox_short_edge_px:
            return False
        ar = bw / float(bh)
        if ar > self.max_aspect_width_over_height:
            return False

        if self.reject_ceiling_shorties:
            top = y1 / float(fh)
            hf = bh / float(fh)
            if top <= self.ceiling_top_frac and hf <= self.ceiling_max_height_frac:
                return False

        strong = [(float(x), float(y), float(c)) for x, y, c in kp_list if c > thr]
        if len(strong) < self.min_visible_keypoints:
            return False

        xs = [p[0] for p in strong]
        ys = [p[1] for p in strong]
        kx1, ky1, kx2, ky2 = min(xs), min(ys), max(xs), max(ys)
        kp_w = max(1e-3, kx2 - kx1)
        kp_h = max(1e-3, ky2 - ky1)
        if kp_w / bw < self.min_keypoint_span_w:
            return False
        if kp_h / bh < self.min_keypoint_span_h:
            return False

        if coco_layout and len(kp_list) >= 13:
            # Shoulder line — real humans have meaningful shoulder separation.
            ls, rs = kp_list[5], kp_list[6]
            if ls[2] > thr and rs[2] > thr:
                sh = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
                diag = math.hypot(float(bw), float(bh))
                if sh < self.min_shoulder_separation_frac_diag * diag:
                    return False

            # Upright pose: nose should sit above the hip midpoint when visible.
            nose = kp_list[0]
            lh, rh = kp_list[11], kp_list[12]
            if nose[2] > thr and lh[2] > thr and rh[2] > thr:
                hip_y = 0.5 * (lh[1] + rh[1])
                if float(nose[1]) > hip_y + 0.04 * float(bh):
                    return False

        return True

    def _is_close_range_upper_body_candidate(
        self,
        bbox: Tuple[int, int, int, int],
        kp_list: List[Tuple[float, float, float]],
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        """Relax pose gating for large, centered upper-body subjects.

        Close subjects in webcam demos often fill the frame with only the
        head, shoulders, and upper torso visible. Those detections can fail
        the normal full-body keypoint-span gates even though they are clearly
        the primary casualty in view. Keep this fallback narrow so it does not
        re-admit wall posters or tiny background false positives.
        """
        fh, fw = int(frame_shape[0]), int(frame_shape[1])
        x1, y1, x2, y2 = bbox
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area_frac = _area(bbox) / max(1.0, float(fh * fw))
        if area_frac < max(0.10, self.min_area_fraction * 3.0):
            return False

        cx = (x1 + x2) * 0.5 / max(1.0, float(fw))
        cy = (y1 + y2) * 0.5 / max(1.0, float(fh))
        if not (0.18 <= cx <= 0.82 and 0.18 <= cy <= 0.88):
            return False

        ar = bw / float(bh)
        if ar > max(1.35, self.max_aspect_width_over_height * 1.15):
            return False

        thr = self.keypoint_conf_threshold
        strong_idx = [idx for idx, (_x, _y, conf) in enumerate(kp_list) if conf > thr]
        strong = [kp_list[idx] for idx in strong_idx]
        if len(strong) < max(4, self.min_visible_keypoints - 2):
            return False

        head_visible = any(idx in {0, 1, 2, 3, 4} for idx in strong_idx)
        shoulder_count = sum(1 for idx in strong_idx if idx in {5, 6})
        upper_count = sum(1 for idx in strong_idx if idx <= 10)
        if not head_visible or shoulder_count < 1 or upper_count < 4:
            return False

        xs = [float(x) for x, _y, _c in strong]
        ys = [float(y) for _x, y, _c in strong]
        kp_w = max(xs) - min(xs)
        kp_h = max(ys) - min(ys)
        if kp_w / float(bw) < max(0.12, self.min_keypoint_span_w * 0.55):
            return False
        if kp_h / float(bh) < max(0.14, self.min_keypoint_span_h * 0.50):
            return False

        if shoulder_count >= 2 and len(kp_list) >= 7:
            ls, rs = kp_list[5], kp_list[6]
            if ls[2] > thr and rs[2] > thr:
                sh = math.hypot(ls[0] - rs[0], ls[1] - rs[1])
                diag = math.hypot(float(bw), float(bh))
                if sh < self.min_shoulder_separation_frac_diag * diag * 0.5:
                    return False

        return True

    # ------------------------------------------------------------------
    def _detect(self, frame_bgr: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float, List[Tuple[float, float, float]]]]:
        h, w = frame_bgr.shape[:2]
        frame_area = float(h * w)
        if self._backend == "yolov8-pose" and self._yolo is not None:
            # Raise the Ultralytics pre-NMS floor so we don't chase junk at
            # 0.25 conf; final gate is still ``self.confidence``.
            infer_conf = max(0.32, min(self.confidence, self.confidence * 0.88))
            results = self._yolo.predict(
                frame_bgr, conf=infer_conf, verbose=False, classes=[0])
            dets = []
            for r in results:
                if r.boxes is None:
                    continue
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls[0].cpu().item()) if box.cls is not None else 0
                    if cls_id != 0:
                        continue
                    conf = float(box.conf[0].cpu().item())
                    if conf < self.confidence:
                        continue
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    box_area = max(0, x2 - x1) * max(0, y2 - y1)
                    if box_area < self.min_area_fraction * frame_area:
                        continue
                    kp_list: List[Tuple[float, float, float]] = []
                    if r.keypoints is not None and i < len(r.keypoints):
                        kp = r.keypoints[i]
                        xy = kp.xy[0].cpu().numpy()
                        confs = kp.conf[0].cpu().numpy() if kp.conf is not None else np.ones(len(xy))
                        kp_list = [(float(x), float(y), float(c)) for (x, y), c in zip(xy, confs)]
                    if not kp_list:
                        continue
                    vis = sum(1 for _x, _y, c in kp_list if c > self.keypoint_conf_threshold)
                    close_range = self._is_close_range_upper_body_candidate(
                        (x1, y1, x2, y2),
                        kp_list,
                        frame_bgr.shape,
                    )
                    if vis < self.min_visible_keypoints and not close_range:
                        continue
                    if not close_range and not self._plausible_human_pose(
                        (x1, y1, x2, y2),
                        kp_list,
                        frame_bgr.shape,
                    ):
                        continue
                    dets.append(((x1, y1, x2, y2), conf, kp_list))
            dets = self._filter_likely_background_prints(dets)
            return self._suppress_nested_detections(dets)

        if self._backend == "mediapipe-pose" and self._mp_pose is not None:
            import cv2

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            res = self._mp_pose.process(rgb)
            if not res.pose_landmarks:
                return []
            xs = [lm.x * w for lm in res.pose_landmarks.landmark]
            ys = [lm.y * h for lm in res.pose_landmarks.landmark]
            bbox = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
            bx1, by1, bx2, by2 = bbox
            box_area = max(0, bx2 - bx1) * max(0, by2 - by1)
            if box_area < self.min_area_fraction * frame_area:
                return []
            kp = [(lm.x * w, lm.y * h, lm.visibility) for lm in res.pose_landmarks.landmark]
            vis = sum(1 for _x, _y, v in kp if v > self.keypoint_conf_threshold)
            if vis < self.min_visible_keypoints:
                return []
            if not self._plausible_human_pose(bbox, kp, frame_bgr.shape, coco_layout=False):
                return []
            return [(bbox, 0.8, kp)]

        # Stub: no detector available.
        return [((int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)), 0.5, [])]

    # ------------------------------------------------------------------
    def _track(self, detections) -> List[TrackedPerson]:
        now = time.time()
        # Age counters
        for tid in list(self._tracks.keys()):
            self._frames_since_seen[tid] = self._frames_since_seen.get(tid, 0) + 1

        used_ids: set[str] = set()
        updated: List[TrackedPerson] = []

        for bbox, conf, kp in detections:
            best_id: Optional[str] = None
            best_iou = self.iou_threshold
            for tid, tp in self._tracks.items():
                if tid in used_ids:
                    continue
                i = _iou(bbox, tp.bbox)
                if i > best_iou:
                    best_iou = i
                    best_id = tid
            if best_id is None:
                best_id = self._allocate_id()
                self._tracks[best_id] = TrackedPerson(track_id=best_id, bbox=bbox, confidence=conf,
                                                      keypoints=kp, last_seen=now)
            else:
                self._tracks[best_id].bbox = bbox
                self._tracks[best_id].confidence = conf
                self._tracks[best_id].keypoints = kp
                self._tracks[best_id].last_seen = now
            used_ids.add(best_id)
            self._frames_since_seen[best_id] = 0
            updated.append(self._tracks[best_id])

        # Drop stale
        for tid in list(self._tracks.keys()):
            if self._frames_since_seen.get(tid, 0) > self.max_age_frames:
                self._tracks.pop(tid, None)
                self._frames_since_seen.pop(tid, None)

        # Coast on the last good bbox when YOLO misses a frame (motion blur,
        # exposure, or our plausibility filters temporarily zeroing dets).
        # Without this, ``updated`` was empty whenever ``detections`` was
        # empty — the main loop saw ``tracked_this_frame=0`` and stopped
        # updating vitals / preview even though the person was still there.
        seen_ids = {tp.track_id for tp in updated}
        for tid, tp in list(self._tracks.items()):
            if tid in seen_ids:
                continue
            if self._frames_since_seen.get(tid, 0) <= self.max_age_frames:
                updated.append(tp)

        return updated

    def _allocate_id(self) -> str:
        idx = self._next_id - 1
        self._next_id += 1
        return f"CAS-{idx + 1:03d}"
