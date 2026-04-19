"""Map pixel locations to body regions using COCO-17 pose keypoints.

Returns underscore-separated region ids (e.g. ``left_thigh``) for dashboard +3D pins.
Falls back to bbox thirds when keypoints are missing or insufficient.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# COCO-17 keypoint indices (YOLOv8-pose order)
NOSE = 0
LEFT_EYE, RIGHT_EYE = 1, 2
LEFT_EAR, RIGHT_EAR = 3, 4
LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6
LEFT_ELBOW, RIGHT_ELBOW = 7, 8
LEFT_WRIST, RIGHT_WRIST = 9, 10
LEFT_HIP, RIGHT_HIP = 11, 12
LEFT_KNEE, RIGHT_KNEE = 13, 14
LEFT_ANKLE, RIGHT_ANKLE = 15, 16

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]


def _as_kp_array(keypoints: List[Tuple[float, float, float]]) -> np.ndarray:
    """Shape (N, 3) x, y, conf."""
    if not keypoints:
        return np.zeros((0, 3), dtype=np.float64)
    return np.array([[float(p[0]), float(p[1]), float(p[2])] for p in keypoints], dtype=np.float64)


def keypoints_to_dict(
    keypoints_tensor: np.ndarray,
    conf_threshold: float = 0.3,
) -> Dict[str, Tuple[float, float]]:
    """Convert COCO-17 rows to {name: (x, y)} for visible keypoints."""
    result: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(COCO_KEYPOINTS):
        if i >= len(keypoints_tensor):
            break
        row = keypoints_tensor[i]
        x, y, conf = float(row[0]), float(row[1]), float(row[2])
        if conf >= conf_threshold:
            result[name] = (x, y)
    return result


def midpoint(
    kps_dict: Dict[str, Tuple[float, float]],
    name_a: str,
    name_b: str,
) -> Optional[Tuple[float, float]]:
    a = kps_dict.get(name_a)
    b = kps_dict.get(name_b)
    if a is None or b is None:
        return None
    if name_a == name_b and a is not None:
        return a
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def region_center(
    kps_dict: Dict[str, Tuple[float, float]],
    names: List[str],
    bias_top: float = 0.5,
) -> Optional[Tuple[float, float]]:
    points = [kps_dict[n] for n in names if n in kps_dict]
    if len(points) < 2:
        return None
    cx = sum(p[0] for p in points) / len(points)
    ys = sorted(p[1] for p in points)
    cy = ys[0] + (ys[-1] - ys[0]) * bias_top
    return (cx, cy)


# Region id -> compute pixel center from named keypoints
BODY_REGION_COMPUTE: Dict[str, Callable[[Dict[str, Tuple[float, float]]], Optional[Tuple[float, float]]]] = {
    "head": lambda k: midpoint(k, "nose", "nose"),
    "face": lambda k: midpoint(k, "left_eye", "right_eye"),
    "neck": lambda k: midpoint(k, "left_shoulder", "right_shoulder"),
    "chest": lambda k: region_center(
        k, ["left_shoulder", "right_shoulder", "left_hip", "right_hip"], bias_top=0.28),
    "abdomen": lambda k: region_center(
        k, ["left_shoulder", "right_shoulder", "left_hip", "right_hip"], bias_top=0.72
    ),
    "left_shoulder": lambda k: k.get("left_shoulder"),
    "right_shoulder": lambda k: k.get("right_shoulder"),
    "left_upper_arm": lambda k: midpoint(k, "left_shoulder", "left_elbow"),
    "right_upper_arm": lambda k: midpoint(k, "right_shoulder", "right_elbow"),
    "left_forearm": lambda k: midpoint(k, "left_elbow", "left_wrist"),
    "right_forearm": lambda k: midpoint(k, "right_elbow", "right_wrist"),
    "left_hand": lambda k: midpoint(k, "left_wrist", "left_wrist"),
    "right_hand": lambda k: midpoint(k, "right_wrist", "right_wrist"),
    "left_thigh": lambda k: midpoint(k, "left_hip", "left_knee"),
    "right_thigh": lambda k: midpoint(k, "right_hip", "right_knee"),
    "left_knee": lambda k: midpoint(k, "left_knee", "left_knee"),
    "right_knee": lambda k: midpoint(k, "right_knee", "right_knee"),
    "left_shin": lambda k: midpoint(k, "left_knee", "left_ankle"),
    "right_shin": lambda k: midpoint(k, "right_knee", "right_ankle"),
    "left_ankle": lambda k: k.get("left_ankle"),
    "right_ankle": lambda k: k.get("right_ankle"),
    "left_foot": lambda k: k.get("left_ankle"),
    "right_foot": lambda k: k.get("right_ankle"),
    "groin": lambda k: midpoint(k, "left_hip", "right_hip"),
}


def _maybe_add_back_region(
    kps_dict: Dict[str, Tuple[float, float]],
    regions: Dict[str, Tuple[float, float]],
) -> None:
    """If hips project above shoulders in 2D, add a ``back`` torso anchor."""
    ls, rs = kps_dict.get("left_shoulder"), kps_dict.get("right_shoulder")
    lh, rh = kps_dict.get("left_hip"), kps_dict.get("right_hip")
    if not all([ls, rs, lh, rh]):
        return
    mid_sh_y = (ls[1] + rs[1]) / 2
    mid_hip_y = (lh[1] + rh[1]) / 2
    # Image y grows downward; hips normally below shoulders (larger y).
    if mid_hip_y < mid_sh_y - 30:
        mid_x = (ls[0] + rs[0] + lh[0] + rh[0]) / 4
        mid_y = (mid_sh_y + mid_hip_y) / 2
        regions["back"] = (mid_x, mid_y)


def compute_body_regions(
    keypoints: List[Tuple[float, float, float]],
    conf_threshold: float = 0.3,
) -> Dict[str, Tuple[float, float]]:
    """Compute pixel centers for anatomical regions from COCO-17 keypoints."""
    arr = _as_kp_array(keypoints)
    if len(arr) < 17:
        return {}
    kps_dict = keypoints_to_dict(arr, conf_threshold=conf_threshold)
    regions: Dict[str, Tuple[float, float]] = {}
    for region_name, compute_fn in BODY_REGION_COMPUTE.items():
        try:
            coord = compute_fn(kps_dict)
            if coord is not None:
                regions[region_name] = coord
        except (KeyError, TypeError, IndexError):
            continue
    _maybe_add_back_region(kps_dict, regions)
    return regions


def locate_wound_on_body(
    wound_center_xy: Tuple[float, float],
    body_regions: Dict[str, Tuple[float, float]],
) -> Tuple[str, float]:
    """Return closest region id and pixel distance."""
    if not body_regions:
        return ("unknown", float("inf"))
    wx, wy = wound_center_xy
    best_region = "unknown"
    best_dist = float("inf")
    for region_name, (rx, ry) in body_regions.items():
        dist = ((wx - rx) ** 2 + (wy - ry) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_region = region_name
    return (best_region, best_dist)


def _fallback_bbox_region(
    wound_bbox: Tuple[int, int, int, int],
    victim_bbox: Tuple[int, int, int, int],
) -> str:
    cx = (wound_bbox[0] + wound_bbox[2]) / 2
    cy = (wound_bbox[1] + wound_bbox[3]) / 2
    x1, y1, x2, y2 = victim_bbox
    h = max(1, y2 - y1)
    rel_y = (cy - y1) / h
    side = "left" if cx < (x1 + x2) / 2 else "right"
    if rel_y < 0.25:
        return "head"
    if rel_y < 0.55:
        return f"{side}_torso"
    if rel_y < 0.8:
        return f"{side}_thigh"
    return f"{side}_lower_leg"


class BodyLocator:
    """Compatibility wrapper: ``locate`` returns underscore region ids."""

    @staticmethod
    def locate(
        wound_bbox: Tuple[int, int, int, int],
        victim_bbox: Tuple[int, int, int, int],
        keypoints: List[Tuple[float, float, float]],
    ) -> str:
        cx = (wound_bbox[0] + wound_bbox[2]) / 2
        cy = (wound_bbox[1] + wound_bbox[3]) / 2

        if keypoints and len(keypoints) >= 17:
            regions = compute_body_regions(keypoints)
            if regions:
                region, _dist = locate_wound_on_body((cx, cy), regions)
                return region

        return _fallback_bbox_region(wound_bbox, victim_bbox)
