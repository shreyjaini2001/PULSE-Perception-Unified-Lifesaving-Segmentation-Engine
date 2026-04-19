"""Face-anchored victim re-identification.

Uses InsightFace ``buffalo_l`` (ArcFace ResNet50 embeddings, 512-D, cosine
similarity) to keep a victim's ID stable across re-entries into the frame.

If InsightFace isn't installed we degrade gracefully: ``match_or_register``
falls back to the caller-supplied ``track_id`` so the rest of the pipeline
keeps working.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class FaceMatchResult:
    victim_id: str
    similarity: float
    is_new: bool
    embedding: Optional[np.ndarray] = None
    face_crop_bgr: Optional[np.ndarray] = None
    # Face bbox in *full-frame* coordinates (x1, y1, x2, y2). Used by the
    # wound segmenter's face-aware suppression gate so detections whose
    # center falls inside the face require extra visual evidence.
    face_bbox_frame: Optional[Tuple[int, int, int, int]] = None


@dataclass
class _VictimFace:
    """Per-victim rolling face descriptor."""

    victim_id: str
    embedding: np.ndarray
    samples: int = 1
    last_updated: float = field(default_factory=time.time)

    def update(self, new_emb: np.ndarray, weight: float = 0.3) -> None:
        """Exponential moving average for robustness to lighting / angle."""
        if self.embedding is None:
            self.embedding = new_emb
        else:
            self.embedding = (1.0 - weight) * self.embedding + weight * new_emb
            n = np.linalg.norm(self.embedding)
            if n > 0:
                self.embedding = self.embedding / n
        self.samples += 1
        self.last_updated = time.time()


class FaceReID:
    """Thin wrapper around InsightFace with graceful absence handling."""

    def __init__(self,
                 name: str = "buffalo_l",
                 det_size: Tuple[int, int] = (640, 640),
                 threshold: float = 0.45,
                 callsigns: Optional[List[str]] = None) -> None:
        self.threshold = float(threshold)
        self.callsigns = callsigns or []
        self._known: Dict[str, _VictimFace] = {}
        self._assigned_seq = 0
        self._app = None
        self._ready = False
        try:
            from insightface.app import FaceAnalysis  # type: ignore
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self._app = FaceAnalysis(name=name, providers=providers)
            self._app.prepare(ctx_id=0, det_size=det_size)
            self._ready = True
            print(f"[face] InsightFace '{name}' ready (thresh={self.threshold}).")
        except Exception as exc:
            print(f"[face] InsightFace unavailable ({exc}); face re-ID disabled.")
            self._app = None

    # ------------------------------------------------------------------
    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    def _next_callsign(self) -> str:
        idx = self._assigned_seq
        self._assigned_seq += 1
        return f"CAS-{idx + 1:03d}"

    # ------------------------------------------------------------------
    def seed_known(self, victim_id: str, embedding: List[float]) -> None:
        """Re-hydrate from a persisted ``Victim.face_embedding``."""
        if not embedding:
            return
        arr = np.asarray(embedding, dtype=np.float32)
        n = np.linalg.norm(arr)
        if n > 0:
            arr = arr / n
        self._known[victim_id] = _VictimFace(victim_id=victim_id, embedding=arr)
        try:
            num = int(victim_id.split("-")[-1])
            self._assigned_seq = max(self._assigned_seq, num)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def embed_person(self, frame_bgr: np.ndarray,
                     person_bbox: Tuple[int, int, int, int],
                     ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """Run face detection inside the person bbox.

        Returns ``(embedding, face_crop_bgr, face_bbox_frame)`` where the
        face bbox is in full-frame coordinates.  Any element can be None.
        """
        if not self._ready or self._app is None:
            return None, None, None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = person_bbox
        # Limit to the upper 45% of the body (where the head usually is) to
        # keep InsightFace fast and accurate.
        head_h = max(64, int((y2 - y1) * 0.45))
        rx1 = max(0, x1 - 10)
        ry1 = max(0, y1 - 10)
        rx2 = min(w, x2 + 10)
        ry2 = min(h, y1 + head_h)
        if rx2 <= rx1 or ry2 <= ry1:
            return None, None, None
        crop = frame_bgr[ry1:ry2, rx1:rx2]
        try:
            faces = self._app.get(crop)
        except Exception:
            return None, None, None
        if not faces:
            return None, None, None
        # Pick the largest face.
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        face = faces[0]
        emb = getattr(face, "normed_embedding", None)
        if emb is None:
            emb = getattr(face, "embedding", None)
            if emb is not None:
                n = np.linalg.norm(emb)
                if n > 0:
                    emb = emb / n
        if emb is None:
            return None, None, None
        fb = face.bbox.astype(int)
        fx1 = max(0, fb[0])
        fy1 = max(0, fb[1])
        fx2 = min(crop.shape[1], fb[2])
        fy2 = min(crop.shape[0], fb[3])
        face_crop = crop[fy1:fy2, fx1:fx2].copy() if (fx2 > fx1 and fy2 > fy1) else None
        face_bbox_frame = (rx1 + fx1, ry1 + fy1, rx1 + fx2, ry1 + fy2)
        return np.asarray(emb, dtype=np.float32), face_crop, face_bbox_frame

    # ------------------------------------------------------------------
    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """Return (best_victim_id, cosine) over known embeddings, or (None, 0)."""
        if embedding is None or not self._known:
            return None, 0.0
        best_id, best_sim = None, -1.0
        for vid, face in self._known.items():
            if face.embedding is None:
                continue
            sim = float(np.dot(face.embedding, embedding))
            if sim > best_sim:
                best_sim = sim
                best_id = vid
        return best_id, best_sim if best_sim > 0 else 0.0

    # ------------------------------------------------------------------
    def match_or_register(self,
                           frame_bgr: np.ndarray,
                           person_bbox: Tuple[int, int, int, int],
                           fallback_id: Optional[str] = None,
                           ) -> Optional[FaceMatchResult]:
        """Detect a face in the bbox, match against known victims, or mint a
        new casualty ID. Returns None when no face was found (callers should
        stick with ``fallback_id`` in that case).
        """
        if not self._ready:
            return None
        emb, face_crop, face_bbox = self.embed_person(frame_bgr, person_bbox)
        if emb is None:
            return None
        best_id, sim = self.match(emb)
        if best_id is not None and sim >= self.threshold:
            self._known[best_id].update(emb)
            return FaceMatchResult(victim_id=best_id, similarity=sim,
                                    is_new=False, embedding=self._known[best_id].embedding,
                                    face_crop_bgr=face_crop,
                                    face_bbox_frame=face_bbox)
        # New identity. Prefer the caller's ``fallback_id`` if it looks fresh,
        # otherwise mint a neutral casualty ID.
        vid = fallback_id if fallback_id and fallback_id not in self._known else self._next_callsign()
        n = np.linalg.norm(emb)
        norm_emb = emb / n if n > 0 else emb
        self._known[vid] = _VictimFace(victim_id=vid, embedding=norm_emb)
        return FaceMatchResult(victim_id=vid, similarity=sim, is_new=True,
                               embedding=norm_emb, face_crop_bgr=face_crop,
                               face_bbox_frame=face_bbox)

    # ------------------------------------------------------------------
    def rename(self, old_id: str, new_id: str) -> None:
        if old_id in self._known and new_id != old_id:
            face = self._known.pop(old_id)
            face.victim_id = new_id
            self._known[new_id] = face
