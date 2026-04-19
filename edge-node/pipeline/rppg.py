"""Remote photoplethysmography (rPPG) — heart rate from facial skin color.

Algorithm: GRGB (Green minus average of Red + Blue) on forehead + cheek ROIs,
bandpassed to 0.7-3.5 Hz (42-210 BPM), peak detected in the frequency domain
via FFT. We keep a per-victim ring buffer of skin-color means and compute a
new HR estimate every N frames.

Respiratory rate is a secondary extraction from a lower-frequency band
(0.1-0.5 Hz / 6-30 BPM) on the same signal.

SpO2 is not estimated here — it requires calibrated RGB or near-IR which we
don't have from a webcam. We leave the field ``None`` and surface that honestly.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class RppgEstimate:
    hr: Optional[float] = None
    rr: Optional[float] = None
    hr_confidence: float = 0.0
    rr_confidence: float = 0.0
    timestamp: float = 0.0


class RppgEstimator:
    """One estimator instance, shared across all victims."""

    FOREHEAD_LANDMARKS = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323]
    LEFT_CHEEK = [234, 93, 132, 58, 172, 136, 150]
    RIGHT_CHEEK = [454, 323, 361, 288, 397, 365, 379]

    def __init__(self,
                 window_seconds: float = 10.0,
                 fps_hint: float = 15.0,
                 min_confidence: float = 0.4) -> None:
        self.window_seconds = window_seconds
        self.fps_hint = fps_hint
        self.min_confidence = min_confidence
        self._buffers: Dict[str, Deque[Tuple[float, float, float, float]]] = defaultdict(
            lambda: deque(maxlen=int(window_seconds * fps_hint * 3))
        )  # (ts, R, G, B)
        self._last_estimate: Dict[str, RppgEstimate] = {}

        self._face_mesh = None
        try:
            import mediapipe as mp  # type: ignore

            if not hasattr(mp, "solutions"):
                # MediaPipe wheels for Python 3.13 ship without the legacy
                # ``mp.solutions`` namespace. We only use FaceMesh here, and
                # the pipeline degrades gracefully when it's missing.
                raise ImportError("mediapipe.solutions not available on this Python build")

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=4,
                refine_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            print("[rppg] MediaPipe FaceMesh loaded.")
        except Exception as exc:
            print(f"[rppg] face mesh unavailable ({exc}); rPPG will show HR as pending.")

    # ------------------------------------------------------------------
    def process(self,
                frame_bgr: np.ndarray,
                victims: List[Tuple[str, Tuple[int, int, int, int]]]) -> Dict[str, RppgEstimate]:
        """For each (victim_id, bbox), sample skin color from face ROIs and update HR.

        Returns a dict {victim_id -> RppgEstimate}. Victims without a face
        mesh lock retain their previous estimate (or report no lock).
        """
        results: Dict[str, RppgEstimate] = {}
        if self._face_mesh is None or not victims:
            return {vid: self._last_estimate.get(vid, RppgEstimate()) for vid, _ in victims}

        import cv2

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        now = time.time()

        for vid, bbox in victims:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 40 or y2 - y1 < 40:
                results[vid] = self._last_estimate.get(vid, RppgEstimate())
                continue

            crop = rgb[y1:y2, x1:x2]
            ch, cw = crop.shape[:2]
            fm = self._face_mesh.process(crop)
            if not fm.multi_face_landmarks:
                results[vid] = self._last_estimate.get(vid, RppgEstimate())
                continue

            lm = fm.multi_face_landmarks[0].landmark
            pts = [(int(lm[i].x * cw), int(lm[i].y * ch)) for i in self.FOREHEAD_LANDMARKS
                   if 0 <= lm[i].x <= 1 and 0 <= lm[i].y <= 1]
            pts += [(int(lm[i].x * cw), int(lm[i].y * ch)) for i in self.LEFT_CHEEK + self.RIGHT_CHEEK
                    if 0 <= lm[i].x <= 1 and 0 <= lm[i].y <= 1]
            if not pts:
                results[vid] = self._last_estimate.get(vid, RppgEstimate())
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            rx1, ry1, rx2, ry2 = max(0, min(xs)), max(0, min(ys)), min(cw, max(xs)), min(ch, max(ys))
            if rx2 - rx1 < 10 or ry2 - ry1 < 10:
                results[vid] = self._last_estimate.get(vid, RppgEstimate())
                continue
            roi = crop[ry1:ry2, rx1:rx2]
            mean = roi.reshape(-1, 3).mean(axis=0)  # R, G, B

            self._buffers[vid].append((now, float(mean[0]), float(mean[1]), float(mean[2])))
            est = self._estimate(vid)
            results[vid] = est
            self._last_estimate[vid] = est

        return results

    # ------------------------------------------------------------------
    def _estimate(self, vid: str) -> RppgEstimate:
        buf = list(self._buffers[vid])
        if len(buf) < int(self.window_seconds * 4):  # need ~4 Hz * 10 s minimum
            return RppgEstimate()

        buf = [p for p in buf if p[0] >= buf[-1][0] - self.window_seconds]
        if len(buf) < 30:
            return RppgEstimate()

        ts = np.array([p[0] for p in buf])
        r = np.array([p[1] for p in buf])
        g = np.array([p[2] for p in buf])
        b = np.array([p[3] for p in buf])

        # Resample to a uniform grid. Duration and sample rate derived from buffer.
        dt = np.median(np.diff(ts))
        if dt <= 0 or not np.isfinite(dt):
            return RppgEstimate()
        fs = 1.0 / dt
        if fs < 4:
            return RppgEstimate()
        duration = ts[-1] - ts[0]
        if duration < self.window_seconds * 0.5:
            return RppgEstimate()

        grid = np.arange(ts[0], ts[-1], dt)
        r_u = np.interp(grid, ts, r)
        g_u = np.interp(grid, ts, g)
        b_u = np.interp(grid, ts, b)

        # GRGB chrominance signal
        signal = g_u - 0.5 * (r_u + b_u)
        signal = signal - np.mean(signal)
        if np.std(signal) < 1e-3:
            return RppgEstimate()

        # Simple band-pass via FFT masking
        n = len(signal)
        freqs = np.fft.rfftfreq(n, d=dt)
        spectrum = np.fft.rfft(signal * np.hanning(n))
        power = np.abs(spectrum) ** 2

        hr_band = (freqs >= 0.7) & (freqs <= 3.5)
        rr_band = (freqs >= 0.1) & (freqs <= 0.5)

        hr_bpm: Optional[float] = None
        hr_conf = 0.0
        if hr_band.any():
            hr_power = power.copy()
            hr_power[~hr_band] = 0
            peak_idx = int(np.argmax(hr_power))
            if hr_power[peak_idx] > 0:
                hr_bpm = float(freqs[peak_idx] * 60.0)
                total = power[hr_band].sum()
                hr_conf = float(hr_power[peak_idx] / total) if total > 0 else 0.0

        rr_bpm: Optional[float] = None
        rr_conf = 0.0
        if rr_band.any():
            rr_power = power.copy()
            rr_power[~rr_band] = 0
            peak_idx = int(np.argmax(rr_power))
            if rr_power[peak_idx] > 0:
                rr_bpm = float(freqs[peak_idx] * 60.0)
                total = power[rr_band].sum()
                rr_conf = float(rr_power[peak_idx] / total) if total > 0 else 0.0

        est = RppgEstimate(hr=hr_bpm, rr=rr_bpm,
                           hr_confidence=hr_conf, rr_confidence=rr_conf,
                           timestamp=time.time())
        if hr_conf < self.min_confidence:
            est.hr = None
        if rr_conf < self.min_confidence:
            est.rr = None
        return est
