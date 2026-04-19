"""Generate a tiny synthetic video for pipeline smoke tests.

Creates edge-node/demo/scenario_synthetic.mp4 — 5 seconds, 640x480, 15 FPS,
with a moving rectangle that mimics a person-ish silhouette so YOLO at
least has something to react to (it will not detect a person, but the
pipeline still runs end-to-end).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "edge-node" / "demo" / "scenario_synthetic.mp4"
OUT.parent.mkdir(parents=True, exist_ok=True)

W, H, FPS, SECONDS = 640, 480, 15, 5
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(OUT), fourcc, FPS, (W, H))

for i in range(FPS * SECONDS):
    frame = np.full((H, W, 3), 30, dtype=np.uint8)
    t = i / FPS
    cx = int(W * (0.2 + 0.6 * (0.5 + 0.5 * np.sin(t))))
    cy = H // 2
    cv2.rectangle(frame, (cx - 40, cy - 110), (cx + 40, cy + 110), (180, 180, 200), -1)
    cv2.circle(frame, (cx, cy - 140), 34, (200, 190, 180), -1)  # head-ish
    cv2.rectangle(frame, (cx - 10, cy - 20), (cx + 10, cy + 15), (40, 40, 220), -1)  # "blood"
    cv2.putText(frame, f"t={t:.1f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    writer.write(frame)

writer.release()
print(f"Wrote {OUT}")
