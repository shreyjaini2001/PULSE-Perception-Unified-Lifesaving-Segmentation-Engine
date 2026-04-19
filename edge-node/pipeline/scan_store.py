"""Shared in-memory store for scan JPEG payloads.

The scan engine registers frames + crops under scan IDs; the HTTP server
serves them from ``/api/scans/<scan_id>/frame.jpg`` and ``/crop.jpg``.

We cap the store to ``MAX_SCANS`` to avoid unbounded memory growth. Oldest
scans are evicted first.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, Optional


MAX_SCANS = 128


class _ScanJpegStore:
    def __init__(self, max_items: int = MAX_SCANS) -> None:
        self._lock = threading.RLock()
        self._frames: "OrderedDict[str, bytes]" = OrderedDict()
        self._crops: "OrderedDict[str, bytes]" = OrderedDict()
        self._faces: "OrderedDict[str, bytes]" = OrderedDict()
        self._max = max_items

    def _evict(self, d: "OrderedDict[str, bytes]") -> None:
        while len(d) > self._max:
            d.popitem(last=False)

    def put(self, scan_id: str, frame_jpeg: bytes, crop_jpeg: bytes,
            face_jpeg: Optional[bytes] = None) -> None:
        with self._lock:
            self._frames[scan_id] = frame_jpeg
            self._crops[scan_id] = crop_jpeg
            if face_jpeg is not None:
                self._faces[scan_id] = face_jpeg
            self._evict(self._frames)
            self._evict(self._crops)
            self._evict(self._faces)

    def get_frame(self, scan_id: str) -> Optional[bytes]:
        with self._lock:
            return self._frames.get(scan_id)

    def get_crop(self, scan_id: str) -> Optional[bytes]:
        with self._lock:
            return self._crops.get(scan_id)

    def get_face(self, scan_id: str) -> Optional[bytes]:
        with self._lock:
            return self._faces.get(scan_id)


# Module-level singleton. ws_server.py imports this.
store = _ScanJpegStore()
