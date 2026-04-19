"""DINOv2/v3 feature extraction (shared backbone).

In the MVP we do NOT actually require a DINO backbone — YOLOv8-pose gives us
detection + keypoints directly, and SAM 2.1 carries its own encoder. This
module exists so the pitch story ("shared DINOv3 features consumed by several
heads") is easy to wire in later: call ``BackboneFeatures.extract(frame)`` once
per frame and hand the tensor to downstream heads.

Falls back to a no-op if PyTorch / transformers are not installed.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


class DinoBackbone:
    def __init__(self, model_id: str = "facebook/dinov2-base", device: str = "auto") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._processor = None
        self._available = False
        try:
            import torch  # noqa: F401
            from transformers import AutoImageProcessor, AutoModel  # noqa: F401

            self._torch = torch
            self._AutoImageProcessor = AutoImageProcessor
            self._AutoModel = AutoModel
        except Exception:
            self._torch = None

    def _lazy_load(self) -> bool:
        if self._available or self._torch is None:
            return self._available
        try:
            self._processor = self._AutoImageProcessor.from_pretrained(self.model_id)
            self._model = self._AutoModel.from_pretrained(self.model_id)
            dev = self._resolve_device()
            self._model = self._model.to(dev).eval()
            self._device = dev
            self._available = True
        except Exception as exc:
            print(f"[backbone] DINO unavailable ({exc}); continuing without shared features.")
            self._available = False
        return self._available

    def _resolve_device(self) -> str:
        if self._torch is None:
            return "cpu"
        if self.device == "auto":
            if self._torch.cuda.is_available():
                return "cuda"
            if getattr(self._torch.backends, "mps", None) and self._torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def extract(self, frame_bgr: np.ndarray) -> Optional[Any]:
        """Return a patch-token tensor or None if unavailable."""
        if not self._lazy_load():
            return None
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        inputs = self._processor(images=rgb, return_tensors="pt").to(self._device)
        with self._torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.last_hidden_state
