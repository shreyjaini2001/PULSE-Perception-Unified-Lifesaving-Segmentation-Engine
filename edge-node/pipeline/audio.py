"""Voice transcription via faster-whisper.

Runs in a background thread so the frame pipeline never blocks on audio.
Chunks mic input every ~3 seconds; emits ``(timestamp, text)`` tuples to a
shared transcript queue consumed by state.scene.

If no mic or whisper is unavailable, the transcriber is a no-op but still
exposes ``push_text(..)`` so command-line / dashboard-originated notes work.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Deque, List, Optional, Tuple


@dataclass
class TranscriptChunk:
    timestamp: float
    text: str


class AudioTranscriber:
    def __init__(self,
                 model: str = "tiny.en",
                 device: str = "auto",
                 compute_type: str = "auto",
                 sample_rate: int = 16000,
                 chunk_seconds: float = 3.0,
                 on_chunk: Optional[Callable[[TranscriptChunk], None]] = None) -> None:
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.on_chunk = on_chunk

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._chunks: Deque[TranscriptChunk] = deque(maxlen=40)

        self._whisper = None
        self._sd = None
        self._available = False

        try:
            from faster_whisper import WhisperModel  # type: ignore

            actual_device = device
            actual_compute = compute_type
            if device == "auto":
                try:
                    import torch  # type: ignore

                    if torch.cuda.is_available():
                        actual_device = "cuda"
                        if actual_compute == "auto":
                            actual_compute = "float16"
                    else:
                        actual_device = "cpu"
                        if actual_compute == "auto":
                            actual_compute = "int8"
                except Exception:
                    actual_device = "cpu"
                    if actual_compute == "auto":
                        actual_compute = "int8"
            elif actual_compute == "auto":
                # explicit device; pick a sensible default compute type
                actual_compute = "float16" if actual_device == "cuda" else "int8"

            self._whisper = WhisperModel(model, device=actual_device, compute_type=actual_compute)
            print(f"[audio] faster-whisper '{model}' loaded on {actual_device} ({actual_compute}).")
        except Exception as exc:
            print(f"[audio] faster-whisper unavailable ({exc}); transcription disabled.")

        try:
            import sounddevice as sd  # type: ignore

            self._sd = sd
        except Exception as exc:
            print(f"[audio] sounddevice unavailable ({exc}); mic capture disabled.")

        self._available = self._whisper is not None and self._sd is not None

    # ------------------------------------------------------------------
    def start(self) -> None:
        if not self._available or self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="audio-transcriber", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def push_text(self, text: str) -> None:
        """Manual note (from dashboard / CLI), bypasses Whisper."""
        chunk = TranscriptChunk(timestamp=time.time(), text=text.strip())
        self._chunks.append(chunk)
        if self.on_chunk:
            self.on_chunk(chunk)

    def recent(self, seconds: float = 30.0) -> List[TranscriptChunk]:
        cutoff = time.time() - seconds
        return [c for c in self._chunks if c.timestamp >= cutoff]

    def recent_text(self, seconds: float = 30.0) -> str:
        return " ".join(c.text for c in self.recent(seconds)).strip()

    # ------------------------------------------------------------------
    def _loop(self) -> None:
        import numpy as np

        block_samples = int(self.sample_rate * self.chunk_seconds)
        buf = np.zeros(block_samples, dtype=np.float32)
        try:
            with self._sd.InputStream(samplerate=self.sample_rate, channels=1,
                                      dtype="float32") as stream:
                while self._running:
                    audio, _ = stream.read(block_samples)
                    audio = audio.reshape(-1)
                    # Gate on RMS so we don't transcribe silence
                    rms = float((audio ** 2).mean() ** 0.5)
                    if rms < 0.005:
                        continue
                    try:
                        segments, _info = self._whisper.transcribe(audio, language="en",
                                                                   vad_filter=True, beam_size=1)
                        text_parts = [s.text.strip() for s in segments if s.text.strip()]
                        text = " ".join(text_parts).strip()
                        if text:
                            self.push_text(text)
                    except Exception as exc:
                        print(f"[audio] transcribe error: {exc}")
        except Exception as exc:
            print(f"[audio] input stream failed: {exc}; transcription stopped.")
            self._running = False
