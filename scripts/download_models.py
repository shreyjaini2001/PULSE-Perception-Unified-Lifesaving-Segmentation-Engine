"""Pre-download every model the edge node may need.

Run this ONCE on a good connection (hackathon venue WiFi cannot be trusted).
Skips models that are already cached and prints a summary at the end.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

results: List[Tuple[str, bool, str]] = []


def try_step(name: str, fn: Callable[[], str]) -> None:
    try:
        info = fn()
        results.append((name, True, info))
        print(f"✓ {name}  {info}")
    except Exception as exc:
        results.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"✗ {name}  {type(exc).__name__}: {exc}")


def download_yolo() -> str:
    from ultralytics import YOLO  # type: ignore

    y = YOLO("yolov8s-pose.pt")
    return f"weights cached at {y.ckpt_path}"


def download_grounding_dino() -> str:
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection  # type: ignore

    AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base")
    return "grounding-dino-base cached"


def download_dinov2() -> str:
    from transformers import AutoImageProcessor, AutoModel  # type: ignore

    AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    AutoModel.from_pretrained("facebook/dinov2-base")
    return "dinov2-base cached"


def download_sam2() -> str:
    import torch  # type: ignore
    from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Cache both base-plus and small so users can toggle without re-downloading.
    SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small", device=device)
    SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-base-plus", device=device)
    return f"sam2-hiera-{{small, base-plus}} cached (device={device})"


def download_whisper() -> str:
    import torch  # type: ignore
    from faster_whisper import WhisperModel  # type: ignore

    # Use small.en on GPU; fall back to tiny.en-int8 on CPU-only machines.
    if torch.cuda.is_available():
        WhisperModel("small.en", device="cuda", compute_type="float16")
        return "faster-whisper small.en cached (device=cuda, float16)"
    WhisperModel("tiny.en", device="cpu", compute_type="int8")
    return "faster-whisper tiny.en cached (device=cpu, int8)"


def download_llama() -> str:
    from huggingface_hub import hf_hub_download  # type: ignore

    local = MODEL_DIR / "llama-3.2-3b-instruct.gguf"
    if local.exists():
        return f"already at {local}"
    path = hf_hub_download(
        repo_id="bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename="Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        local_dir=str(MODEL_DIR),
    )
    # Rename for convenience
    os.replace(path, local)
    return f"downloaded to {local}"


def main() -> None:
    print(f"Downloading into {MODEL_DIR}\n")
    try_step("YOLOv8s-pose (ultralytics)", download_yolo)
    try_step("Grounding DINO base",        download_grounding_dino)
    try_step("DINOv2 base",                download_dinov2)
    try_step("SAM 2.1 base-plus",          download_sam2)
    try_step("Whisper (auto size)",        download_whisper)
    try_step("Llama 3.2 3B Q4_K_M",        download_llama)

    print("\nSummary:")
    for name, ok, info in results:
        mark = "✓" if ok else "✗"
        print(f"  {mark} {name:<32} {info}")
    failed = [r for r in results if not r[1]]
    if failed:
        print(f"\n{len(failed)} model(s) failed to download. The pipeline degrades gracefully, "
              f"but install the deps in requirements.txt (and any gated HF access) and re-run "
              f"this script before the demo.")
        sys.exit(1)


if __name__ == "__main__":
    main()
