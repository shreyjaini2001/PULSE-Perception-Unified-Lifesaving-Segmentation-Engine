# Benchmarks

Run `python benchmarks/fps_profiler.py --duration 30` to (re)generate.

Target numbers for the pitch:

| Metric | Target | Measured | Notes |
|---|---|---|---|
| End-to-end FPS (laptop, RTX 3060-class) | 8–12 FPS | _tbd_ | full pipeline with SAM on |
| End-to-end FPS (laptop, CPU-only) | 3–6 FPS | _tbd_ | GDINO + SAM off, color fallback |
| Peak RAM | < 1.5 GB | _tbd_ | |
| Llama MIST generation | < 6 s / card | _tbd_ | Llama 3.2 3B Q4_K_M |
| CoT publish latency | < 500 ms | _tbd_ | tag-confirm → ATAK pin |
| Per-frame stages (p50) | | | |
| &nbsp;&nbsp; Person detection | < 25 ms | _tbd_ | YOLOv8n-pose |
| &nbsp;&nbsp; Wound segmentation | < 120 ms | _tbd_ | GDINO + SAM, per-victim |
| &nbsp;&nbsp; rPPG sampling | < 15 ms | _tbd_ | MediaPipe FaceMesh + FFT |

Extrapolation: on a Snapdragon 8 Gen 3 NPU with the same weights quantized to INT8 via
ExecuTorch, published Meta benchmarks suggest ~3× headroom over an RTX 3060, which
maps to a steady ~20-30 FPS for the full pipeline — well within Ray-Ban Meta /
compact compute pack envelopes.
