"""End-to-end pipeline FPS + per-stage latency profiler.

Runs the edge pipeline over a video (or webcam) for N seconds and reports:
  * Per-stage wall-clock ms (p50, p95).
  * End-to-end FPS.
  * Peak host RSS in MB.
  * Peak GPU VRAM in MB per stage + overall (CUDA only).
  * Optionally, a **precision sweep** — the same workload is repeated at
    multiple dtypes (FP32 / FP16 / BF16) and a markdown comparison row
    is emitted for the deck.

Outputs:
  * ``benchmarks/results_{stamp}.csv`` — raw rows (one per stage, plus
    summary rows for FPS / peak memory).
  * ``benchmarks/results_{stamp}.md``  — judge-friendly table.

CLI:
  python benchmarks/fps_profiler.py --source 0 --duration 20
  python benchmarks/fps_profiler.py --source demo/scenario_blast.mp4 \
         --duration 15 --compare-precisions fp32,fp16
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
EDGE = ROOT / "edge-node"
sys.path.insert(0, str(EDGE))

import numpy as np  # noqa: E402
import psutil  # noqa: E402


def _import_pipeline():
    from pipeline import (AudioTranscriber, BodyLocator, MistGenerator,
                           PersonDetector, RppgEstimator, WoundSegmenter)  # noqa
    return locals()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="0")
    p.add_argument("--duration", type=float, default=20.0, help="seconds per precision pass")
    p.add_argument("--no-sam", action="store_true")
    p.add_argument("--no-rppg", action="store_true")
    p.add_argument("--compare-precisions", default="",
                   help="Comma-separated dtypes to sweep (e.g. 'fp32,fp16' or 'fp16,bf16'). "
                        "Empty = single run at default (fp16 if CUDA else fp32).")
    return p.parse_args()


# --------------------------------------------------------------------
# CUDA / precision helpers
# --------------------------------------------------------------------


def _torch():
    try:
        import torch  # type: ignore
        return torch
    except Exception:
        return None


def _set_precision(dtype: str) -> str:
    """Apply a dtype preference to torch + cuDNN. Returns the label used."""
    torch = _torch()
    if torch is None:
        return "fp32"
    dtype = (dtype or "").lower()
    if dtype in {"fp16", "float16", "half"}:
        torch.set_default_dtype(torch.float32)  # weights still materialize as fp32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_float32_matmul_precision("high")
        return "fp16"
    if dtype in {"bf16", "bfloat16"}:
        torch.set_default_dtype(torch.float32)
        torch.set_float32_matmul_precision("high")
        return "bf16"
    # Default / fp32
    torch.set_default_dtype(torch.float32)
    torch.set_float32_matmul_precision("highest")
    return "fp32"


def _autocast_ctx(dtype_label: str):
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        class _Null:
            def __enter__(self): return self
            def __exit__(self, *a): return False
        return _Null()
    import contextlib
    if dtype_label == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    if dtype_label == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _reset_vram_peaks() -> None:
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _vram_peak_mb() -> float:
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


# --------------------------------------------------------------------
# Single run
# --------------------------------------------------------------------


def _run_once(args, dtype_label: str) -> Dict[str, object]:
    """One profiling pass at a given precision. Returns a results dict."""
    mods = _import_pipeline()

    import cv2

    _set_precision(dtype_label)
    _reset_vram_peaks()

    person = mods["PersonDetector"](confidence=0.35)
    wounds = mods["WoundSegmenter"](
        gdino_prompt="person . blood . laceration . burn . tourniquet",
        use_grounding_dino=not args.no_sam,
        use_sam=not args.no_sam,
    )
    rppg = None if args.no_rppg else mods["RppgEstimator"](window_seconds=8.0, fps_hint=10)

    vram_after_load = _vram_peak_mb()

    src = int(args.source) if str(args.source).isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open {src}")
        return {"dtype": dtype_label, "fps": 0.0, "frames": 0, "stages": {}}

    stage_ms: Dict[str, List[float]] = {"person": [], "wound": [], "rppg": []}
    stage_vram_mb: Dict[str, float] = {"person": 0.0, "wound": 0.0, "rppg": 0.0}
    frame_count = 0
    t0 = time.time()
    proc = psutil.Process(os.getpid())
    peak_rss = 0.0

    cast = _autocast_ctx(dtype_label)
    with cast:
        while time.time() - t0 < args.duration:
            ok, frame = cap.read()
            if not ok:
                if isinstance(src, str):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            _reset_vram_peaks()
            t = time.time()
            tracked = person.process(frame)
            stage_ms["person"].append((time.time() - t) * 1000)
            stage_vram_mb["person"] = max(stage_vram_mb["person"], _vram_peak_mb())

            if tracked:
                _reset_vram_peaks()
                t = time.time()
                for tp in tracked:
                    wounds.process(frame, tp.bbox)
                stage_ms["wound"].append((time.time() - t) * 1000)
                stage_vram_mb["wound"] = max(stage_vram_mb["wound"], _vram_peak_mb())

                if rppg:
                    _reset_vram_peaks()
                    t = time.time()
                    rppg.process(frame, [(tp.track_id, tp.bbox) for tp in tracked])
                    stage_ms["rppg"].append((time.time() - t) * 1000)
                    stage_vram_mb["rppg"] = max(stage_vram_mb["rppg"], _vram_peak_mb())

            rss_mb = proc.memory_info().rss / (1024 * 1024)
            peak_rss = max(peak_rss, rss_mb)
            frame_count += 1

    cap.release()
    duration = time.time() - t0
    fps = frame_count / duration if duration > 0 else 0.0

    # Free the models between precision passes so VRAM measurements for
    # subsequent passes reflect only that pass's load.
    del person, wounds, rppg
    gc.collect()
    torch = _torch()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    overall_vram = max([vram_after_load, *stage_vram_mb.values()])

    return {
        "dtype": dtype_label,
        "frames": frame_count,
        "duration": duration,
        "fps": fps,
        "peak_rss_mb": peak_rss,
        "vram_after_load_mb": vram_after_load,
        "vram_overall_mb": overall_vram,
        "stages_ms": stage_ms,
        "stages_vram_mb": stage_vram_mb,
    }


def _pct(xs: List[float], p: int) -> float:
    if not xs:
        return float("nan")
    return statistics.quantiles(xs, n=100)[p - 1] if len(xs) > 1 else xs[0]


def _print_run(res: Dict[str, object]) -> None:
    print(f"\n=== Benchmark results ({res['dtype']}) ===")
    print(f"Frames processed : {res['frames']}")
    print(f"Duration         : {res['duration']:.2f}s")
    print(f"End-to-end FPS   : {res['fps']:.2f}")
    print(f"Peak RSS         : {res['peak_rss_mb']:.0f} MB")
    print(f"VRAM (after load): {res['vram_after_load_mb']:.0f} MB")
    print(f"VRAM (overall)   : {res['vram_overall_mb']:.0f} MB")
    for stage, xs in res["stages_ms"].items():
        if not xs:
            continue
        p50 = statistics.median(xs)
        p95 = _pct(xs, 95)
        vram = res["stages_vram_mb"].get(stage, 0.0)
        print(f"  {stage:<10} p50={p50:6.1f}ms  p95={p95:6.1f}ms  "
              f"vram={vram:5.0f}MB  n={len(xs)}")


def _write_csv(path: Path, runs: List[Dict[str, object]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dtype", "stage", "p50_ms", "p95_ms", "vram_mb", "samples"])
        for res in runs:
            for stage, xs in res["stages_ms"].items():
                if not xs:
                    continue
                w.writerow([
                    res["dtype"], stage,
                    f"{statistics.median(xs):.2f}",
                    f"{_pct(xs, 95):.2f}",
                    f"{res['stages_vram_mb'].get(stage, 0.0):.0f}",
                    len(xs),
                ])
            w.writerow([res["dtype"], "end_to_end_fps", f"{res['fps']:.2f}",
                        "", "", res["frames"]])
            w.writerow([res["dtype"], "peak_rss_mb", f"{res['peak_rss_mb']:.0f}",
                        "", "", ""])
            w.writerow([res["dtype"], "vram_overall_mb",
                        f"{res['vram_overall_mb']:.0f}", "", "", ""])


def _write_markdown(path: Path, runs: List[Dict[str, object]]) -> None:
    """A judge-friendly table. If >1 precision, emit a side-by-side block."""
    with open(path, "w") as f:
        f.write("# MASCAL edge — benchmark results\n\n")
        f.write(f"_Generated_: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Summary\n\n")
        f.write("| dtype | FPS | Peak RSS (MB) | Peak VRAM (MB) | Frames |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for res in runs:
            f.write(f"| {res['dtype']} | {res['fps']:.2f} | "
                    f"{res['peak_rss_mb']:.0f} | {res['vram_overall_mb']:.0f} | "
                    f"{res['frames']} |\n")

        f.write("\n## Per-stage (p50 / p95 ms, peak VRAM MB)\n\n")
        stages = ["person", "wound", "rppg"]
        header = "| stage |" + "|".join(f" {r['dtype']} p50 | {r['dtype']} p95 | {r['dtype']} VRAM "
                                        for r in runs) + "|\n"
        sep = "|---|" + "|".join([":---:"] * (3 * len(runs))) + "|\n"
        f.write(header)
        f.write(sep)
        for stage in stages:
            cells = [f"| {stage} "]
            for r in runs:
                xs = r["stages_ms"].get(stage, [])
                vram = r["stages_vram_mb"].get(stage, 0.0)
                if xs:
                    p50 = statistics.median(xs)
                    p95 = _pct(xs, 95)
                    cells.append(f"| {p50:.1f} | {p95:.1f} | {vram:.0f} ")
                else:
                    cells.append("| — | — | — ")
            f.write("".join(cells) + "|\n")

        # Precision delta row (first vs last) is a good talking point
        # for the deck: "FP16 runs at 1.7× FPS with 42% less VRAM."
        if len(runs) >= 2:
            base = runs[0]
            other = runs[-1]
            if base["fps"] > 0:
                fps_ratio = other["fps"] / base["fps"]
                vram_delta = (other["vram_overall_mb"] - base["vram_overall_mb"]) \
                             / max(1.0, base["vram_overall_mb"]) * 100
                f.write("\n## Precision delta\n\n")
                f.write(f"**{other['dtype']} vs {base['dtype']}**: "
                        f"{fps_ratio:.2f}× FPS, "
                        f"{vram_delta:+.1f}% VRAM.\n")


def main() -> None:
    args = parse_args()

    # Decide which precisions to run. The plan called this out as "FP16
    # vs INT8" — INT8 requires per-model quantization (graph surgery for
    # GDINO/SAM) that isn't stable enough for a one-click benchmark; we
    # expose an honest **precision sweep** over the dtypes torch can
    # toggle at runtime (fp32 / fp16 / bf16). Teams that wire in an INT8
    # backend can add its label to this list later.
    if args.compare_precisions:
        requested = [s.strip() for s in args.compare_precisions.split(",") if s.strip()]
    else:
        torch = _torch()
        default = "fp16" if (torch is not None and torch.cuda.is_available()) else "fp32"
        requested = [default]

    runs: List[Dict[str, object]] = []
    for dtype in requested:
        print(f"\n[bench] running pass at {dtype} for {args.duration:.0f}s…")
        res = _run_once(args, dtype)
        _print_run(res)
        runs.append(res)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_out = ROOT / "benchmarks" / f"results_{stamp}.csv"
    md_out = ROOT / "benchmarks" / f"results_{stamp}.md"
    _write_csv(csv_out, runs)
    _write_markdown(md_out, runs)
    print(f"\nSaved {csv_out}")
    print(f"Saved {md_out}")


if __name__ == "__main__":
    main()
