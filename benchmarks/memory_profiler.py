"""Lightweight memory profiler: samples RSS every 0.5s while the pipeline runs.

Run:
    python benchmarks/memory_profiler.py --duration 30
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--source", default="0")
    args = p.parse_args()

    cmd = [sys.executable, str(ROOT / "edge-node" / "main.py"),
           "--source", args.source, "--headless"]
    print("launching:", " ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT))
    ps = psutil.Process(proc.pid)

    samples = []
    t0 = time.time()
    try:
        while time.time() - t0 < args.duration and proc.poll() is None:
            time.sleep(0.5)
            try:
                rss = ps.memory_info().rss / (1024 * 1024)
                samples.append((time.time() - t0, rss))
            except psutil.NoSuchProcess:
                break
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=3)
        except Exception:
            proc.kill()

    if not samples:
        print("no samples")
        return
    peak = max(s[1] for s in samples)
    steady = sorted(s[1] for s in samples)[len(samples) // 2]
    print(f"peak RSS  : {peak:.0f} MB")
    print(f"median RSS: {steady:.0f} MB")

    out = ROOT / "benchmarks" / f"memory_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(out, "w") as f:
        f.write("t_seconds,rss_mb\n")
        for t, rss in samples:
            f.write(f"{t:.2f},{rss:.1f}\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
