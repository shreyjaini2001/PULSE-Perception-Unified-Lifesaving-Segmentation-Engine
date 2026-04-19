# MASCAL Triage Relay

> A perception relay worn by a lead medic that auto-captures per-victim context, builds a scene map, triages each patient to TCCC/SALT standards, and broadcasts a live casualty dashboard into ATAK — all running offline on commodity edge hardware.

**Hackathon problem statements addressed:** Meta #15 (MASCAL AI perception) & Meta #16 (Edge Inference)

See `MASCAL_Implementation_Plan.md` for the full design document.

---

## Project status — what's built

_Kept current with every session. Last updated: full CUDA stack online on RTX 4060 Laptop (8 GB VRAM)._

### Phases complete
- **Phase 0 — Scaffold:** repo tree, `requirements.txt`, `scripts/download_models.py`, `scripts/smoke_test.py`, config files (`scenarios.json`, `prompts.yaml`, `runtime.yaml`), `.gitignore`.
- **Phase 1 — MVP core loop:** `edge-node/main.py` entry point + capture loop. Person detection/tracking with YOLOv8-pose (`pipeline/person.py`). Wound + blood segmentation with Grounding DINO + SAM 2.1, with HSV color-heuristic fallback (`pipeline/wound.py`). rPPG HR/RR via MediaPipe FaceMesh + GRGB FFT (`pipeline/rppg.py`). Body-region mapping (`pipeline/body_pose.py`). Per-victim + scene state (`state/victim.py`, `state/scene.py`). WebSocket + HTTP dashboard (`broadcast/ws_server.py`, `dashboard/`).
- **Phase 2 — Clinical layer:** Whisper transcription (`pipeline/audio.py`). MARCH state machine (`state/march.py`). SALT tag suggestion with hard-gated GREY/BLACK (`state/salt.py`). Llama 3.2 MIST synthesis with rule-based fallback (`pipeline/llm.py`). Scenario mode config + hot-swap.
- **Phase 3 — Multi-device + ATAK:** polished dashboard (SALT colors, MARCH dots, anatomical avatar with pose-locked overlay, live transcript, event log). ATAK CoT bridge via `atakcots` + raw UDP fallback (`broadcast/atak_bridge.py`).
- **Phase 4 — Polish:** intervention timers (TQ countdown with warn/crit visual stages). Voice handoff via browser `speechSynthesis`. JSONL audit trail with actor attribution (`edge-node/audit.py`). Sentence-cased UI copy.
- **Phase 5 — Benchmarks + demo prep:** `benchmarks/fps_profiler.py`, `benchmarks/memory_profiler.py`, `benchmarks/results.md` template. `docs/DEMO_SCRIPT.md` with 2-minute runbook + fallbacks. `docs/PITCH_DECK.md` with 10-slide outline. `docs/ETHICS_AND_SCOPE.md`.
- **Phase 6 — GPU acceleration:** venv rebuilt on Python 3.10 (MediaPipe-compatible). PyTorch 2.6.0 + CUDA 12.4. `llama-cpp-python` 0.3.4 with pre-built CUBLAS wheel. Torch's bundled `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` copied alongside `llama.dll`; `llm.py` also calls `os.add_dll_directory` at import time so reinstalls survive. Heavy-model variants bumped (YOLOv8s-pose, Grounding DINO **base**, SAM 2.1 **small**, Whisper **small.en** float16, Llama 3.2 3B Q4_K_M fully GPU-offloaded).

### Model stack on this machine (all GPU)
| Stage | Model | VRAM | Device |
|---|---|---|---|
| Person + pose | `yolov8s-pose` (11 M) | ~150 MB | CUDA |
| Open-vocab grounding | `IDEA-Research/grounding-dino-base` (233 M) | ~1.0 GB | CUDA |
| Wound segmentation | `facebook/sam2-hiera-small` (46 M) | ~1.2 GB | CUDA |
| Vitals | MediaPipe FaceMesh + GRGB rPPG | CPU delegate (XNNPACK) | CPU |
| Voice | `faster-whisper small.en` (244 M) | ~800 MB | CUDA float16 |
| MIST synthesis | Llama 3.2 3B Instruct Q4\_K\_M (29 / 29 layers on GPU) | ~2.5 GB | CUDA |

**Measured steady-state:** 5.9 GB / 8 GB VRAM, ~6 FPS on webcam with 1 victim in frame. Llama inference ~0.9 s wall-clock for a short MIST prompt.

### Verified working (not just compiled)
- Full Python import graph and `py_compile` clean.
- `scripts/smoke_test.py` exercises MARCH → SALT → MIST end-to-end on a synthetic victim.
- `python -u edge-node/main.py --source 0` brings everything up in ~10 s: YOLO loads, Grounding DINO base on CUDA, SAM 2.1 small on CUDA, MediaPipe FaceMesh live, faster-whisper small.en on CUDA (float16), Llama 3.2 with 29/29 layers on GPU, HTTP dashboard serves 200, WebSocket accepts connections.
- `scripts/ws_smoke.py` round-trips dashboard control messages (set_scenario observed in subsequent broadcast).
- **All 6 models cached** via `scripts/download_models.py` (run from the venv): YOLOv8s-pose, Grounding DINO base, DINOv2 base, SAM 2.1 hiera-small **and** hiera-base-plus (both cached; config toggles which is loaded), Whisper small.en (CUDA float16), Llama 3.2 3B Q4\_K\_M.
- rPPG is actually live now: `[rppg] MediaPipe FaceMesh loaded.` appears in the banner, HR/RR update when a face is visible.

### Known caveats
- **Always use the venv Python.** `.\.venv\Scripts\python.exe <script>` — not plain `python` from PowerShell.
- **Python 3.10 is a hard requirement** on this machine for MediaPipe's legacy `mp.solutions` API (rPPG). 3.13 strips it; 3.11/3.12 would work too, but only 3.10 was already installed locally. MediaPipe is pinned at `0.10.14` in `requirements.txt` — newer 0.10.33 also dropped the legacy API.
- **llama-cpp-python needs CUDA runtime DLLs next to `llama.dll`.** `edge-node/pipeline/llm.py` auto-registers Torch's `lib/` via `os.add_dll_directory` at import time, but the three files (`cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll`) are also hand-copied into `.venv\Lib\site-packages\llama_cpp\lib\` as belt-and-braces.
- **Wound pipeline throttle.** Per-victim GDINO+SAM pass is throttled by `wound_scan_interval_seconds` in `runtime.yaml` (default 1.0 s). Dropping it to 0.35 s cuts FPS from ~6 to ~0.6; bumping to 2.0 s roughly doubles FPS. Wounds don't actually change frame-to-frame, so 1 s is the sweet spot.
- **SCAN mode vs OpenCV HUD.** In `operating_mode: scan`, persisted `wound_regions` stay empty until the medic taps **Scan this victim** on the dashboard. The local preview window still runs an optional **rotating HUD wound preview** (`scan_wound_preview_*` in `runtime.yaml`) so orange injury boxes are visible while you demo — that overlay is labeled **HUD preview** and does not change dashboard state until a real scan completes.
- **VRAM is tight at 97 % with SAM base-plus + Llama resident simultaneously.** Config ships with SAM `small` as a result; to use `base-plus`, disable Llama (`llm_enabled: false`) or expect OOM risk. Both SAM variants are pre-cached, so switching is instant.
- **First YOLO cold start is ~10 s** — README and `main.py` both use unbuffered stdout (`-u`) so startup progress is visible on Windows PowerShell.
- **Windows TIME\_WAIT on port restart** — `SO_REUSEADDR` enabled so rapid dev restarts work.
- **Gated Llama 3.2 on HF** — fall back to `Qwen2.5-3B-Instruct-GGUF` (same prompt format, ungated) if HF access isn't granted. Llama is already cached locally, so this is documented for other machines.

### Not yet done / next session
- Point a moulage image or prosthetic at the webcam and confirm wound masks render on the dashboard. (Pipeline is wired; not yet visually confirmed with real blood/wound mockup.)
- Record 3 real moulage scenario videos into `edge-node/demo/` (the "insurance policy" fallback).
- Populate `benchmarks/results.md` with numbers from `benchmarks/fps_profiler.py --duration 30` on the new GPU stack.
- 5× demo rehearsal per `docs/DEMO_SCRIPT.md`.
- (Optional stretch) Android capture client in `mobile-app/android/` — directory stubbed, not implemented.
- (Optional stretch) Swap GDINO base → SAM 2.1 base-plus by disabling Llama; demo the "max-accuracy" posture as a separate mode.

---

## Quick start

### 1. Install dependencies

Use Python 3.10 so MediaPipe's legacy `solutions` API (needed for rPPG) is available, and install PyTorch separately so you can pick a CUDA build:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1

# GPU (NVIDIA, CUDA 12.4 driver or newer):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# --- OR CPU-only fallback: ---
# pip install torch torchvision

pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/sam2.git"
```

For GPU-offloaded MIST synthesis, also install the CUDA-enabled `llama-cpp-python` wheel (Windows x64, Python 3.10, CUDA 12.4 prebuilt):

```powershell
pip install "llama-cpp-python==0.3.4" --upgrade --force-reinstall --no-cache-dir --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124
```

On Windows, `llama.dll` needs three CUDA runtime DLLs that PyTorch already ships. Copy them next to `llama.dll` once:

```powershell
$t = ".\.venv\Lib\site-packages\torch\lib"
$l = ".\.venv\Lib\site-packages\llama_cpp\lib"
Copy-Item "$t\cudart64_12.dll","$t\cublas64_12.dll","$t\cublasLt64_12.dll" -Destination $l -Force
```

(`edge-node/pipeline/llm.py` also calls `os.add_dll_directory` on Torch's lib folder at import time, so this survives a future `llama-cpp-python` reinstall.)

On first run, YOLOv8 auto-downloads its weights. Heavier assets (SAM 2.1, Grounding DINO, Llama 3.2) are optional — the pipeline degrades gracefully when they're absent:

- No SAM/GDINO → color-heuristic blood detection (no wound class labels).
- No Llama → rule-based MIST card template (still JSON, still valid).
- No MediaPipe FaceMesh → rPPG shows "HR pending" instead of fake numbers.
- No microphone / Whisper → dashboard-typed notes still work via the "Add a note" field.
- No GPU → flip `llm_enabled: false` and `whisper_compute_type: int8` in `edge-node/config/runtime.yaml`, and knock `gdino_model` down to `IDEA-Research/grounding-dino-tiny` / `sam_model` to `facebook/sam2-hiera-tiny`.

Pre-fetch all six models on a good connection with:

```powershell
python scripts/download_models.py
```

**Hugging Face gated checkpoints (`--profile max`).** Some ids (e.g. `facebook/sam3`, `facebook/dinov3-*`) require you to accept the license on the model page, then authenticate so downloads return 200 instead of 401:

```powershell
.\.venv\Scripts\huggingface-cli.exe login
# or: setx HF_TOKEN "hf_..."   # then open a new terminal
```

Without login, **SAM3** still delegates to Grounding DINO + SAM 2.1 (already working). The **anomaly prior** automatically falls back to the public `facebook/dinov2-base` when DINOv3 is unreachable.

### 2. Verify the install

```powershell
python scripts/smoke_test.py        # exercises the state layer + MIST template
python scripts/make_test_video.py   # synthesizes a 5s synthetic clip (no deps beyond cv2 + numpy)
```

### 3. Run the edge node

```powershell
python -u edge-node/main.py --source 0                                # webcam
python -u edge-node/main.py --source edge-node/demo/scenario_synthetic.mp4 --headless
```

Flags:

- `--source 0`             webcam index (default)
- `--source path/to.mp4`   video file for demo fallback
- `--source rtsp://...`    phone RTSP stream (stretch)
- `--scenario combat_blast` | `civilian_mci` | `fire_structure` | `mva` | `cbrn`
- `--mode scan` (default)  manual per-victim capture with face re-ID; records only persist on tap
- `--mode live`            continuous per-frame update (old behaviour; loses identity when victim leaves FOV)
- `--profile fast` | `balanced` (default) | `max`   model-tier preset (see `config/runtime.yaml`)
- `--no-llm`               skip Llama; use rule-based MIST templates
- `--no-sam`               skip SAM/GDINO; use color-based blood detection
- `--no-face`              disable InsightFace re-ID (victim ids fall back to tracker ids)
- `--headless`             no preview window

### Scan mode vs Live mode

**Scan mode is the default** and is the primary interaction: the medic points
the camera at a casualty and taps "Scan this victim" on the tile. The edge
node runs SAM on the person bbox to produce a silhouette, body-gates every
wound/blood candidate against that silhouette (no more red-wall false
positives), embeds the face with InsightFace so the casualty keeps the same
callsign on re-entry, and persists a frozen `ScanRecord` with arrow-annotated
wound crops. Only scans persist identity + injury context across time.

**Live mode** re-enables the old continuous-update path and is one toggle
away in the dashboard navbar. Useful for smoke-testing the pipeline but it
**does not retain victim identity when they leave the FOV** — the
confirmation modal warns the medic about that before switching.

### Model profiles

Profiles overlay the `pipeline:` block in `runtime.yaml` at startup:

| Profile | Person | Detection | Segmentation | ASR | LLM |
|---|---|---|---|---|---|
| `fast` | YOLOv8s-pose | GDINO tiny | SAM 2.1 small | Whisper small.en | Llama 3.2 3B |
| `balanced` (default) | YOLOv8m-pose | GDINO base | SAM 2.1 base+ | Whisper small.en | Llama 3.2 3B |
| `max` | YOLOv8x-pose | **SAM 3.1** (text-prompt detect+segment) | SAM 2.1 large fallback | Whisper medium.en | Llama 3.1 8B if available |

`max` loads `facebookresearch/sam3` at runtime. If the repo or the gated
`facebook/sam3.1` checkpoint isn't available it logs and falls back to the
GDINO+SAM2 path — nothing crashes. Profiles can be hot-swapped from the
dashboard navbar; the wound backend rebuilds on a worker thread.

### Face re-ID (InsightFace)

`pipeline/face_reid.py` uses InsightFace `buffalo_l` (ArcFace 512-D
embeddings, cosine ≥ 0.45 to match). Each victim's face embedding is stored
on the `Victim` record so callsigns persist across reconnects. Install:

```powershell
pip install insightface onnxruntime-gpu       # or onnxruntime on CPU-only hosts
```

First run downloads ~400 MB of ONNX weights into `~/.insightface`.

> **Windows gotcha — `opencv-python-headless` collision.** The InsightFace
> wheel lists `opencv-python-headless` as a dependency, and because both
> `opencv-python` and `opencv-python-headless` register as the `cv2` module
> the last one installed wins. The headless build ships without highgui, so
> `cv2.imshow` raises `"The function is not implemented"`. The edge node
> auto-disables its preview window on that error (dashboard at
> http://localhost:8080/ is unaffected), but if you want the live preview
> back, force the GUI build to the front:
>
> ```powershell
> pip uninstall -y opencv-python-headless
> pip install --force-reinstall opencv-python
> ```
>
> Or run headless intentionally with `python edge-node/main.py --headless`.

### 4. Open the dashboard

Open <http://localhost:8080/> in a browser (or on any device on the same WiFi). The edge node serves the dashboard via a local HTTP server and pushes state over WebSocket at `ws://<host>:8081/`.

### 5. (Optional) ATAK integration

Set `atak.enabled: true` in `edge-node/config/runtime.yaml` (or remove `--no-atak`) and point `atak.host` at the ATAK-CIV tablet's IP. Confirmed casualty tags emit CoT markers to UDP port 4242.

---

## Repo layout

```
mascal-triage/
├── edge-node/       # Python — AI pipeline + state + broadcast
│   ├── pipeline/    # Perception stages (person, wound, rppg, audio, llm)
│   ├── state/       # Victim / MARCH / SALT / Scene data models
│   ├── broadcast/   # WebSocket + ATAK CoT bridge
│   ├── config/      # Scenario configs, prompts, runtime settings
│   ├── demo/        # Pre-recorded scenario videos (placeholder)
│   └── main.py      # Entry point
├── dashboard/       # Vanilla JS receiver UI
├── benchmarks/      # fps/memory profilers
├── scripts/         # Model download, utilities
├── docs/            # Demo script, pitch deck outline, ethics notes
└── models/          # Local model cache (gitignored)
```

---

## Design principles

1. **The AI never decides triage autonomously.** Every SALT tag is AI-suggested, human-confirmed. Hard-gated: GREY/BLACK require explicit confirmation.
2. **Offline-first.** Nothing leaves the edge node except intentional broadcasts on the local mesh.
3. **Graceful degradation.** If SAM is slow, fall back to bounding boxes. If Llama fails, fall back to templates. If rPPG is noisy, show "pending" — never invent vitals.
4. **Audit trail.** Every state change is logged to `logs/audit_*.jsonl` with actor attribution (AI-suggested vs. human-confirmed).

---

## Scenario modes

| Mode | Triggers | Focus |
|---|---|---|
| `combat_blast` | Shrapnel, amputation, tourniquet | TCCC MARCH → 9-line MEDEVAC |
| `civilian_mci` | Laceration, crush, impalement | SALT + EMS PCR |
| `fire_structure` | Burn, soot, blistered skin | Airway-first; rough burn % |
| `mva` | Glass lac, crush, deformity | START-style triage |
| `cbrn` | Chemical burn, blistering | Flag decontamination needs |

Hot-swap at runtime via the dashboard dropdown.

---

## License / scope

Hackathon prototype. Decision *support* only — not decision-*making*, not FDA-cleared, not for clinical use. See `docs/ETHICS_AND_SCOPE.md`.
