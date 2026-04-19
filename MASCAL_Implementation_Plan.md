# MASCAL Triage Relay — Hackathon Implementation Plan

**Problem statements addressed:** Meta #15 (MASCAL AI perception) and Meta #16 (Edge Inference)
**Judge-facing pitch:** "A perception relay worn by a lead medic that auto-captures per-victim context, builds a scene map, triages each patient to TCCC/SALT standards, and broadcasts a live casualty dashboard into ATAK — all running offline on commodity edge hardware."

---

## 0. How to use this document

Read front-to-back once before you start coding. After that, work phase-by-phase. Each phase block is designed to be roughly self-contained: **Objective → Dependencies → Tasks → Tech decisions → Acceptance criteria → Fallback**. Don't start phase N+1 until phase N's acceptance criteria pass, even if it hurts. A demo that does 60% of the pipeline end-to-end beats a demo that does 100% of the pipeline with a broken seam at the final step.

---

## 1. AI coding assistant — what to use

**Primary recommendation: Cursor + Claude Sonnet 4.6 (or Opus 4.7 if you have access).**

Reasoning:
- Cursor's Cmd+K inline-edit loop is the fastest way to iterate on glue code and model wiring, which is 70% of this project.
- Cursor's Composer (agent) mode handles multi-file changes well — useful when you're refactoring the state model across pipeline, broadcast, and dashboard at once.
- Tab completion with codebase context is genuinely valuable when you're calling unfamiliar Meta/HuggingFace APIs.

**Secondary recommendation: Claude Code in the terminal.**

Use it alongside Cursor for:
- One-shot project scaffolding at the start ("create the repo structure per this plan and stub every file")
- Benchmark scripts and dev-ops (running profilers, comparing FP16 vs INT8 numbers)
- The ATAK CoT bridge (Python, heavy shell work)
- Agent-mode debugging loops where you want Claude to run-test-edit-rerun without you clicking

**Why not Codex / ChatGPT as primary:** OpenAI's models are fine, but Claude models (particularly Sonnet 4.6+ and Opus 4.7) are currently stronger on the kind of ML-pipeline + systems-integration work this project demands, and Cursor makes the integration seamless. If one teammate prefers Codex, let them use it — don't force-standardize. But the main build should be Cursor + Claude.

**Cost note:** Cursor Pro ($20/mo, 1-month trial works for the hackathon). Claude Code runs off an Anthropic API key or Claude Pro/Max subscription.

---

## 2. Team split recommendation

Assuming 3–4 people. If fewer, collapse roles.

| Role | Primary responsibility | Phase ownership |
|---|---|---|
| **Pipeline / ML** (Person A) | Python edge-node, models, segmentation, rPPG | Phases 1, 2, 5 |
| **Clinical + LLM** (Person B) | MARCH/SALT logic, Llama prompts, scenario configs | Phases 2, 4 |
| **Mobile + Dashboard** (Person C) | Capture client, receiver UI, local networking | Phases 1, 3 |
| **Integration + Demo** (Person D) | ATAK bridge, mesh, demo scenarios, deck | Phases 3, 5 |

If 3 people: merge B+D. If 2 people: merge A+B and C+D.

---

## 3. Architecture — what we're building

**Deployment model:** Two-tier edge. The lead medic carries a phone (capture + light inference) and a rugged compute node in their pack (laptop-class, heavy inference). Receiver medics carry phones/tablets that consume the broadcast. For the hackathon demo we collapse this: one laptop plays both the "phone" and the "compute node" role, with a phone acting as an IP camera for authenticity.

**Data flow:**
1. Camera + mic stream into the edge-node Python process.
2. A shared DINOv3-style vision backbone runs once per frame; multiple heads consume its features.
3. Per-frame outputs (person detection, wound masks, blood regions, pose) get aggregated into a persistent per-victim state.
4. Voice notes are transcribed by Whisper; MIST cards synthesized by Llama 3.2.
5. MARCH and SALT rules drive a triage-tag suggestion (human-confirmed).
6. Final state broadcasts over local WebSocket to the receiver dashboard and over CoT/UDP into ATAK.

**Critical design decision:** The AI never decides triage autonomously. It *suggests* a SALT color and populates MARCH fields; the lead medic confirms with one tap. This is both an ethics posture and a liability shield — say it in the pitch.

---

## 4. Tech stack commitments

Pin these early. Don't let anyone introduce alternatives mid-hackathon.

### 4.1 Languages
- **Python 3.11** — edge-node / all ML. Use `uv` for package management (faster than pip).
- **JavaScript (vanilla + WebSocket)** — receiver dashboard. Don't add React for this.
- **Kotlin** — Android capture client (only if we get to Phase 3 stretch).

### 4.2 ML frameworks
- **PyTorch 2.4+** — primary training/inference
- **ONNX Runtime** — for SAM variants that have ONNX exports
- **llama.cpp / llama-cpp-python** — Llama 3.2 quantized inference (GGUF format, easiest path)
- **MediaPipe** — face mesh, pose, hand detection (pre-built, no training)
- **whisper.cpp / faster-whisper** — audio transcription

### 4.3 Models — committed choices with upgrade paths

| Role | MVP choice (use this) | Upgrade target (if time) | HuggingFace / source |
|---|---|---|---|
| Vision backbone | `facebook/dinov2-base` | `facebook/dinov3-convnext-small-pretrain-lvd1689m` | HF |
| Open-vocab detection | **Grounding DINO** (`IDEA-Research/grounding-dino-tiny`) | SAM 3 concept segmentation | HF |
| Segmentation | **SAM 2.1** (`facebook/sam2-hiera-tiny`) | SAM 3 (`facebook/sam3`) | HF |
| Body / pose | MediaPipe BlazePose | SAM 3D Body | Google / Meta |
| LLM | **Llama 3.2 3B Q4_K_M** (GGUF, via llama.cpp) | Llama 3.2 1B SpinQuant on ExecuTorch | HF (`bartowski/Llama-3.2-3B-Instruct-GGUF`) |
| Audio | **Whisper tiny.en** | Whisper small.en | openai / HF |
| rPPG | Custom GRGB implementation | `pyVHR` library | Implement from MDPI paper |

**Why these MVP picks:** SAM 2.1 and Grounding DINO are the most-documented, most-tutorialized zero-shot segmentation combo as of early 2026. You will find working example code in 15 minutes. SAM 3.1 is the right upgrade — mention it in your pitch as "our production target" — but burning hackathon time fighting newer model tooling is a losing move. Same logic for DINOv2 vs v3: v2 is rock-solid, v3 is the aspirational target.

**Critical constraint:** Llama 3.2 is gated on HuggingFace. Request access Day -3 (gets approved in hours, not instantly). If it's not approved by hackathon start, use `Qwen2.5-3B-Instruct-GGUF` as a drop-in — the prompt format works the same way and it's ungated.

### 4.4 Infrastructure
- **Local mesh**: WebSocket over WiFi for the hackathon. Mention goTenna / Rajant Kinetic Mesh as production path; don't buy hardware.
- **ATAK bridge**: `atakcots` Python package (pip-installable, pushes CoT markers without needing a TAK server).
- **ATAK client**: ATAK-CIV on a tablet (free, Play Store). Alternatively iTAK on an iPad.
- **Receiver dashboard**: Plain HTML + vanilla JS + WebSocket. Served by the edge-node over `http.server`.

---

## 5. Repo structure

Scaffold this Day 0. Commit the directory tree as empty files so nothing blocks imports.

```
mascal-triage/
├── README.md
├── IMPLEMENTATION_PLAN.md          # This document
├── requirements.txt                # Or pyproject.toml if using uv
│
├── edge-node/                      # Python — main AI + state + broadcast
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── backbone.py             # DINOv2/v3 feature extraction
│   │   ├── person.py               # Detection + tracking
│   │   ├── wound.py                # Grounding DINO + SAM segmentation
│   │   ├── body_pose.py            # MediaPipe BlazePose → body graph
│   │   ├── rppg.py                 # Face mesh + GRGB → HR/RR/SpO2
│   │   ├── audio.py                # Whisper + airway sound classifier
│   │   └── llm.py                  # Llama 3.2 for MIST synthesis
│   ├── state/
│   │   ├── victim.py               # Per-victim data model
│   │   ├── march.py                # MARCH state machine
│   │   ├── salt.py                 # SALT tagging rules
│   │   └── scene.py                # Scene-level aggregation
│   ├── broadcast/
│   │   ├── ws_server.py            # WebSocket to dashboard
│   │   └── atak_bridge.py          # CoT publisher via atakcots
│   ├── config/
│   │   ├── scenarios.json          # Mode switcher (combat/civilian/fire/...)
│   │   ├── prompts.yaml            # SAM + Llama prompts per mode
│   │   └── runtime.yaml            # Device config, thresholds
│   ├── demo/
│   │   ├── scenario_blast.mp4      # Pre-recorded moulage footage
│   │   ├── scenario_fire.mp4
│   │   └── scenario_mva.mp4
│   └── main.py                     # Entry point
│
├── dashboard/                      # Receiver UI
│   ├── index.html
│   ├── app.js
│   └── styles.css
│
├── mobile-app/                     # Optional — stretch goal
│   └── android/                    # Kotlin + CameraX
│
├── benchmarks/
│   ├── fps_profiler.py
│   ├── memory_profiler.py
│   └── results.md                  # Populated Day 3
│
└── docs/
    ├── DEMO_SCRIPT.md
    ├── PITCH_DECK.md
    └── ETHICS_AND_SCOPE.md
```

---

## 6. Phase 0 — Pre-hackathon setup (do before Day 1)

**Objective:** Remove every environment blocker so Day 1 starts with writing product code, not fighting CUDA.

**Duration:** 4–6 hours of prep, spread across a few evenings.

### 6.1 Hardware & access
- Two Android phones (any flagship from 2022+), one tablet with ATAK-CIV installed, one laptop (ideally with a discrete GPU — RTX 3060 or better, 16GB+ RAM). M-series Mac works too; CUDA just makes SAM inference faster.
- HuggingFace account with access tokens. **Request Llama 3.2 gated access immediately** at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct. Request DINOv3 access too if you want to try the upgrade.
- A GitHub repo with the structure above scaffolded. Protect `main`; work on branches.

### 6.2 Software environment

```bash
# Python environment with uv (faster than pip)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision transformers opencv-python mediapipe \
  llama-cpp-python faster-whisper websockets atakcots pyyaml \
  onnxruntime-gpu pillow numpy scipy

# SAM 2
uv pip install git+https://github.com/facebookresearch/sam2.git

# Grounding DINO
uv pip install groundingdino-py
```

If on Mac without CUDA, use `onnxruntime` (not `-gpu`) and install PyTorch with MPS backend support. Llama.cpp will use Metal automatically.

### 6.3 Pre-download models

Write a `scripts/download_models.py` that fetches every model ID to a local `models/` cache. Run it Day 0 because downloading 5GB of Llama at the hackathon venue on shared WiFi will ruin your morning.

Specifically download:
- `facebook/sam2-hiera-tiny` (SAM 2 tiny checkpoint)
- `IDEA-Research/grounding-dino-tiny`
- `facebook/dinov2-base`
- `bartowski/Llama-3.2-3B-Instruct-GGUF` (pick `Q4_K_M` variant, ~2GB)
- Whisper tiny.en (auto-downloaded by faster-whisper on first run)

### 6.4 Demo assets
- Buy a moulage kit on Amazon ($40–60). Practice applying simulated lacerations, burns, GSW wounds to a mannequin or willing teammate.
- Pre-record 3 scenario videos (blast, fire, MVA) with moulage-painted victims. 30–60 seconds each. These are your insurance policy — if the live demo glitches, you fall back to canned footage and the pipeline still runs.
- Print laminated SALT tag color references for the demo table — judges will notice.

### 6.5 Clinical reference materials
Download and skim:
- TCCC Guidelines (Jan 2024) — memorize MARCH order.
- SALT Triage Algorithm (CHEMM) — memorize the five categories and decision rules.
- Example MIST cards — understand the fields.

**Acceptance criteria for Phase 0:**
- [ ] Every team member can `python edge-node/main.py` and see a stub "hello world" response.
- [ ] Every model is cached locally. `ls models/` shows the downloaded files.
- [ ] ATAK-CIV installed on the tablet, opening without crashing, showing a map.
- [ ] Moulage kit and demo mannequin on hand.

---

## 7. Phase 1 — MVP core loop (first ~8–10 hours)

**Objective:** A single Python process on a laptop that takes a webcam feed, detects people, segments wounds and blood on them, estimates HR from faces, and publishes a per-victim JSON state over WebSocket to a browser dashboard.

**Why this phase first:** It proves the core pipeline. Nothing else matters if the AI doesn't work. Defer triage logic, LLM, ATAK, and mobile to later phases.

### 7.1 Tasks

**Task 1.1 — Capture + frame loop** (1–2 hours)
Build `edge-node/main.py` that opens `cv2.VideoCapture(0)` (or a video file), reads frames at ~10 FPS, and passes them through a stub pipeline.

**Task 1.2 — Person detection + tracking** (2 hours)
Use `ultralytics` YOLOv8-pose or Grounding DINO with prompt `"person"`. Assign each detection a persistent ID via IoU-based tracking (DeepSORT is overkill; a simple tracker works). Emit detections with ID, bounding box, keypoints.

**Implementation note:** YOLOv8-pose is faster to wire up and gives you pose keypoints for free (which you'll use later for body-graph mapping). Start with YOLOv8. Mention DINOv3 as the backbone in the pitch — you can legitimately say "we use DINOv3 features for re-identification" if you're honest that the primary detector is YOLO + DINOv3 features for embeddings.

**Task 1.3 — Wound and blood segmentation** (3 hours)
For each detected person's bounding box, run Grounding DINO with the text prompt (e.g., `"laceration . blood . burn . tourniquet . exposed bone"`). Take the bounding boxes GD returns, feed them into SAM 2.1 as box prompts, get back per-region masks.

**Implementation note:** This is the most fragile part of the pipeline. Start with a clean test image of your moulage mannequin and tune prompts until wound recall looks good. Document your prompt set in `config/prompts.yaml`. Add a confidence threshold — below 0.3, don't emit.

**Task 1.4 — rPPG heart rate estimation** (2 hours)
Use MediaPipe FaceMesh to get forehead + cheek landmarks. Extract mean RGB from those ROIs over a rolling 10-second window. Apply the GRGB algorithm (implement from the MDPI paper, ~100 lines of NumPy). Output: HR in BPM, confidence score.

**Implementation note:** rPPG needs ~10 seconds of stable face visibility to produce a reliable number. During the demo, when the medic scans a victim, hold the face in frame for at least that long. Cache last known HR per victim ID.

**Task 1.5 — State aggregation** (1 hour)
Build `state/victim.py` dataclass:
```python
@dataclass
class Victim:
    id: str
    bbox: Tuple[int, int, int, int]
    keypoints: List[Tuple[float, float]]
    wound_masks: List[WoundRegion]
    blood_regions: List[BloodRegion]
    vitals: Vitals  # HR, RR, SpO2, confidence
    last_seen: float
    tag: Optional[SaltTag]  # None until Phase 2
```

A global `Scene` holds `Dict[str, Victim]`. Update on every frame.

**Task 1.6 — WebSocket broadcast + browser dashboard** (2 hours)
`broadcast/ws_server.py` runs a WebSocket endpoint. On every state update, emit a JSON snapshot to all connected clients. `dashboard/index.html` connects, renders a grid of victim tiles with bbox, vitals, and wound-count. No styling effort yet — plain boxes with text.

### 7.2 Acceptance criteria
- [ ] Point laptop webcam at a person; their body is detected, tracked, face mesh lands correctly.
- [ ] Paint a moulage wound; mask appears over it within ~500ms.
- [ ] After 10 seconds of face-in-frame, HR shows up in the tile and is within ±10 BPM of a pulse oximeter.
- [ ] Two browser tabs open on the dashboard both show the same live state.

### 7.3 Fallback if something fails
- If Grounding DINO is too slow: drop to every-5th-frame inference for wound masks, interpolate in between.
- If rPPG signal is noisy: fall back to MediaPipe's built-in face-presence detection and display "HR estimation requires stable face-in-frame" as a status string. Judges will respect the honest UX.
- If SAM 2.1 is unstable: Grounding DINO bounding boxes alone are enough for the MVP. Masks are nice-to-have, not critical.

---

## 8. Phase 2 — Clinical intelligence layer (next ~6–8 hours)

**Objective:** Turn raw detections into a structured clinical record. Add voice transcription, MARCH state, SALT tag suggestion, and Llama-generated MIST cards.

### 8.1 Tasks

**Task 2.1 — Voice transcription** (1.5 hours)
Wire `faster-whisper` with `tiny.en` model. Run in a separate thread/process consuming audio from the mic. Emit transcribed text chunks with timestamps. Store the last 30 seconds of transcript per active victim.

**Implementation note:** The lead medic's voice is the primary input. Expect domain jargon ("GSW", "tension pneumo", "TQ at 14:02"). Whisper tiny.en handles most of this; if confidence is low, attach the raw audio clip to the victim record for human review.

**Task 2.2 — MARCH state machine** (2 hours)
Build `state/march.py`. A per-victim MARCH record is five fields:
```python
@dataclass
class MarchState:
    massive_hemorrhage: Status  # none / suspected / confirmed / treated
    airway: Status              # clear / at_risk / compromised / managed
    respiration: Status         # normal / distressed / absent / assisted
    circulation: Status         # stable / shock / critical / arrest
    head_hypothermia: Status    # normal / tbi_suspected / hypothermic
```

Derive each field from:
- **M**: blood region total area > threshold, or "tourniquet" in wound masks, or transcript mentions "bleeding" / "hemorrhage" / "TQ"
- **A**: audio classifier detects gurgling/gasping, or transcript mentions "airway"
- **R**: rPPG respiratory rate out of 8–30 band, or chest rise not detected, or transcript flags
- **C**: rPPG HR out of 60–120 band, or pallor from skin-tone analysis, or rPPG can't lock
- **H**: pose keypoints show unusual head position, or transcript mentions "TBI" / "unconscious"

Each field outputs both a machine-derived status and a confidence. Human confirmation overrides.

**Task 2.3 — SALT tag suggestion** (1.5 hours)
Build `state/salt.py`. Implement the SALT flowchart verbatim:

```
if not breathing after airway maneuver → BLACK (Dead)
elif obeys commands AND has peripheral pulse AND not in distress → GREEN (Minimal)
elif not obeying commands OR no peripheral pulse OR respiratory distress OR uncontrolled hemorrhage:
    if likely to survive given resources → RED (Immediate)
    else → GREY (Expectant)
else → YELLOW (Delayed)
```

Inputs come from the MARCH state + detections. Output is a suggested color + reasoning string. Human confirms with a tap on the dashboard.

**Critical:** Never auto-emit GREY (Expectant) or BLACK (Dead). Always require human confirmation for those two. Code this in as a hard gate.

**Task 2.4 — Llama MIST card synthesis** (2 hours)
Launch `llama-cpp-python` server with Llama 3.2 3B Q4_K_M. Build a prompt template that takes all the structured signals and emits a MIST card in JSON:

```
System: You are a MASCAL scribe. Output ONLY a valid JSON MIST card.

User: Victim ID: {id}
Mechanism (from scenario mode): {mechanism}
Injuries detected: {wound_list_with_locations}
Vitals: HR={hr}, RR={rr}, SpO2={spo2}
Medic's voice notes: "{transcript}"
MARCH status: {march_state}

Emit JSON with keys: mechanism, injuries, signs, treatment, notes.
Keep each field under 20 words. If data is missing, use "unknown".
```

Validate JSON output with Pydantic. On parse failure, retry once with a stricter prompt; on second failure, fall back to a rule-based template.

**Implementation note:** Keep Llama context short — under 500 input tokens, 200 output. On a laptop with Q4_K_M, you should get 30–50 tokens/sec, which means a MIST card in 4–6 seconds. Trigger generation only when the medic hits "confirm scan" on a victim, not on every frame.

**Task 2.5 — Scenario mode loader** (1 hour)
Build `config/scenarios.json` with entries for each mode (combat_blast, civilian_mci, fire_structure, mva, cbrn). Each specifies:
- Prompt list for Grounding DINO
- Triage scheme (TCCC+SALT vs SALT vs CBRN-specific)
- Llama system prompt adjustments
- Default sort order

Dashboard has a dropdown; changing it hot-swaps the config without restarting.

### 8.2 Acceptance criteria
- [ ] Speak a note over the laptop mic; text appears in the victim's transcript panel.
- [ ] A mannequin with a simulated tourniquet and large painted blood region auto-suggests "Immediate (Red)" with "massive hemorrhage — TQ applied" as reasoning.
- [ ] A standing, moving person with no injuries auto-suggests "Minimal (Green)".
- [ ] Clicking "Generate MIST" on a victim tile produces a formatted MIST card within 6 seconds.
- [ ] Changing scenario mode from "Combat" to "Fire" changes the Grounding DINO prompt set (verify by checking which wound categories are detected).

### 8.3 Fallback
- If Llama inference is too slow: use a handwritten template that fills a MIST card from the structured fields. Lose a little polish, keep the demo working.
- If audio classifier doesn't ship: skip the airway sound detection; MARCH "A" status derives only from transcript mentions.

---

## 9. Phase 3 — Multi-device + ATAK integration (next ~6–8 hours)

**Objective:** The lead's laptop publishes state to (a) receiver medics watching a browser dashboard on their own phones/tablets, and (b) an ATAK-CIV tablet showing casualty markers on a map.

### 9.1 Tasks

**Task 3.1 — Multi-client WebSocket broadcast** (1 hour)
Already done in Phase 1, but now: load-test with 3 concurrent browser clients, make sure state stays consistent. Add a simple reconnection protocol (client resends last-seen-timestamp; server replays any newer events).

**Task 3.2 — Dashboard polish** (2–3 hours)
This is where you win demo polish points. Upgrade the dashboard from "plain boxes" to production-looking:
- Victim tiles colored by SALT tag (red/yellow/green/grey/black border)
- MARCH dots (5 small status indicators per tile)
- Body avatar (use MediaPipe pose keypoints to draw a stick figure with wound pins)
- Scene map placeholder (scrollable image with victim markers — we're not doing real SLAM)
- Top bar shows scenario mode, scan count, time since last update
- "Tap to tag" workflow: click a victim tile → see expanded view → confirm/override tag → tap to publish

Keep vanilla JS. The whole thing is ~800 lines.

**Task 3.3 — ATAK CoT bridge** (2 hours)
Build `broadcast/atak_bridge.py`. For each confirmed-tagged victim, emit a CoT XML message via `atakcots`:

```python
from atakcots import CotConfig, push_cot

cot_config = CotConfig(
    uid=f"mascal.victim.{victim.id}",
    lat=victim.geo_lat,
    lon=victim.geo_lon,
    type="a-f-G-E-V-C",  # CASEVAC marker type
    callsign=f"Casualty-{victim.id}",
    remarks=mist_card_as_string,
    stale_seconds=600,
)
push_cot(cot_config, atak_tablet_ip)
```

Emit on state change, not every frame (avoid flooding). Attach the MIST card to the `remarks` field so it shows in ATAK's marker detail panel.

**Implementation note:** You won't have real GPS during the demo. Hardcode a lat/lon per victim based on their position in the scene layout ("corner 1 = 38.8895,-77.0353" type mapping), or generate small random offsets from the demo venue's coordinates. Judges understand.

**Task 3.4 — Phone as capture client** (2 hours — stretch)
If time permits: use DroidCam (free app) or IP Webcam on the phone to stream the phone's camera to the laptop over WiFi. Now the lead "walks around with the phone" and the pipeline sees what the phone camera sees. Configure the edge-node to read from the phone's RTSP stream instead of the laptop webcam.

If this doesn't work: use the laptop webcam held by the presenter and walk it around. Same demo experience.

### 9.2 Acceptance criteria
- [ ] Dashboard looks *deliberate*. Someone glancing at it understands who needs help first.
- [ ] Two browser clients on different devices (one tablet, one phone) stay in sync within 1 second.
- [ ] Confirming a Red tag on the dashboard makes a red pin appear on the ATAK-CIV tablet's map within 3 seconds.
- [ ] Clicking the ATAK pin shows the full MIST card.

### 9.3 Fallback
- If `atakcots` has issues: write a tiny Python script that socket-sends raw CoT XML over UDP to port 4242 on the ATAK tablet. That's what `atakcots` does internally. Total fallback: ~30 lines.
- If ATAK refuses to display our markers: screenshot a working CoT marker from the docs and photoshop our data in for the slide. *Don't do this in the live demo — only as a desperate backup image in the deck.*

---

## 10. Phase 4 — Scenario modes + polish (next ~4–6 hours)

**Objective:** Make the system feel like a product, not a prototype. Prove scenario flexibility. Tighten every demo moment.

### 10.1 Tasks

**Task 4.1 — Full scenario mode implementation** (2 hours)
Finish the mode switcher that was stubbed in Phase 2. Each mode should demonstrably change behavior:
- **Combat/blast**: detects "shrapnel," "amputation," "tourniquet." Generates 9-line MEDEVAC draft.
- **Civilian MCI**: detects "laceration," "crush," "impalement." Generates EMS patient care report.
- **Fire/structure**: detects "burn," "soot." Airway-first priority. Calculates rough burn %.
- **MVA**: detects "glass laceration," "crush injury." START-style triage.
- **CBRN**: detects "chemical burn," "blistering." Flags decontamination needs.

Have a demo script that flips modes mid-pitch to show this off.

**Task 4.2 — Intervention timers** (1 hour)
When a tourniquet is detected (or when medic says "TQ on"), start a 2-hour countdown on the victim tile. Visual alerts at 1hr / 1.5hr / 2hr. Small feature, big clinical respect from judges who know TCCC.

**Task 4.3 — Voice handoff burst** (1 hour)
Add a button to each victim tile: "Play handoff." Uses browser's built-in `speechSynthesis` to read a 5-second TTS summary: *"Bravo-3, immediate. GSW right thigh. Tourniquet applied fourteen oh two. Heart rate one thirty-five. Unresponsive to voice."* Tiny code, huge demo moment.

**Task 4.4 — Audit trail** (1 hour)
Every state change logs to `logs/audit_{timestamp}.jsonl`. Fields: timestamp, victim_id, event_type, previous_state, new_state, actor (AI vs human). This is the "post-incident review" and "liability shield" talking point.

**Task 4.5 — UX and copy polish** (1–2 hours)
- Sentence-case every label. No "TRIAGE" shouting.
- Remove jargon from anything the user sees (use "critical bleeding" not "massive hemorrhage" in UI; internal fields can keep TCCC terms).
- Add a subtle "scanning" animation on the active victim tile.
- Make sure color contrast works in both bright (outdoor) and dim (indoor) environments — this matters because judges may test your demo in unexpected lighting.

### 10.2 Acceptance criteria
- [ ] Flipping scenario mode visibly changes what the system detects and recommends.
- [ ] Tourniquet timer counts down on the tile.
- [ ] Handoff button produces a coherent spoken summary.
- [ ] Every user-facing string has been read by at least one non-team-member and deemed clear.

---

## 11. Phase 5 — Benchmarks + demo prep (final ~4–6 hours)

**Objective:** Generate the numbers that make judges believe this runs at the edge. Rehearse the demo until it's muscle memory.

### 11.1 Tasks

**Task 5.1 — Benchmarks** (2 hours)
Write `benchmarks/fps_profiler.py` that:
1. Runs the full pipeline on a 30-second test video.
2. Logs per-stage latency (backbone, GDINO, SAM, rPPG, LLM).
3. Reports peak RAM, peak VRAM.
4. Compares FP16 baseline vs INT8 quantized where applicable.

Produce a single results table for the pitch deck. Target numbers:
- End-to-end pipeline: 8–12 FPS on laptop (extrapolate to "~20 FPS on Snapdragon 8 Gen 3 NPU").
- RAM: ~1.1 GB.
- Llama MIST generation: <6 seconds per card.
- CoT publish latency: <500ms from tag confirmation to ATAK pin.

**Task 5.2 — Deck (~10 slides)** (2 hours)
1. Title + team + one-liner
2. The problem (medic perception at scale — use the "24% potentially survivable" stat from the problem statement)
3. Current state + precedents (mention PRONTO and DARPA Triage Challenge briefly — shows you did research)
4. Our approach (architecture diagram)
5. AI pipeline (DINOv3 backbone + SAM + Llama with edge benchmarks)
6. Clinical alignment (TCCC MARCH + SALT + MIST)
7. Scenario modes (show the 5-mode table)
8. Deployability (ATAK integration, mesh-ready)
9. Ethics + scope (human-in-the-loop, data stays on device, audit trail)
10. Roadmap to production (SAM 3.1, DINOv3 upgrade, actual Ray-Ban integration, field trials with unit medics)

**Task 5.3 — Demo rehearsal** (2 hours)
Run the 2-minute demo end-to-end at least 5 times. Identify every failure mode. Stage workarounds for each. Print a physical demo script card for the presenter.

Demo script (2 minutes):
- 0:00 — "Mass casualty incident: three victims with blast-pattern injuries." (gesture at moulage mannequins)
- 0:15 — Walk up with phone/laptop, scan victim 1. Voice: "GSW right flank. Tourniquet applied." Point at screen: detection, wound mask, vitals.
- 0:40 — SALT suggestion appears. Confirm Red with a tap. CoT fires.
- 0:50 — Cut to tablet: ATAK shows red pin with MIST card in detail panel.
- 1:05 — Second device: receiver dashboard shows updated victim with MARCH dots.
- 1:20 — Flip scenario mode to "Fire." Scan victim 2 (burn moulage). Show different concept prompts firing, different output format.
- 1:40 — Flip to benchmarks slide: FPS, memory, offline-capable.
- 1:55 — Ethics + roadmap card. Done.

### 11.2 Acceptance criteria
- [ ] Benchmarks table generated, numbers verified reproducible.
- [ ] Deck polished, no typos, one concept per slide.
- [ ] Demo can be run start-to-finish in 2 minutes with no manual interventions beyond tap-to-confirm.

---

## 12. Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Llama HF access denied at hackathon start | Low | High | Pre-download and cache Day 0. Qwen2.5-3B-Instruct as backup. |
| SAM 2 too slow on demo hardware | Medium | Medium | Fall back to Grounding DINO bounding boxes only; no masks. |
| rPPG unreliable in demo venue lighting | Medium | Low | Feature-flag it; if confidence < 0.5, show "HR: pending" instead of fake numbers. |
| Moulage doesn't fool SAM | Medium | High | Pre-test with your specific moulage kit. Tune Grounding DINO prompts to match. Pre-recorded scenario video as ultimate fallback. |
| ATAK tablet unreachable on venue WiFi | Medium | Medium | Bring your own travel router; run everything on an isolated subnet. |
| Phone streaming (Task 3.4) fails | High | Low | It's a stretch goal. Use laptop webcam. |
| One teammate's laptop can't run SAM | Medium | Low | Everyone runs the pipeline at least once on Day 0. Identify the "demo laptop" early. |
| Demo day wifi is saturated | High | Low | Demo doesn't require internet once models are downloaded. Turn off cellular hotspots in range. |
| You run out of time | High | High | Cut Phase 4's nice-to-haves. MVP is Phases 1+2+3. Everything after is optional. |

---

## 13. What to cut if you're running out of time (in cut order)

1. Phase 4 Task 4.4 (audit trail) — mention in pitch, don't implement.
2. Phase 4 Task 4.3 (voice handoff) — nice polish, not essential.
3. Phase 3 Task 3.4 (phone as capture) — laptop webcam is fine.
4. Phase 4 Task 4.2 (intervention timers) — clinical polish, not core.
5. Phase 2 Task 2.4 (Llama MIST) — fall back to template-based MIST card from structured fields.
6. Phase 4 Task 4.1 partial — demo only 2 scenario modes instead of 5.

Do NOT cut:
- Person + wound detection (Phase 1 — it's the whole point)
- SALT tagging (Phase 2 — clinical alignment win)
- ATAK integration (Phase 3 — deployability win)
- Benchmarks (Phase 5 — edge inference win)

---

## 14. Appendix A — Prompt templates

### Grounding DINO prompt per scenario mode

Use `.` as separator, lowercase, concrete nouns.

**combat_blast:**
```
person . blood . laceration . shrapnel . tourniquet . amputation . burn . exposed bone
```

**civilian_mci:**
```
person . blood . laceration . crush injury . impalement . burn . bruising . swelling
```

**fire_structure:**
```
person . burn . soot on face . blistered skin . singed clothing . smoke inhalation sign
```

**mva:**
```
person . blood . glass laceration . crush injury . seatbelt pattern . deformity . deformed limb
```

**cbrn:**
```
person . chemical burn . blistering . skin discoloration . respiratory distress . eye irritation
```

### Llama 3.2 MIST card system prompt

```
You are a medical scribe for combat medics in a mass casualty incident. Your only job is to emit structured MIST cards in JSON format based on the observations provided. You never invent facts. You never diagnose. If a field is uncertain, use "unknown".

MIST = Mechanism, Injuries, Signs, Treatment.

Output ONLY valid JSON matching this schema:
{
  "mechanism": "string (under 15 words)",
  "injuries": ["string", ...],
  "signs": {"hr": int|null, "rr": int|null, "spo2": int|null, "consciousness": "alert|voice|pain|unresponsive|unknown"},
  "treatment": ["string", ...],
  "notes": "string (under 25 words)"
}

Do not include any explanation, preamble, or markdown. Only the JSON object.
```

### Llama user message template

```
Victim ID: {victim.id}
Scenario: {scenario.name}
Mechanism hint (scenario-wide): {scenario.default_mechanism}

Detected wound regions (from computer vision):
{for wound in victim.wounds: "- {wound.label} at {wound.body_location} (confidence {wound.confidence:.2f})"}

Vitals (remote photoplethysmography):
- HR: {vitals.hr or "unknown"}
- RR: {vitals.rr or "unknown"}
- SpO2: {vitals.spo2 or "unknown"}

Medic voice notes (last 20 seconds):
"{victim.transcript_excerpt}"

Derived MARCH state:
- M: {march.massive_hemorrhage}
- A: {march.airway}
- R: {march.respiration}
- C: {march.circulation}
- H: {march.head_hypothermia}

Emit the MIST card now.
```

---

## 15. Appendix B — CoT message schema for ATAK

Standard CASEVAC marker. Fields to populate from our state:

```xml
<event version="2.0"
       uid="mascal.victim.{id}"
       type="a-f-G-E-V-C"
       how="m-g"
       time="{iso_now}"
       start="{iso_now}"
       stale="{iso_now_plus_10min}">
  <point lat="{lat}" lon="{lon}" hae="0" ce="9999999.0" le="9999999.0"/>
  <detail>
    <contact callsign="Casualty-{id}"/>
    <remarks>{mist_card_as_text}</remarks>
    <color argb="-65536"/>  <!-- red for Immediate -->
    <usericon iconsetpath="COT_MAPPING_2525B/a-f/a-f-G-E-V-C"/>
    <__medevac priority="{immediate|delayed|minimal|expectant|dead}"
              patient_status="{mist_json_compact}"/>
  </detail>
</event>
```

Color mapping: Red = `-65536`, Yellow = `-256`, Green = `-16711936`, Grey = `-8355712`, Black = `-16777216` (ARGB int).

---

## 16. Appendix C — Judge Q&A prep

Rehearse answers to these:

**Q: What if the AI is wrong?**
A: Every suggestion is confirmed by the lead medic before it leaves their device. Downstream medics still do their own MARCH assessment on arrival. The AI's job is perception and communication, not decision-making. We never auto-emit Expectant or Dead tags — those require explicit human confirmation by design.

**Q: Why not just use a cloud LLM?**
A: The problem statement explicitly calls out DIL environments. Our entire pipeline runs offline. We'll demo this in airplane mode if you'd like.

**Q: How does this actually run on glasses / phone?**
A: Today our benchmarks show the full pipeline at 8–12 FPS on a laptop-class device. With the quantized Llama 3.2 and distilled SAM variants (EdgeSAM, MobileSAM), Meta's published benchmarks indicate ~30 FPS on flagship smartphone NPUs. Our architecture is ExecuTorch-compatible. Ray-Ban Meta glasses become the capture + display layer; a phone or rugged tablet in the medic's pack does inference.

**Q: How do you handle privacy / accountability?**
A: All data stays on device unless explicitly exported. Every state change is logged to an audit trail with actor attribution (AI-suggested vs human-confirmed). Post-incident review is built in.

**Q: Doesn't PRONTO already do this?**
A: PRONTO and the DARPA Triage Challenge systems use drones and ground robots for *standoff* triage. Our niche is the *medic-worn* case — first-person capture from the medic actually in the scene, with voice-note integration and direct ATAK handoff. Complementary, not competing.

**Q: Is this FDA-cleared / validated?**
A: No — it's a hackathon prototype, and we're explicit that it's decision *support* not decision-*making*. Path to clearance would be SaMD pre-submission, likely Class II, with clinical validation trials against simulated MASCAL exercises. That's 12–24 months of work.

---

## 17. Appendix D — Reading list (skim before Day 1)

**Clinical (must read):**
- TCCC Guidelines, Jan 2024 — MARCH order, care under fire, tactical field care.
- SALT Triage algorithm (CHEMM) — the five categories and decision tree.

**Technical (skim relevant sections):**
- SAM 3 paper (Meta, Nov 2025) — concept segmentation mechanics.
- DINOv3 paper (Meta, Aug 2025) — frozen backbone, distillation, adapter patterns.
- Llama 3.2 quantization blog post (Meta, Oct 2024) — SpinQuant, QAT+LoRA.
- EdgeSAM / MobileSAM papers — edge deployment numbers you can cite.
- GRGB rPPG paper (MDPI, 2023) — implement from this.

**Precedents (know them in case judges ask):**
- DARPA Triage Challenge results (2024, 2025, 2026 finals).
- UPenn PRONTO team architecture.
- Team DART (Systems winner) approach.

---

## 18. Closing notes

This plan is aggressive but achievable. Three things matter more than anything else:

1. **Get to an end-to-end demo by end of Phase 1.** Even if it's ugly, it proves the concept and gives you something to iterate. Every hour you spend polishing a single stage without a full loop is an hour wasted.
2. **Clinical correctness over novelty.** If a TCCC-certified medic watched your demo, would they nod or cringe? Optimize for nodding. Judges will include people who know the protocols.
3. **The AI is a relay, not a decider.** Say it in the pitch. Print it on a slide. Code it into the tag confirmation flow. This framing is your ethical and technical moat.

Good luck. Ship it.
