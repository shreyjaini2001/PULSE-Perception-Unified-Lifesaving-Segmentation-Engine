# MASCAL Triage Relay — pitch deck outline

Ten slides. One concept per slide. No walls of text.

---

## 1. Title
**MASCAL Triage Relay** — _A perception relay for the lead medic._

Team: <names>. Problem statements: Meta #15 + Meta #16.
One-liner: "Sees what the medic sees, triages who needs help first, relays it to everyone who needs to know — offline."

## 2. The problem
- Modern MASCAL incidents overwhelm lead-medic working memory in seconds.
- DoD data (cited in the problem statement): **24% of combat deaths are potentially survivable** with faster, better-organized triage.
- Existing tools (paper MIST, radio handoffs) don't scale past ~3 victims before fidelity collapses.
- Cloud LLMs don't work in DIL.

## 3. Precedents
- **DARPA Triage Challenge** (2024-2026) — standoff robotic triage with drones + UGVs.
- **UPenn PRONTO** — drone-based first-pass triage.
- **Team DART** (Systems winner) — autonomous multi-agent triage.
- **Our niche:** medic-worn, first-person, voice-aware, offline, ATAK-native. Complementary, not competing.

## 4. Architecture
_(diagram: capture → shared DINOv3 backbone → several heads → state aggregator → broadcast (WS + CoT))_

Two-tier edge: capture device (phone / glasses) + compute node (laptop / rugged tablet in pack). Demoed collapsed to one laptop.

## 5. AI pipeline
- **Perception:** DINOv3 features + YOLOv8-pose (detection) + Grounding DINO + SAM 2.1 (open-vocab wound masks).
- **Vitals:** MediaPipe FaceMesh + GRGB remote-PPG → HR / RR.
- **Voice:** faster-whisper tiny.en → rolling transcript.
- **Reasoning:** Llama 3.2 3B Q4_K_M (llama.cpp) → MIST card JSON.
- **All local. All quantized. All ExecuTorch-compatible.**

## 6. Clinical alignment
- **MARCH** (TCCC) primary assessment — derived per-victim as structured state.
- **SALT** triage algorithm — suggestion engine, not decision engine.
- **MIST** card — LLM-synthesised handoff packet.
- Hard gate: GREY (Expectant) and BLACK (Dead) _never_ auto-emitted.

## 7. Scenario modes
Live-demo scenario switch. Same pipeline, different prompts + triage scheme:
| Mode | Prompt focus | Handoff format |
|---|---|---|
| Combat / Blast | shrapnel, amputation, TQ | 9-line MEDEVAC |
| Civilian MCI | lacs, crush, impalement | EMS PCR |
| Fire / Structure | burns, soot, airway-first | EMS PCR |
| MVA | glass lac, crush, deformity | START + EMS PCR |
| CBRN | chemical burn, blistering | EMS PCR + decon flag |

## 8. Deployability
- **ATAK-CIV integration** via CoT markers. CASEVAC (`a-f-G-E-V-C`) type with MIST card in remarks, SALT color, MEDEVAC priority.
- **Local mesh** via WebSocket today; goTenna / Rajant production path.
- **Benchmarks:** 8–12 FPS, <1.5 GB RAM, MIST in <6 s, CoT publish <500 ms — all on a laptop. Path to glasses via ExecuTorch.

## 9. Ethics & scope
- **Human in the loop.** AI suggests, medic confirms. Every tag.
- **Data stays on device.** Nothing egresses without explicit action.
- **Audit trail.** JSONL with actor attribution on every state change.
- **Decision support, not decision-making.** FDA path is SaMD Class II, 12-24 months.

## 10. Roadmap
- **Today (demo):** DINOv2 + SAM 2.1 + Llama 3.2 3B on laptop.
- **Q+1:** SAM 3.1 concept segmentation, DINOv3 distilled, SpinQuant Llama 3.2 1B.
- **Q+2:** Ray-Ban Meta as capture layer, compact compute pack, goTenna mesh.
- **Q+3:** Field trials with operational unit medics + validated MASCAL exercise sets.
- **Q+6:** FDA SaMD pre-submission.

---

### Backup slides (keep handy)
- Benchmarks table (generated from `benchmarks/results.md`)
- MIST card example (screenshot of dashboard)
- CoT XML sample (Appendix B of the plan)
- Audit JSONL excerpt (one REDLINE, one tag_confirmed, one scenario_changed)
