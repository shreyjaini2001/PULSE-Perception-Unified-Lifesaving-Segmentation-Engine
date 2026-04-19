# Ethics & scope

This is a hackathon prototype. Read the following before using or demoing this system.

## Scope
- **Decision support, not decision-making.** Every SALT tag, every MIST card, every downstream broadcast originates as an AI *suggestion*. Nothing reaches ATAK or a receiver medic until a human has tapped "confirm."
- **No clinical validation.** The thresholds in `state/march.py` and `state/salt.py` are informed by published TCCC / SALT guidelines but have not been clinically validated. They should be treated as a demonstration of the control loop, not a clinical recommendation.
- **Not FDA-cleared.** Path to clearance is Software-as-a-Medical-Device (SaMD), likely Class II, which requires ~12–24 months of validation trials against simulated MASCAL exercises.
- **Hackathon-grade models.** YOLOv8-pose, Grounding DINO, SAM 2.1, Llama 3.2 3B are excellent for demos but are not certified for trauma triage.

## Hard gates in code
- GREY (Expectant) and BLACK (Dead) SALT tags are **never** auto-emitted. Both require explicit human confirmation. Enforced in both `main.py` (UI-side: tag is replaced with UNTAGGED until confirmed) and `salt.py` (suggestion flag `requires_explicit_confirmation`).
- ATAK CoT broadcasts only fire for `salt_tag_confirmed == True` victims.
- Audit log is append-only and records both the AI suggestion and the human confirmation separately.

## Data handling
- **No egress.** The pipeline never makes an outbound internet call during operation. Model downloads happen once, via `scripts/download_models.py`, on a provisioning machine.
- **Local mesh only.** WebSocket broadcasts bind to the local subnet. ATAK CoT is UDP to a single configured tablet.
- **Audit trail.** Every state change logs to `logs/audit_*.jsonl` with `actor` ("medic" for human confirmations, "ai" for automated suggestions), previous state, and new state. Post-incident review is built in.
- **Voice transcripts** are held in a rolling 30-second buffer per victim and persist only in the audit log if a medic explicitly attaches them to a note.

## Known failure modes & how we communicate them
- **rPPG noise / face not visible.** UI shows "HR pending" instead of a fabricated number. `hr` stays `None` with `hr_confidence < min_confidence` gate in `rppg.py`.
- **Moulage vs. real wound.** Grounding DINO prompts are tuned to a specific moulage kit; real injuries will produce different distributions. Mention this explicitly in the pitch.
- **LLM hallucination.** Output is Pydantic-validated, constrained to a tight JSON schema, and falls back to a deterministic template on parse failure. The MIST card's `source` field always discloses `"llama"` vs. `"template"` so reviewers know which path generated it.
- **Single-tap misclicks.** Tag confirmation is final in the MVP; production would add a 2-second undo window and require re-confirmation for irreversible tags (BLACK).

## If someone asks "can I use this on a real patient today?"
**No.** Full stop. This is a demonstration of an architecture and a control loop. Use in the field requires clinical validation, human-factors testing, hardware certification, and regulatory clearance that this project has not attempted.
