# MASCAL Triage Relay — 2-minute demo script

Print this and hand it to the presenter. Rehearse 5× minimum.

## Pre-flight (before the timer starts)
- [ ] Laptop on `combat_blast`, `python edge-node/main.py`
- [ ] ATAK-CIV tablet open, map centered on demo coords, wifi connected to same LAN
- [ ] Two moulage mannequins staged (GSW + TQ, fire burn)
- [ ] Receiver phone + tablet each on the dashboard URL
- [ ] Demo script card in presenter's hand

## Narration (verbatim)

**0:00 — "Mass casualty incident: three victims with blast-pattern injuries.
One gunshot wound, one traumatic amputation, one burn."** (gesture at mannequins)

**0:15 — Walk up to Victim 1 with the phone/laptop.** Voice note (hold mic):
> "GSW right flank. Tourniquet applied fourteen oh two."

Point at screen:
- "Person detection, persistent track ID."
- "Wound mask on the tourniquet — picked up by Grounding DINO plus SAM."
- "Heart rate coming back from remote photoplethysmography on the face."

**0:40 — "Our SALT tagger sees tourniquet + critical hemorrhage and suggests
Immediate."** Tap Red. **"The medic confirms with one tap. The AI never
decides triage autonomously — that line is in the code as a hard gate."**

**0:50 — Cut to the ATAK tablet.** "Red pin just appeared with the MIST card
in the marker detail panel. Less than half a second from tap to CoT."

**1:05 — Cut to the receiver phone.** "Downstream medic sees the same
victim, updated MARCH dots, handoff summary ready." Tap **Play handoff** —
phone speaks the summary aloud.

**1:20 — Flip scenario mode to Fire.** Scan victim 2 (burn moulage). "Same
pipeline, different prompt set — now detecting burns, soot, airway risk
because fire victims are airway-first in TCCC."

**1:40 — Cut to benchmarks slide.** "Full pipeline at 8–12 FPS on a laptop,
under a gig and a half of RAM, entirely offline. We can demo this in
airplane mode if you want."

**1:55 — Ethics slide.** "Human-in-the-loop, audit trail, data stays on
device. Decision support, never decision-making. We think that's the only
way this ships responsibly."

**2:00 — Done.** Wait for questions.

---

## Fallbacks (if things go wrong)

- **Camera won't pick up moulage wound:** say "we're going to fall back to our
  pre-recorded scenario footage so you can see the end-to-end flow" and run
  `--source edge-node/demo/scenario_blast.mp4`. No apology needed.
- **rPPG doesn't lock on:** tile shows "HR pending" — this is the intended
  UX for low-confidence output. Mention it as a feature.
- **ATAK tablet refuses CoT:** skip the tablet beat; the receiver phone shows
  the same tag propagation and judges understand.
- **LLM stalls >8s:** MIST card falls back to the rule-based template
  automatically. Card still appears, source reads "template" instead of "llama".
- **WiFi saturated / venue network hostile:** switch laptop hotspot on, have
  every device rejoin. Everything runs on localhost anyway.

---

## Q&A one-liners (have these ready)

- "What if the AI is wrong?" → Every tag is human-confirmed. Grey/Black require explicit taps. Never auto-emitted.
- "Why not a cloud LLM?" → DIL environments. Everything runs offline. We'll demo in airplane mode.
- "Does this already run on glasses?" → Benchmarks show 8-12 FPS on laptop. Ray-Ban Meta + pocket compute pack is the production target via ExecuTorch. Same architecture, smaller weights.
- "Privacy / accountability?" → Audit trail in JSONL, actor attribution on every change, data never leaves the device unless the medic broadcasts it.
- "Isn't this just PRONTO / DARPA Triage Challenge?" → Those are standoff. We're medic-worn first-person. Complementary.
- "FDA?" → Hackathon prototype. Decision support, not decision-making. SaMD Class II pre-submission path is ~12-24 months.
