"""Smoke test: import every module and exercise the pure-Python state layer.

Does NOT open a camera or run the full pipeline — that's for `main.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "edge-node"))


def main() -> None:
    print("-- imports --")
    from state import (derive_march, suggest_salt, SaltTag, Scene, Victim,
                       Vitals, WoundRegion, BloodRegion)
    from state.victim import InterventionTimer
    from pipeline import (PersonDetector, WoundSegmenter, BodyLocator,
                          RppgEstimator, AudioTranscriber, MistGenerator)
    from broadcast import BroadcastServer, AtakBridge
    from broadcast.atak_bridge import AtakConfig
    from audit import AuditLog
    print("OK")

    print("-- state roundtrip --")
    scene = Scene(scenario="combat_blast")
    v = Victim(id="Alpha-1", bbox=(10, 10, 300, 500))
    v.wound_regions.append(WoundRegion(label="tourniquet", confidence=0.82,
                                       bbox=(100, 200, 180, 260),
                                       body_location="right_thigh", severity="serious"))
    v.blood_regions.append(BloodRegion(area_px=4200, bbox=(110, 210, 190, 270),
                                        fractional_coverage=0.12))
    v.vitals = Vitals(hr=138, rr=22, hr_confidence=0.55, rr_confidence=0.45)
    v.transcript = "GSW right thigh, TQ applied 14:02, heart rate rising"
    scene.upsert_victim(v)

    march = derive_march(v)
    print("MARCH:", {k: march.to_dict()[k]["status"] for k in "MARCH"})

    suggestion = suggest_salt(v, march)
    print(f"SALT suggestion: {suggestion.tag.value}  ({suggestion.reason})")
    assert suggestion.tag == SaltTag.RED

    v.salt_tag = suggestion.tag
    v.salt_tag_confirmed = True
    v.timers.append(InterventionTimer(kind="tourniquet", started_at=0.0, duration_seconds=7200))
    snap = scene.snapshot()
    assert snap["victims"][0]["salt_tag"] == suggestion.tag.value

    print("-- MIST generator (rule-based path) --")
    mist = MistGenerator(enabled=False).generate(v, march, {"name": "Combat",
                                                             "default_mechanism": "blast injury"})
    print(mist.to_dict())

    print("-- body locator --")
    print("location:", BodyLocator.locate((150, 230, 170, 260), v.bbox, v.keypoints))

    print("-- ATAK CoT (dry run, disabled) --")
    bridge = AtakBridge(AtakConfig(enabled=False))
    bridge.publish(v, force=True)
    print("publish returned (no-op because disabled=True)")

    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
