"""MARCH state derivation.

MARCH = Massive hemorrhage, Airway, Respiration, Circulation, Head/Hypothermia.
This is the TCCC primary assessment order. Each field carries a status + a
confidence so the UI can flag low-confidence AI suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any, Dict, Iterable

from .victim import Victim, WoundRegion


class Status(str, Enum):
    UNKNOWN = "unknown"
    NORMAL = "normal"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"
    CRITICAL = "critical"
    TREATED = "treated"


@dataclass
class MarchField:
    status: Status = Status.UNKNOWN
    confidence: float = 0.0
    reason: str = ""


@dataclass
class MarchState:
    massive_hemorrhage: MarchField = None  # type: ignore[assignment]
    airway: MarchField = None              # type: ignore[assignment]
    respiration: MarchField = None         # type: ignore[assignment]
    circulation: MarchField = None         # type: ignore[assignment]
    head_hypothermia: MarchField = None    # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.massive_hemorrhage is None:
            self.massive_hemorrhage = MarchField()
        if self.airway is None:
            self.airway = MarchField()
        if self.respiration is None:
            self.respiration = MarchField()
        if self.circulation is None:
            self.circulation = MarchField()
        if self.head_hypothermia is None:
            self.head_hypothermia = MarchField()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "M": asdict(self.massive_hemorrhage),
            "A": asdict(self.airway),
            "R": asdict(self.respiration),
            "C": asdict(self.circulation),
            "H": asdict(self.head_hypothermia),
        }


# --- Heuristics -------------------------------------------------------------

HEMORRHAGE_LABELS = {"blood", "laceration", "amputation", "exposed bone", "gsw",
                     "gunshot wound", "penetrating wound", "shrapnel"}
TOURNIQUET_LABELS = {"tourniquet", "tq"}
BURN_LABELS = {"burn", "blistered skin", "soot on face"}
AIRWAY_LABELS = {"soot on face", "burn"}  # airway-burn correlation
TBI_KEYWORDS = {"tbi", "head injury", "unconscious", "unresponsive", "brain"}
HEMORRHAGE_KEYWORDS = {"bleeding", "hemorrhage", "gsw", "tq ", "tourniquet", "blood"}
AIRWAY_KEYWORDS = {"airway", "gurgl", "gasp", "obstruction"}


def _wound_labels(wounds: Iterable[WoundRegion]) -> set[str]:
    return {w.label.lower() for w in wounds}


def _total_blood_fraction(victim: Victim) -> float:
    return sum(b.fractional_coverage for b in victim.blood_regions)


def _floor_blood_severity_score(victim: Victim) -> float:
    """Map largest floor-pool bucket to 0..1 for MARCH M."""
    order = {"": 0.0, "<500ml": 0.25, "500-1500ml": 0.55, ">1500ml": 1.0}
    best = 0.0
    for b in victim.blood_regions:
        if not getattr(b, "is_floor_pool", False):
            continue
        best = max(best, order.get(b.volume_bucket or "", 0.0))
    return best


def derive_march(victim: Victim, scenario: Dict[str, Any] | None = None) -> MarchState:
    """Derive a MARCH snapshot from the aggregated victim state.

    This is conservative by design: it *suggests* states and exposes reasons.
    The lead medic confirms before downstream effects (tag broadcast, ATAK).
    """
    state = MarchState()
    labels = _wound_labels(victim.wound_regions)
    transcript = (victim.transcript or "").lower()
    blood_frac = _total_blood_fraction(victim)
    floor_score = _floor_blood_severity_score(victim)

    # M — massive hemorrhage (tourniquet wins; then floor pools; then coverage/labels)
    if labels & TOURNIQUET_LABELS or "tourniquet" in transcript or " tq " in f" {transcript} ":
        state.massive_hemorrhage = MarchField(
            status=Status.TREATED,
            confidence=0.85,
            reason="tourniquet detected",
        )
    elif floor_score >= 0.55:
        state.massive_hemorrhage = MarchField(
            status=Status.CRITICAL if floor_score >= 1.0 else Status.SUSPECTED,
            confidence=min(0.9, 0.45 + floor_score * 0.45),
            reason="large floor blood pool (volume bucket)",
        )
    elif (
        blood_frac > 0.05
        or labels & HEMORRHAGE_LABELS
        or any(k in transcript for k in HEMORRHAGE_KEYWORDS)
        or floor_score >= 0.25
    ):
        severity = "critical" if blood_frac > 0.12 or floor_score >= 1.0 else "suspected"
        pool_note = f"; floor_pool={floor_score:.0%}" if floor_score > 0 else ""
        state.massive_hemorrhage = MarchField(
            status=Status.CRITICAL if severity == "critical" else Status.SUSPECTED,
            confidence=min(0.95, 0.4 + blood_frac * 4.0 + floor_score * 0.2),
            reason=f"blood coverage {blood_frac:.1%}; labels={sorted(labels & HEMORRHAGE_LABELS) or '[]'}{pool_note}",
        )
    else:
        state.massive_hemorrhage = MarchField(status=Status.NORMAL, confidence=0.3,
                                              reason="no bleeding signal detected")

    # A — airway
    if any(k in transcript for k in AIRWAY_KEYWORDS):
        state.airway = MarchField(status=Status.SUSPECTED, confidence=0.7,
                                  reason="airway concern in medic transcript")
    elif labels & AIRWAY_LABELS and scenario and scenario.get("airway_priority"):
        state.airway = MarchField(status=Status.SUSPECTED, confidence=0.55,
                                  reason="facial burns in fire scenario — airway risk")
    else:
        state.airway = MarchField(status=Status.NORMAL, confidence=0.3,
                                  reason="no airway signal")

    # R — respiration
    rr = victim.vitals.rr if victim.vitals else None
    if rr is None or (victim.vitals and victim.vitals.rr_confidence < 0.3):
        state.respiration = MarchField(status=Status.UNKNOWN, confidence=0.2,
                                       reason="RR not yet estimated")
    elif 8 <= rr <= 30:
        state.respiration = MarchField(status=Status.NORMAL, confidence=victim.vitals.rr_confidence,
                                       reason=f"RR {rr:.0f} within 8-30")
    else:
        state.respiration = MarchField(status=Status.CRITICAL, confidence=victim.vitals.rr_confidence,
                                       reason=f"RR {rr:.0f} outside 8-30")

    # C — circulation
    hr = victim.vitals.hr if victim.vitals else None
    if hr is None or (victim.vitals and victim.vitals.hr_confidence < 0.3):
        state.circulation = MarchField(status=Status.UNKNOWN, confidence=0.2,
                                       reason="HR not yet locked")
    elif hr < 40 or hr > 140:
        state.circulation = MarchField(status=Status.CRITICAL, confidence=victim.vitals.hr_confidence,
                                       reason=f"HR {hr:.0f} outside safe band")
    elif 60 <= hr <= 120:
        state.circulation = MarchField(status=Status.NORMAL, confidence=victim.vitals.hr_confidence,
                                       reason=f"HR {hr:.0f} within 60-120")
    else:
        state.circulation = MarchField(status=Status.SUSPECTED, confidence=victim.vitals.hr_confidence,
                                       reason=f"HR {hr:.0f} borderline")

    # H — head / hypothermia
    if any(k in transcript for k in TBI_KEYWORDS):
        state.head_hypothermia = MarchField(status=Status.SUSPECTED, confidence=0.7,
                                            reason="TBI concern in transcript")
    else:
        state.head_hypothermia = MarchField(status=Status.NORMAL, confidence=0.25,
                                            reason="no head/hypothermia signal")

    return state
