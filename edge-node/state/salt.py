"""SALT triage suggestion.

SALT = Sort, Assess, Life-saving interventions, Treat / Transport. We implement
the clinical flowchart verbatim and return a suggested tag plus a human-readable
reason. The *suggestion* is consumed by the dashboard; the tag does not take
effect until the lead medic confirms it.

Hard gate: GREY (Expectant) and BLACK (Dead) are never auto-emitted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .march import MarchState, Status
from .victim import SaltTag, Victim


@dataclass
class SaltSuggestion:
    tag: SaltTag
    confidence: float
    reason: str
    requires_explicit_confirmation: bool = False


def _is_obeying_commands(victim: Victim) -> Optional[bool]:
    """Heuristic: pose keypoints visible + motion detected implies obeying.

    Returns None if we genuinely cannot tell yet.
    """
    if not victim.keypoints:
        return None
    visible = [kp for kp in victim.keypoints if len(kp) >= 3 and kp[2] > 0.5]
    if len(visible) < 5:
        return None
    # MediaPipe/YOLO-pose expose 17+ keypoints. If the upper body is visible
    # and the victim hasn't been static for the whole capture, assume yes.
    return True


def _has_peripheral_pulse(victim: Victim) -> Optional[bool]:
    """Proxy: rPPG lock with HR in a plausible band."""
    v = victim.vitals
    if v is None or v.hr is None or v.hr_confidence < 0.4:
        return None
    return 40 <= v.hr <= 180


def suggest_salt(victim: Victim, march: MarchState) -> SaltSuggestion:
    """Return a SALT tag suggestion based on victim + MARCH state."""

    # Not breathing after airway maneuver → BLACK, but we never auto-emit BLACK.
    if march.respiration.status == Status.CRITICAL and march.airway.status in {
        Status.SUSPECTED,
        Status.CONFIRMED,
        Status.CRITICAL,
    }:
        return SaltSuggestion(
            tag=SaltTag.BLACK,
            confidence=0.5,
            reason="respiration absent with airway compromise — confirm Dead manually",
            requires_explicit_confirmation=True,
        )

    # Signals that push toward RED (Immediate)
    critical_m = march.massive_hemorrhage.status == Status.CRITICAL
    treated_m = march.massive_hemorrhage.status == Status.TREATED
    critical_c = march.circulation.status == Status.CRITICAL
    critical_r = march.respiration.status == Status.CRITICAL
    airway_compromised = march.airway.status in {Status.SUSPECTED, Status.CONFIRMED, Status.CRITICAL}

    obeying = _is_obeying_commands(victim)
    has_pulse = _has_peripheral_pulse(victim)

    # GREEN: walking / obeying / stable vitals / no hemorrhage.
    if (
        obeying is True
        and has_pulse is True
        and not critical_m
        and not treated_m
        and not critical_r
        and not airway_compromised
        and not critical_c
    ):
        return SaltSuggestion(
            tag=SaltTag.GREEN,
            confidence=0.7,
            reason="ambulatory, pulse present, no critical MARCH signals",
        )

    # RED vs GREY: both have the "not obeying OR no pulse OR severe distress" pattern.
    # In a hackathon we cannot meaningfully decide "likely to survive given resources",
    # so GREY always requires manual confirmation.
    if (
        critical_m
        or treated_m
        or critical_c
        or critical_r
        or airway_compromised
        or obeying is False
        or has_pulse is False
    ):
        reasons = []
        if critical_m:
            reasons.append("massive hemorrhage")
        if treated_m:
            reasons.append("massive hemorrhage treated")
        if critical_c:
            reasons.append("circulatory critical")
        if critical_r:
            reasons.append("respiratory critical")
        if airway_compromised:
            reasons.append("airway compromised")
        if obeying is False:
            reasons.append("not obeying commands")
        if has_pulse is False:
            reasons.append("peripheral pulse absent")
        return SaltSuggestion(
            tag=SaltTag.RED,
            confidence=min(0.95, 0.6 + 0.1 * len(reasons)),
            reason="; ".join(reasons),
        )

    # Otherwise YELLOW (Delayed)
    return SaltSuggestion(
        tag=SaltTag.YELLOW,
        confidence=0.5,
        reason="non-ambulatory or uncertain status without life-threats yet",
    )
