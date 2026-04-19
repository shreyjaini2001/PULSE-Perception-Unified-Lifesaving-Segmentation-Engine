"""P1-P5 priority matrix derivation.

The priority matrix is a hackathon-friendly, medic-agnostic summary that
combines SALT tag + worst wound severity + airway/breathing flags. It maps
onto commonly-used prehospital triage labels:

  P1  Immediate threat to life      (RED + critical severity, or airway/M failure)
  P2  Serious, surgical/ICU soon    (RED + serious, or major hemorrhage under control)
  P3  Delayed                       (YELLOW)
  P4  Minor / ambulatory            (GREEN)
  P5  Minimal / walking wounded or unassessed

Expectant (GREY) and Deceased (BLACK) are intentionally NOT coerced into a P
level — they surface as themselves on the UI so a medic must confirm.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

SEVERITY_RANK = {
    "critical": 4,
    "serious": 3,
    "moderate": 2,
    "minor": 1,
    # "possible" is explicitly a *low-confidence* finding — it must NOT
    # promote the victim to P1/P2. It ranks below "minor" so that a scan
    # full of possibles still defaults to P4/P5 absent corroborating signs.
    "possible": 0,
    "unknown": 0,
}


def worst_severity(wounds: Iterable[Any]) -> str:
    best = "unknown"
    best_rank = -1
    for w in wounds or []:
        sev = getattr(w, "severity", None) if not isinstance(w, dict) else w.get("severity")
        r = SEVERITY_RANK.get((sev or "unknown"), 0)
        if r > best_rank:
            best_rank = r
            best = sev or "unknown"
    return best


def _march_flag(march: Dict[str, Any], key: str) -> str:
    if not march:
        return "unknown"
    field = march.get(key) or {}
    status = field.get("status") if isinstance(field, dict) else None
    return str(status or "unknown")


def derive_priority(salt_tag: str,
                     wounds: Iterable[Any],
                     march: Optional[Dict[str, Any]] = None,
                     ) -> str:
    """Return one of P1..P5 (never GREY/BLACK — those surface separately)."""
    tag = str(salt_tag or "UNTAGGED").upper()
    sev = worst_severity(wounds)
    m_status = _march_flag(march or {}, "M")
    a_status = _march_flag(march or {}, "A")

    if tag in ("GREY", "BLACK"):
        # Let the UI render the tag directly; we still expose a safe P level
        # so no downstream code breaks on missing fields.
        return "P5"

    if tag == "RED":
        if sev == "critical" or a_status == "critical" or m_status == "critical":
            return "P1"
        return "P2"
    if tag == "YELLOW":
        # Yellow with active massive hemorrhage or airway concern escalates.
        if a_status in ("critical", "suspected") or m_status in ("critical", "suspected"):
            return "P2"
        return "P3"
    if tag == "GREEN":
        return "P4"
    return "P5"


PRIORITY_LABELS: Dict[str, str] = {
    "P1": "Immediate",
    "P2": "Serious",
    "P3": "Delayed",
    "P4": "Minor",
    "P5": "Minimal",
}

PRIORITY_COLORS: Dict[str, str] = {
    "P1": "#e53935",
    "P2": "#ff6d00",
    "P3": "#fbc02d",
    "P4": "#43a047",
    "P5": "#455a64",
}
