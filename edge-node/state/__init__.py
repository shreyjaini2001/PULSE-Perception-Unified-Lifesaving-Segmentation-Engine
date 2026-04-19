"""State models for victims, MARCH, SALT, and scene aggregation."""

from .victim import (
    Victim,
    Vitals,
    WoundRegion,
    BloodRegion,
    SaltTag,
    SALT_COLORS,
    ScanRecord,
)
from .march import MarchState, Status, derive_march
from .salt import suggest_salt, SaltSuggestion
from .scene import Scene
from .burns import estimate_tbsa_percent
from .priority import (
    derive_priority,
    worst_severity,
    PRIORITY_LABELS,
    PRIORITY_COLORS,
)
from .tccc import scan_transcript

__all__ = [
    "Victim",
    "Vitals",
    "WoundRegion",
    "BloodRegion",
    "SaltTag",
    "SALT_COLORS",
    "ScanRecord",
    "MarchState",
    "Status",
    "derive_march",
    "suggest_salt",
    "SaltSuggestion",
    "Scene",
    "estimate_tbsa_percent",
    "derive_priority",
    "worst_severity",
    "PRIORITY_LABELS",
    "PRIORITY_COLORS",
    "scan_transcript",
]
