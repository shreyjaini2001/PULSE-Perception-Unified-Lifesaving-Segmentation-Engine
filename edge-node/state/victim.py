"""Per-victim data model.

Pure dataclasses so the whole state can be JSON-serialized and pushed over
WebSocket without any adapter layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SaltTag(str, Enum):
    """SALT triage categories (Sort, Assess, Life-saving interventions, Treat/Transport)."""

    RED = "RED"        # Immediate
    YELLOW = "YELLOW"  # Delayed
    GREEN = "GREEN"    # Minimal
    GREY = "GREY"      # Expectant
    BLACK = "BLACK"    # Dead
    UNTAGGED = "UNTAGGED"


# Hex for dashboard; ARGB ints for ATAK in broadcast layer.
SALT_COLORS: Dict[SaltTag, str] = {
    SaltTag.RED: "#e53935",
    SaltTag.YELLOW: "#fbc02d",
    SaltTag.GREEN: "#43a047",
    SaltTag.GREY: "#9e9e9e",
    SaltTag.BLACK: "#212121",
    SaltTag.UNTAGGED: "#455a64",
}


@dataclass
class Vitals:
    """Vital signs with per-channel confidence."""

    hr: Optional[float] = None         # beats per minute
    rr: Optional[float] = None         # breaths per minute
    spo2: Optional[float] = None       # percent saturation (best-effort)
    hr_confidence: float = 0.0
    rr_confidence: float = 0.0
    spo2_confidence: float = 0.0
    last_updated: float = 0.0


@dataclass
class WoundRegion:
    """A single detected wound or injury signal."""

    label: str                              # "laceration", "burn", "tourniquet", ...
    confidence: float
    bbox: Tuple[int, int, int, int]         # x1, y1, x2, y2 in frame coordinates
    mask_area_px: int = 0
    body_location: str = "unknown"           # underscore region id from body-pose module
    severity: str = "unknown"               # critical | serious | moderate | minor | possible | unknown
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    # Medic confirmation: "pending" (awaiting review), "confirmed" (✓),
    # "rejected" (✗ — excluded from triage/priority).
    confirmation: str = "pending"
    # Evidence fingerprint (blood_frac / dark_frac / skin_frac / edge_density
    # + consensus / anomaly scores when available).  Kept in the wound so
    # downstream UI / audit can show *why* a detection landed where it did.
    evidence: Dict[str, float] = field(default_factory=dict)


@dataclass
class BloodRegion:
    """A blood pooling detection (color-heuristic or SAM-derived)."""

    area_px: int
    bbox: Tuple[int, int, int, int]
    fractional_coverage: float = 0.0       # fraction of victim bbox area
    last_seen: float = field(default_factory=time.time)
    estimated_volume_ml: Optional[float] = None
    volume_bucket: str = ""
    is_floor_pool: bool = False             # True if detected outside victim bbox (ground)


@dataclass
class InterventionTimer:
    """A clinical timer, e.g. a tourniquet applied at a specific moment.

    ``auto`` is set when the timer was started by the pipeline (from a
    wound detection or TCCC codeword) rather than the medic tapping
    Start.  ``source`` records *why* it auto-started so the audit trail
    is interpretable.  The fields default to ``False`` / empty so hand-
    crafted timers (e.g. unit-tests, legacy scan records) still validate.
    """

    kind: str
    started_at: float
    duration_seconds: float = 7200.0
    note: str = ""
    auto: bool = False
    source: str = ""
    # Which milestones have already fired a notification? Prevents the
    # dashboard or audit log from re-announcing 60M every frame after the
    # clock crosses the line.
    alerted: List[str] = field(default_factory=list)


@dataclass
class ScanRecord:
    """A frozen snapshot captured by the lead medic for a single victim.

    Scans are the primary unit of information shared across medics: they
    bundle the bounding box, wounds (with arrow anchors), vitals, MARCH/SALT
    assessment, transcript excerpt, and references to cached JPEGs the
    dashboard fetches via ``/api/scans/<scan_id>/frame.jpg`` and
    ``/crop.jpg``.
    """

    scan_id: str
    victim_id: str
    timestamp: float
    bbox: Tuple[int, int, int, int]
    frame_shape: Tuple[int, int]           # (height, width) of the captured frame
    wounds: List[Dict[str, Any]] = field(default_factory=list)
    blood: List[Dict[str, Any]] = field(default_factory=list)
    vitals: Dict[str, Any] = field(default_factory=dict)
    march: Dict[str, Any] = field(default_factory=dict)
    salt_tag: str = "UNTAGGED"
    salt_reason: str = ""
    priority: str = "P5"
    tbsa_burn_percent: Optional[float] = None
    transcript_snippet: str = ""
    notes: str = ""
    face_crop_url: Optional[str] = None
    frame_url: Optional[str] = None
    crop_url: Optional[str] = None
    # Short keyword phrases the UI can render as chips ("critical burn — left_arm",
    # "TQ applied", "bleeding pool — large"). Populated by the ScanEngine.
    keywords: List[str] = field(default_factory=list)
    # How many frames the head-to-toe sweep aggregated (1 = single shot).
    sweep_frames: int = 1
    sweep_duration_sec: float = 0.0
    # Which sequential scan this is for the owning victim (1 = first scan).
    scan_index: int = 1
    # Optional detection-pipeline telemetry (consensus, anomaly scores, ...).
    detector_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Victim:
    """Aggregated per-victim record. Everything pushed to the dashboard lives here."""

    id: str
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    keypoints: List[Tuple[float, float, float]] = field(default_factory=list)
    wound_regions: List[WoundRegion] = field(default_factory=list)
    blood_regions: List[BloodRegion] = field(default_factory=list)
    vitals: Vitals = field(default_factory=Vitals)
    transcript: str = ""
    transcript_updated: float = 0.0

    # Clinical state
    march: Dict[str, Any] = field(default_factory=dict)
    salt_tag: SaltTag = SaltTag.UNTAGGED
    salt_tag_confirmed: bool = False
    salt_tag_reason: str = ""
    mist: Optional[Dict[str, Any]] = None

    timers: List[InterventionTimer] = field(default_factory=list)

    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Archive support — once a victim has any scans they are kept across
    # prune cycles. ``off_screen`` flips True whenever the victim is not
    # visible in the current frame; the UI renders an "AWAY" pill. When the
    # face matcher sees them again we clear the flag and reattach the live
    # bbox, preserving the full scan history.
    off_screen: bool = False
    last_on_screen: float = field(default_factory=time.time)
    total_scan_count: int = 0  # Total scans ever completed (includes re-scans).

    geo_lat: Optional[float] = None
    geo_lon: Optional[float] = None

    tbsa_burn_percent: Optional[float] = None

    # Face re-ID support (InsightFace ArcFace 512-D embedding, L2-normalised).
    face_embedding: Optional[List[float]] = None
    face_thumb_url: Optional[str] = None

    # Derived P1-P5 priority (see state.priority.derive_priority).
    priority: str = "P5"

    # TCCC codewords detected in the transcript (plain strings; UI renders chips).
    tccc_codewords: List[str] = field(default_factory=list)

    # Medic-curated session negatives for this victim: tuples of (label,
    # body_region) that the medic rejected — future scans skip them.
    rejected_findings: List[Tuple[str, str]] = field(default_factory=list)

    # Frozen scans (primary interaction unit in scan mode).
    scans: List[ScanRecord] = field(default_factory=list)
    last_scan_id: Optional[str] = None

    def touch(self) -> None:
        self.last_seen = time.time()
        self.last_on_screen = self.last_seen
        self.off_screen = False

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot suitable for broadcast."""
        d = asdict(self)
        d["salt_tag"] = self.salt_tag.value
        d["salt_color"] = SALT_COLORS[self.salt_tag]
        now = time.time()
        for i, t in enumerate(self.timers):
            started = t.started_at
            d["timers"][i]["elapsed_seconds"] = now - started
        return d
