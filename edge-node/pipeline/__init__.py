"""Perception pipeline stages.

Each module exposes a class with an ``__init__(config)`` and a ``process(...)``
method so ``main.py`` can swap implementations or disable stages without
touching the call sites.
"""

from .person import PersonDetector, TrackedPerson
from .wound import (
    WoundSegmenter,
    GdinoSam2Backend,
    Sam3Backend,
    estimate_wound_severity,
)
from .body_pose import (
    BodyLocator,
    compute_body_regions,
    keypoints_to_dict,
    locate_wound_on_body,
)
from .rppg import RppgEstimator
from .audio import AudioTranscriber
from .llm import MistGenerator
from .face_reid import FaceReID, FaceMatchResult
from .scan_engine import ScanEngine
from .scan_store import store as scan_store

__all__ = [
    "PersonDetector",
    "TrackedPerson",
    "WoundSegmenter",
    "GdinoSam2Backend",
    "Sam3Backend",
    "estimate_wound_severity",
    "BodyLocator",
    "compute_body_regions",
    "keypoints_to_dict",
    "locate_wound_on_body",
    "RppgEstimator",
    "AudioTranscriber",
    "MistGenerator",
    "FaceReID",
    "FaceMatchResult",
    "ScanEngine",
    "scan_store",
]
