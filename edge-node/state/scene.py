"""Scene-level aggregation: the dict of victims plus global metadata."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .victim import Victim


@dataclass
class Scene:
    scenario: str = "combat_blast"
    started_at: float = field(default_factory=time.time)
    victims: Dict[str, Victim] = field(default_factory=dict)
    frame_count: int = 0
    last_frame_ts: float = 0.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # Aggregate (scenario-wide) voice transcript buffer; per-victim transcript
    # is copied from this as the medic focuses on a given victim.
    global_transcript: str = ""

    # Median optical-flow per frame (for minimap / ego-motion hint)
    camera_flow: Dict[str, Any] = field(default_factory=dict)

    def upsert_victim(self, victim: Victim) -> None:
        with self._lock:
            self.victims[victim.id] = victim

    def get(self, vid: str) -> Optional[Victim]:
        with self._lock:
            return self.victims.get(vid)

    def all_victims(self) -> List[Victim]:
        with self._lock:
            return list(self.victims.values())

    def prune_stale(self, max_age_seconds: float = 30.0) -> None:
        """Drop victims that have been off-camera AND have no frozen scans.

        Scanned victims are archived forever: even after they leave the
        scene we keep their record (face embedding + scans) so that when
        they walk back in, :class:`pipeline.face_reid.FaceReID` can match
        them to the SAME ``Victim`` instance and the medic's prior scans
        are preserved instead of orphaned into a new callsign.
        """
        now = time.time()
        with self._lock:
            stale: List[str] = []
            for vid, v in self.victims.items():
                off_age = now - v.last_seen
                if off_age <= max_age_seconds:
                    continue
                if v.scans:
                    # Archive: mark off-screen and hold onto the record.
                    v.off_screen = True
                    continue
                stale.append(vid)
            for vid in stale:
                self.victims.pop(vid, None)

    def mark_off_screen(self, visible_ids: set, now: Optional[float] = None) -> None:
        """Flip ``off_screen`` to True for anyone not in ``visible_ids``.

        Called from the main capture loop after each frame to keep the
        dashboard aware of who is currently on-camera vs archived.
        """
        t = now or time.time()
        with self._lock:
            for vid, v in self.victims.items():
                if vid in visible_ids:
                    v.off_screen = False
                    v.last_on_screen = t
                else:
                    v.off_screen = True

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "scenario": self.scenario,
                "started_at": self.started_at,
                "frame_count": self.frame_count,
                "last_frame_ts": self.last_frame_ts,
                "camera_flow": dict(self.camera_flow),
                "victims": [v.to_dict() for v in self.victims.values()],
            }
