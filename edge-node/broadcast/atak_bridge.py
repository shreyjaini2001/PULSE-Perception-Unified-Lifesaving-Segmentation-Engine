"""ATAK CoT publisher.

Primary: if ``atakcots`` is installed, use its ``push_cot`` helper.
Fallback: hand-crafted CoT XML sent over UDP to the ATAK tablet's port 4242.

We debounce: only emit on state change (tag or MIST updated) or at most every
15 seconds per victim, whichever comes first.
"""

from __future__ import annotations

import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from state.victim import SaltTag, Victim


SALT_TO_MEDEVAC_PRIORITY = {
    SaltTag.RED: "immediate",
    SaltTag.YELLOW: "delayed",
    SaltTag.GREEN: "minimal",
    SaltTag.GREY: "expectant",
    SaltTag.BLACK: "dead",
    SaltTag.UNTAGGED: "unknown",
}

SALT_TO_ARGB = {
    SaltTag.RED: -65536,
    SaltTag.YELLOW: -256,
    SaltTag.GREEN: -16711936,
    SaltTag.GREY: -8355712,
    SaltTag.BLACK: -16777216,
    SaltTag.UNTAGGED: -1,
}


@dataclass
class AtakConfig:
    enabled: bool = False
    host: str = "192.168.1.50"
    port: int = 4242
    anchor_lat: float = 38.8895
    anchor_lon: float = -77.0353


class AtakBridge:
    def __init__(self, cfg: AtakConfig) -> None:
        self.cfg = cfg
        self._last_push: Dict[str, float] = {}
        self._last_payload_key: Dict[str, str] = {}
        self._sock: Optional[socket.socket] = None
        self._atakcots = None

        if not cfg.enabled:
            return
        try:
            import atakcots  # type: ignore

            self._atakcots = atakcots
            print("[atak] atakcots library loaded.")
        except Exception as exc:
            print(f"[atak] atakcots unavailable ({exc}); using raw UDP.")

        if self._atakcots is None:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # ------------------------------------------------------------------
    def publish(self, victim: Victim, force: bool = False) -> None:
        if not self.cfg.enabled:
            return
        if not victim.salt_tag_confirmed:
            return

        now = time.time()
        key = json.dumps({
            "tag": victim.salt_tag.value,
            "mist": victim.mist or {},
            "transcript": victim.transcript,
        }, sort_keys=True)
        last = self._last_push.get(victim.id, 0.0)
        if not force and key == self._last_payload_key.get(victim.id) and now - last < 15.0:
            return
        self._last_push[victim.id] = now
        self._last_payload_key[victim.id] = key

        lat = victim.geo_lat if victim.geo_lat is not None else self.cfg.anchor_lat
        lon = victim.geo_lon if victim.geo_lon is not None else self.cfg.anchor_lon

        if self._atakcots is not None:
            try:
                self._publish_with_atakcots(victim, lat, lon)
                return
            except Exception as exc:
                print(f"[atak] atakcots push failed ({exc}); falling back to UDP.")

        self._publish_raw_udp(victim, lat, lon)

    # ------------------------------------------------------------------
    def _publish_with_atakcots(self, victim: Victim, lat: float, lon: float) -> None:
        CotConfig = getattr(self._atakcots, "CotConfig", None)
        push_cot = getattr(self._atakcots, "push_cot", None)
        if CotConfig is None or push_cot is None:
            raise RuntimeError("atakcots API not recognized")

        cfg = CotConfig(
            uid=f"mascal.victim.{victim.id}",
            lat=lat,
            lon=lon,
            type="a-f-G-E-V-C",
            callsign=f"Casualty-{victim.id}",
            remarks=_mist_as_text(victim),
            stale_seconds=600,
        )
        push_cot(cfg, self.cfg.host)

    def _publish_raw_udp(self, victim: Victim, lat: float, lon: float) -> None:
        assert self._sock is not None
        xml = _build_cot_xml(victim, lat, lon)
        try:
            self._sock.sendto(xml.encode("utf-8"), (self.cfg.host, self.cfg.port))
        except Exception as exc:
            print(f"[atak] UDP sendto failed: {exc}")


def _iso_now(offset_seconds: int = 0) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() + offset_seconds)) + "Z"


def _mist_as_text(victim: Victim) -> str:
    lines = [f"Callsign: Casualty-{victim.id}",
             f"SALT: {victim.salt_tag.value}"]
    if victim.vitals and (victim.vitals.hr or victim.vitals.rr):
        lines.append(f"Vitals: HR={victim.vitals.hr or 'n/a'} RR={victim.vitals.rr or 'n/a'}")
    if victim.mist:
        m = victim.mist
        lines.append(f"Mechanism: {m.get('mechanism', 'unknown')}")
        lines.append("Injuries: " + "; ".join(m.get("injuries", [])))
        lines.append("Treatment: " + "; ".join(m.get("treatment", [])))
        if m.get("notes"):
            lines.append(f"Notes: {m['notes']}")
    return "\n".join(lines)


def _build_cot_xml(victim: Victim, lat: float, lon: float) -> str:
    argb = SALT_TO_ARGB.get(victim.salt_tag, -1)
    priority = SALT_TO_MEDEVAC_PRIORITY.get(victim.salt_tag, "unknown")
    mist_compact = json.dumps(victim.mist or {}, separators=(",", ":"))
    remarks = _xml_escape(_mist_as_text(victim))
    return (
        f'<event version="2.0" uid="mascal.victim.{victim.id}" '
        f'type="a-f-G-E-V-C" how="m-g" '
        f'time="{_iso_now()}" start="{_iso_now()}" stale="{_iso_now(600)}">'
        f'<point lat="{lat}" lon="{lon}" hae="0" ce="9999999.0" le="9999999.0"/>'
        f'<detail>'
        f'<contact callsign="Casualty-{victim.id}"/>'
        f'<remarks>{remarks}</remarks>'
        f'<color argb="{argb}"/>'
        f'<usericon iconsetpath="COT_MAPPING_2525B/a-f/a-f-G-E-V-C"/>'
        f'<__medevac priority="{priority}" patient_status=\'{_xml_escape(mist_compact)}\'/>'
        f'</detail>'
        f'</event>'
    )


def _xml_escape(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;")
             .replace("'", "&apos;"))
