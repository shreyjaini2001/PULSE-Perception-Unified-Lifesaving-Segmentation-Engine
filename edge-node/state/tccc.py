"""TCCC / MARCH codeword scanner.

Scans a transcript window for tactical-medicine keywords so the dashboard
can surface them as chips. The dictionary is intentionally small and
hand-curated — we only want high-signal matches, not every medical word.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

# Each entry: (display, MARCH letter, list of regex patterns)
_CODEWORDS: List[Tuple[str, str, List[str]]] = [
    ("TQ applied", "M", [r"\btq\b", r"tourniquet applied", r"tq high.?and.?tight", r"applying\s+tq"]),
    ("Hemorrhage controlled", "M", [r"hemorrhage controlled", r"bleed(?:ing)? controlled", r"pressure dressing"]),
    ("Junctional TQ", "M", [r"junctional tq", r"combat gauze", r"quikclot"]),
    ("Airway: NPA", "A", [r"\bnpa\b", r"nasopharyngeal"]),
    ("Airway: OPA", "A", [r"\bopa\b", r"oropharyngeal"]),
    ("Cricothyrotomy", "A", [r"\bcric\b", r"cricothyrotomy", r"surgical airway"]),
    ("Needle-D", "R", [r"needle\-?d", r"needle decompression", r"chest decompression"]),
    ("Chest seal", "R", [r"chest seal", r"occlusive dressing"]),
    ("Tension pneumothorax", "R", [r"tension pneumothorax", r"tension pneumo"]),
    ("Hypovolemic shock", "C", [r"hypovolemic", r"shock symptoms", r"cap refill"]),
    ("TXA administered", "C", [r"\btxa\b", r"tranexamic"]),
    ("IV access", "C", [r"\biv\b in", r"saline lock", r"ioio", r"intraosseous"]),
    ("Hypothermia risk", "H", [r"hypothermia", r"blizzard wrap", r"cold victim"]),
    ("Head injury", "H", [r"\btbi\b", r"head wound", r"gcs\s*\d"]),
]


def scan_transcript(text: str) -> List[Dict[str, str]]:
    """Return codeword hits as a list of dicts ``{codeword, march_letter}``.

    Empty/blank input returns ``[]`` without raising.
    """
    if not text:
        return []
    low = text.lower()
    hits: List[Dict[str, str]] = []
    seen = set()
    for display, letter, patterns in _CODEWORDS:
        for pat in patterns:
            if re.search(pat, low):
                key = (display, letter)
                if key not in seen:
                    hits.append({"codeword": display, "march_letter": letter})
                    seen.add(key)
                break
    return hits
