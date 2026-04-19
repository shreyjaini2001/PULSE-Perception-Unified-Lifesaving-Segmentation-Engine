"""Append-only JSONL audit trail.

Every state change (AI-suggested vs. human-confirmed) is logged with
actor attribution. Satisfies the "post-incident review" pitch and provides
a real answer to the "how do you handle accountability" judge question.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Optional


class AuditLog:
    def __init__(self, log_dir: str = "logs") -> None:
        os.makedirs(log_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(log_dir, f"audit_{stamp}.jsonl")
        self._lock = threading.Lock()
        self._fh = open(self.path, "a", encoding="utf-8", buffering=1)

    def write(self,
              event_type: str,
              actor: str,
              victim_id: Optional[str] = None,
              previous_state: Any = None,
              new_state: Any = None,
              payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        evt = {
            "ts": time.time(),
            "event_type": event_type,
            "actor": actor,
            "victim_id": victim_id,
            "previous_state": previous_state,
            "new_state": new_state,
        }
        if payload:
            evt["payload"] = payload
        line = json.dumps(evt, default=str)
        with self._lock:
            self._fh.write(line + "\n")
        return evt

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
