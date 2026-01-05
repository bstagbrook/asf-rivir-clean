#!/usr/bin/env python3
"""
One-time consent gate with affirmative signal detection.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set


DEFAULT_AFFIRMATIVES: Set[str] = {
    "yes",
    "y",
    "ok",
    "okay",
    "sure",
    "go",
    "do it",
    "allow",
    "approved",
    "affirm",
}

NEGATIVE_SIGNALS: Set[str] = {
    "no",
    "n",
    "nope",
    "stop",
    "deny",
    "denied",
    "reject",
    "cancel",
}


@dataclass
class ConsentGate:
    log_path: Optional[Path] = None
    affirmatives: Set[str] = field(default_factory=lambda: set(DEFAULT_AFFIRMATIVES))
    _tokens: Dict[str, int] = field(default_factory=dict)

    def _log(self, event: dict):
        if not self.log_path:
            return
        try:
            event = dict(event)
            event["ts"] = time.time()
            if self.log_path.exists():
                data = json.loads(self.log_path.read_text())
            else:
                data = []
            data.append(event)
            self.log_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def request(self, action: str, signal: str) -> bool:
        """Grant a one-time token if signal is affirmative."""
        normalized = signal.strip().lower()
        if "*" in self.affirmatives:
            allowed = normalized != "" and normalized not in NEGATIVE_SIGNALS
        else:
            allowed = normalized in self.affirmatives
        self._log({"event": "request", "action": action, "signal": normalized, "allowed": allowed})
        if allowed:
            self._tokens[action] = self._tokens.get(action, 0) + 1
        return allowed

    def consume(self, action: str) -> bool:
        """Consume a one-time token if present."""
        count = self._tokens.get(action, 0)
        if count <= 0:
            self._log({"event": "consume", "action": action, "allowed": False})
            return False
        if count == 1:
            self._tokens.pop(action, None)
        else:
            self._tokens[action] = count - 1
        self._log({"event": "consume", "action": action, "allowed": True})
        return True
