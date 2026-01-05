"""
Living governance runtime: hats + causal field enforcement.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class FieldProperty:
    name: str
    threshold: float
    description: str


@dataclass
class Hat:
    name: str
    concerns: List[str]


@dataclass
class GovernanceDecision:
    action_id: str
    decision: str
    field_alignment: float
    per_property: Dict[str, float]
    hat_scores: Dict[str, float]
    notes: str


class GovernanceRuntime:
    def __init__(self, field_spec: dict):
        self.field_spec = field_spec
        self.properties = [
            FieldProperty(p["name"], float(p["threshold"]), p.get("description", ""))
            for p in field_spec.get("properties", [])
        ]
        self.hats = [Hat(h["name"], list(h.get("concerns", []))) for h in field_spec.get("hats", [])]
        self.enforcement = field_spec.get("enforcement", {})
        self.receipts: List[dict] = []

    @classmethod
    def from_json(cls, path: str) -> "GovernanceRuntime":
        with open(path, "r", encoding="utf-8") as handle:
            spec = json.load(handle)
        return cls(spec)

    def evaluate_action(self, action: dict) -> GovernanceDecision:
        action_id = str(action.get("id", f"a-{len(self.receipts) + 1}"))
        signals = dict(action.get("signals", {}))
        per_property = {p.name: float(signals.get(p.name, 0.0)) for p in self.properties}
        if self.properties:
            field_alignment = sum(per_property.values()) / len(self.properties)
        else:
            field_alignment = 0.0

        hat_scores = {}
        for hat in self.hats:
            scores = [float(signals.get(c, 0.0)) for c in hat.concerns]
            hat_scores[hat.name] = sum(scores) / len(scores) if scores else 0.0

        decision, notes = self._decide(per_property, field_alignment)
        return GovernanceDecision(
            action_id=action_id,
            decision=decision,
            field_alignment=field_alignment,
            per_property=per_property,
            hat_scores=hat_scores,
            notes=notes,
        )

    def _decide(self, per_property: Dict[str, float], field_alignment: float) -> Tuple[str, str]:
        hard_block = set(self.enforcement.get("hard_block", []))
        min_alignment = float(self.enforcement.get("min_alignment", 0.0))
        mode = self.enforcement.get("mode", "allow")
        violated = []
        for prop in self.properties:
            score = per_property.get(prop.name, 0.0)
            if score < prop.threshold:
                violated.append(prop.name)
        if hard_block and any(p in hard_block for p in violated):
            return "block", f"hard block on {sorted(set(violated) & hard_block)}"
        if field_alignment < min_alignment:
            if mode == "allow_then_repair":
                return "repair", f"alignment {field_alignment:.2f} below {min_alignment:.2f}"
            return "block", f"alignment {field_alignment:.2f} below {min_alignment:.2f}"
        return "allow", "field-aligned"

    def apply_action(self, action: dict) -> GovernanceDecision:
        decision = self.evaluate_action(action)
        receipt = {
            "seq": len(self.receipts) + 1,
            "timestamp": time.time(),
            "action": action,
            "decision": decision.decision,
            "field_alignment": decision.field_alignment,
            "per_property": decision.per_property,
            "hat_scores": decision.hat_scores,
            "notes": decision.notes,
        }
        self.receipts.append(receipt)
        return decision

    def summary(self) -> dict:
        return {
            "field": self.field_spec.get("name"),
            "receipts": len(self.receipts),
        }


def load_runtime(path: str) -> GovernanceRuntime:
    return GovernanceRuntime.from_json(path)


