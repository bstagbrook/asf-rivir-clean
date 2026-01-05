"""
TUI demo: conversational spec -> ASF1 "does" shape in real time.
"""

import curses
import json
import os
import random
import re
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

try:
    from . import asf1
    from .structural_intelligence import StructuralIntelligence, DeepSignal
    from .consent_gate import ConsentGate
except Exception:
    import asf1
    from structural_intelligence import StructuralIntelligence, DeepSignal
    from consent_gate import ConsentGate


def sexpr_to_string(expr):
    if isinstance(expr, asf1.Symbol):
        return expr.name
    if isinstance(expr, asf1.SList):
        inner = " ".join(sexpr_to_string(x) for x in expr.items)
        return "(" + inner + ")"
    return "<?>"


def to_symbol(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "item"


class ComposerState:
    def __init__(self):
        self.name = "unnamed"
        self.inputs: List[str] = []
        self.outputs: List[str] = []
        self.body: List[str] = []

    def update_from_text(self, text: str):
        t = text.strip()
        if not t:
            return
        lower = t.lower()
        if lower.startswith("name:"):
            self.name = to_symbol(t.split(":", 1)[1])
            return
        if lower.startswith("inputs:") or lower.startswith("input:"):
            raw = t.split(":", 1)[1]
            items = re.split(r"[ ,]+", raw.strip())
            self.inputs = [to_symbol(i) for i in items if i]
            return
        if lower.startswith("outputs:") or lower.startswith("output:"):
            raw = t.split(":", 1)[1]
            items = re.split(r"[ ,]+", raw.strip())
            self.outputs = [to_symbol(i) for i in items if i]
            return
        if lower.startswith("steps:") or lower.startswith("behavior:") or lower.startswith("body:"):
            raw = t.split(":", 1)[1]
            items = [to_symbol(x) for x in re.split(r"[;,]+", raw) if x.strip()]
            self.body.extend(items)
            return
        self.body.append(to_symbol(t))

    def asf1_expr(self):
        return asf1.asf1_does(self.inputs, self.outputs, self.body)

    def snapshot(self):
        return {
            "name": self.name,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "body": list(self.body),
        }

def load_vibe_spec(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_question(text: str) -> bool:
    return "?" in text or text.strip().lower().startswith(("how", "what", "why", "can ", "do "))

def _voice_policy(vibe: dict) -> dict:
    policy = vibe.get("voice_policy", {})
    return {
        "require_question": bool(policy.get("require_question", True)),
        "min_chars": int(policy.get("min_chars", 6)),
        "block_patterns": list(policy.get("block_patterns", ["(", ")", "_", "http", "python3", "/"])),
    }


def _should_speak(vibe: dict, response: str, line: str) -> bool:
    policy = _voice_policy(vibe)
    if len(response.strip()) < policy["min_chars"]:
        return False
    if policy["require_question"] and not _is_question(line):
        return False
    lower = response.lower()
    for pat in policy["block_patterns"]:
        if pat in lower:
            return False
    return True


def rivir_response(vibe: dict, expr, line: str, state: ComposerState) -> str:
    templates = vibe.get("response_templates", [])
    silence_prob = float(vibe.get("response_silence_probability", 1.0))
    if not templates or random.random() < silence_prob:
        return ""
    lower = line.strip().lower()
    if "ready" in lower:
        if state.inputs and state.outputs and state.body:
            return "Ready is true when inputs, outputs, and steps are present."
        return "Ready is false because inputs/outputs/steps are missing. Try: inputs:, outputs:, steps:."
    if lower.startswith(("inputs:", "input:")):
        return "Inputs locked."
    if lower.startswith(("outputs:", "output:")):
        return "Outputs locked."
    if lower.startswith(("steps:", "behavior:", "body:")):
        return "Steps locked."
    if _is_question(line):
        return "I can help. Give me inputs, outputs, and steps."
    template = random.choice(templates)
    try:
        return template.format(asf1=sexpr_to_string(expr))
    except Exception:
        return template

def satisfaction_guess(text: str) -> float:
    t = text.strip().lower()
    if t.startswith("no") or "not" in t:
        return 0.1
    if "yes" in t or "exactly" in t or "correct" in t:
        return 0.9
    return 0.5


def _safe_addstr(stdscr, y: int, x: int, text: str, width: int):
    if width <= 0:
        return
    try:
        stdscr.addstr(y, x, text[:width])
    except Exception:
        pass


def draw(
    stdscr,
    messages: List[Tuple[str, str]],
    state: ComposerState,
    receipts: List[dict],
    voice_enabled: bool,
    allowed_modalities: set,
    si_enabled: bool,
    vibe_loaded: bool,
):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    split = max(8, h // 2)
    stats_w = 0
    if w >= 90:
        stats_w = min(40, w // 3)
    left_w = w - stats_w - (1 if stats_w else 0)

    _safe_addstr(stdscr, 0, 0, "ASF Composer  |  /help for commands", left_w)
    stdscr.hline(1, 0, "-", left_w)

    y = 2
    for role, text in messages[-(split - 3):]:
        _safe_addstr(stdscr, y, 0, f"{role}: {text}", left_w)
        y += 1

    stdscr.hline(split, 0, "=", left_w)
    _safe_addstr(stdscr, split + 1, 0, f"Name: {state.name}", left_w)
    _safe_addstr(stdscr, split + 2, 0, f"Inputs: {state.inputs}", left_w)
    _safe_addstr(stdscr, split + 3, 0, f"Outputs: {state.outputs}", left_w)
    _safe_addstr(stdscr, split + 4, 0, f"Body: {state.body[-5:]}", left_w)

    expr = state.asf1_expr()
    sexpr = sexpr_to_string(expr)
    dyck = asf1.serialize_dyck(asf1.encode_sexpr(expr))

    _safe_addstr(stdscr, split + 6, 0, "ASF1:", left_w)
    _safe_addstr(stdscr, split + 7, 0, sexpr, left_w)
    _safe_addstr(stdscr, split + 8, 0, "Dyck prefix:", left_w)
    dyck_line = dyck[:left_w - 1] + ("..." if len(dyck) > left_w - 1 else "")
    _safe_addstr(stdscr, split + 9, 0, dyck_line, left_w)

    ready = bool(state.inputs and state.outputs and state.body)
    _safe_addstr(stdscr, split + 11, 0, f"Ready: {ready}", left_w)
    if receipts:
        last = receipts[-1]
        if split + 12 < h:
            _safe_addstr(stdscr, split + 12, 0, f"Receipt: sat={last['satisfaction']:.2f} utterance={last['utterance'][:30]}", left_w)
        if split + 13 < h:
            _safe_addstr(stdscr, split + 13, 0, f"Receipts: {len(receipts)} saved", left_w)

    if stats_w:
        x0 = left_w + 1
        stdscr.vline(0, x0 - 1, "|", h)
        _safe_addstr(stdscr, 0, x0, "STATUS", stats_w)
        stdscr.hline(1, x0, "-", stats_w)
        _safe_addstr(stdscr, 2, x0, f"voice: {'on' if voice_enabled else 'off'}", stats_w)
        _safe_addstr(stdscr, 3, x0, f"modalities: {sorted(allowed_modalities) if allowed_modalities else 'none'}", stats_w)
        _safe_addstr(stdscr, 4, x0, f"si: {'on' if si_enabled else 'off'}", stats_w)
        _safe_addstr(stdscr, 5, x0, f"vibe: {'loaded' if vibe_loaded else 'none'}", stats_w)
        _safe_addstr(stdscr, 6, x0, f"receipts: {len(receipts)}", stats_w)
    stdscr.refresh()


def _config_from_env() -> dict:
    return {
        "vibe_path": os.environ.get("RIVIR_VIBE_SPEC"),
        "receipts_path": os.environ.get("RIVIR_RECEIPTS"),
        "si_db_path": os.environ.get("RIVIR_SI_DB"),
        "consent_log": os.environ.get("RIVIR_CONSENT_LOG"),
        "voice": os.environ.get("RIVIR_VOICE"),
    }


def _config_from_args(argv: List[str]) -> dict:
    config = {
        "vibe_path": None,
        "receipts_path": None,
        "si_db_path": None,
        "consent_log": None,
        "voice": None,
    }
    args = list(argv)
    while args:
        arg = args.pop(0)
        if arg == "--vibe" and args:
            config["vibe_path"] = args.pop(0)
        elif arg == "--receipts" and args:
            config["receipts_path"] = args.pop(0)
        elif arg == "--si-db" and args:
            config["si_db_path"] = args.pop(0)
        elif arg == "--consent-log" and args:
            config["consent_log"] = args.pop(0)
        elif arg == "--voice" and args:
            config["voice"] = args.pop(0)
    return config


def main(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(False)
    messages: List[Tuple[str, str]] = []
    state = ComposerState()
    receipts: List[dict] = []
    config = _config_from_env()
    config.update({k: v for k, v in _config_from_args(sys.argv[1:]).items() if v})

    default_vibe = Path(__file__).parent / "fields" / "rivir_vibe_spec.json"
    default_receipts = Path("tui_receipts.json")
    default_si = "rivir_si.db"

    receipts_path = Path(config["receipts_path"]) if config.get("receipts_path") else default_receipts
    vibe_path = Path(config["vibe_path"]) if config.get("vibe_path") else default_vibe
    vibe = load_vibe_spec(vibe_path) if vibe_path else {}
    consent_log = Path(config["consent_log"]) if config.get("consent_log") else None
    gate = ConsentGate(log_path=consent_log)
    voice_enabled = str(config.get("voice", "")).lower() in ("1", "true", "yes", "on")
    allowed_modalities = set()

    si = StructuralIntelligence(db_path=config.get("si_db_path") or default_si)

    try:
        while True:
            draw(
                stdscr,
                messages,
                state,
                receipts,
                voice_enabled,
                allowed_modalities,
                si is not None,
                bool(vibe),
            )
            stdscr.addstr(curses.LINES - 1, 0, "You> ")
            stdscr.clrtoeol()
            curses.echo()
            try:
                line = stdscr.getstr(curses.LINES - 1, 5).decode("utf-8")
            except Exception:
                line = ""
            curses.noecho()

            if line.startswith("/"):
                cmd = line[1:].strip()
                if cmd in ("quit", "exit"):
                    return
                if cmd.startswith("allow "):
                    parts = cmd.split(" ", 2)
                    if len(parts) < 3:
                        messages.append(("system", "usage: /allow <action> <signal>"))
                        continue
                    _, action, signal = parts
                    allowed = gate.request(action, signal)
                    messages.append(("system", f"allow {action}: {'granted' if allowed else 'denied'}"))
                    continue
                if cmd.startswith("do "):
                    action = cmd.split(" ", 1)[1].strip()
                    allowed = gate.consume(action)
                    messages.append(("system", f"do {action}: {'allowed' if allowed else 'blocked'}"))
                    continue
                if cmd == "reset":
                    state = ComposerState()
                    messages.append(("system", "reset"))
                    continue
                if cmd == "help":
                    messages.append(("system", "commands: /help /reset /quit /allow /do /voice /stats /modality /set /vibe"))
                    continue
                if cmd == "stats":
                    if not si:
                        messages.append(("system", "no SI DB configured"))
                        continue
                    summary = si.get_invariant_summary()
                    total = sum(len(v) for v in summary.values())
                    trend = si.get_satisfaction_trend(5)
                    messages.append(("system", f"invariants: {total}"))
                    if trend:
                        avg = sum(trend) / len(trend)
                        messages.append(("system", f"recent satisfaction: {avg:.2f}"))
                    if receipts:
                        last = receipts[-1]
                        messages.append(("system", f"last receipt: {last.get('utterance','')[:30]}"))
                    continue
                if cmd.startswith("modality") or cmd.startswith("modalities"):
                    parts = cmd.split(" ", 3)
                    if len(parts) >= 2 and parts[1] == "list":
                        if allowed_modalities:
                            messages.append(("system", f"modalities: {sorted(allowed_modalities)}"))
                        else:
                            messages.append(("system", "modalities: none"))
                        continue
                    if len(parts) >= 3 and parts[1] == "allow":
                        if len(parts) == 3:
                            name, signal = "voice", parts[2]
                        else:
                            _, _, name, signal = parts
                        allowed = gate.request(f"modality:{name}", signal)
                        if allowed:
                            allowed_modalities.add(name)
                            if name == "voice":
                                voice_enabled = True
                                messages.append(("system", "voice: on"))
                        messages.append(("system", f"modality {name}: {'granted' if allowed else 'denied'}"))
                        continue
                    messages.append(("system", "usage: /modality allow <name> <signal> | /modality list"))
                    continue
                if cmd.startswith("set "):
                    parts = cmd.split(" ", 2)
                    if len(parts) < 3:
                        messages.append(("system", "usage: /set <vibe|receipts|si|consent> <path>"))
                        continue
                    _, key, value = parts
                    value = value.strip()
                    if key == "vibe":
                        vibe_path = Path(value)
                        vibe = load_vibe_spec(vibe_path)
                        if vibe:
                            messages.append(("system", f"vibe loaded: {vibe_path}"))
                        else:
                            messages.append(("system", f"vibe load failed: {vibe_path}"))
                        continue
                    if key == "receipts":
                        receipts_path = Path(value)
                        messages.append(("system", f"receipts path: {receipts_path}"))
                        continue
                    if key == "si":
                        if si:
                            si.close()
                        si = StructuralIntelligence(db_path=value)
                        messages.append(("system", f"si db: {value}"))
                        continue
                    if key == "consent":
                        consent_log = Path(value)
                        gate.log_path = consent_log
                        messages.append(("system", f"consent log: {consent_log}"))
                        continue
                    messages.append(("system", "unknown set key"))
                    continue
                if cmd == "vibe":
                    if vibe_path and vibe:
                        messages.append(("system", f"vibe loaded: {vibe_path}"))
                    elif vibe_path and not vibe:
                        messages.append(("system", f"vibe load failed: {vibe_path}"))
                    else:
                        messages.append(("system", "vibe: none"))
                    continue
                if cmd.startswith("voice"):
                    parts = cmd.split(" ", 1)
                    if len(parts) == 2 and parts[1].strip().lower() in ("on", "off"):
                        if parts[1].strip().lower() == "on" and "voice" not in allowed_modalities:
                            messages.append(("system", "voice modality not granted"))
                            continue
                        voice_enabled = parts[1].strip().lower() == "on"
                        messages.append(("system", f"voice: {'on' if voice_enabled else 'off'}"))
                        continue
                    messages.append(("system", "usage: /voice on|off"))
                    continue
                messages.append(("system", f"unknown command: {cmd}"))
                continue

            messages.append(("user", line))
            state.update_from_text(line)
            expr = state.asf1_expr()
            dyck = asf1.serialize_dyck(asf1.encode_sexpr(expr))
            sat = satisfaction_guess(line)
            deep = DeepSignal(
                pressure="live composition",
                urgency="now",
                prior_blockers="",
                stakeholders=[vibe.get("owner", "bruce_stagbrook")],
                drives=vibe.get("drives", []),
                commitments=vibe.get("commitments", []),
                dreams=vibe.get("dreams", []),
            )
            receipts.append({
                "utterance": line,
                "state": state.snapshot(),
                "shape": dyck,
                "satisfaction": sat,
            })
            if receipts_path:
                try:
                    receipts_path.write_text(json.dumps(receipts, indent=2))
                except Exception:
                    messages.append(("system", "failed to save receipts"))
            if si:
                si.record_interaction(
                    utterance=line,
                    shape_key=dyck,
                    satisfaction=sat,
                    shape_dyck=dyck,
                    deep_signals=deep,
                    full_disclosure="tui_demo"
                )
            response = rivir_response(vibe, expr, line, state)
            if response:
                messages.append(("assistant", response))
                if voice_enabled and _should_speak(vibe, response, line):
                    try:
                        subprocess.run(["say", response], check=False)
                    except Exception:
                        messages.append(("system", "voice failed"))
    finally:
        if si:
            si.close()


if __name__ == "__main__":
    curses.wrapper(main)
