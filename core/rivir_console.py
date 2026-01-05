#!/usr/bin/env python3
"""
RIVIR Console: chat-first, intuitive, delightful.
"""

import curses
import json
import os
import random
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

try:
    from .structural_intelligence import StructuralIntelligence, DeepSignal
    from .consent_gate import ConsentGate
    from .speak_see_experience import WaveGenerator
except Exception:
    from structural_intelligence import StructuralIntelligence, DeepSignal
    from consent_gate import ConsentGate
    from speak_see_experience import WaveGenerator


@dataclass
class Message:
    role: str
    text: str


DEFAULT_VIBE = Path(__file__).parent / "fields" / "rivir_vibe_spec.json"
DEFAULT_RECEIPTS = Path("rivir_receipts.json")
DEFAULT_SI_DB = "rivir_si.db"
DEFAULT_LLM_KIND = "mlx_lm"
DEFAULT_LLM_MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"



def load_vibe(path: Path) -> dict:
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _is_question(text: str) -> bool:
    t = text.strip().lower()
    return "?" in t or t.startswith(("how", "what", "why", "can ", "do "))


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


def _glitch_reply(line: str) -> str:
    parts = [p for p in line.strip().split() if p]
    parts.reverse()
    return " ".join(parts) if parts else "..."


def rivir_reply(vibe: dict, line: str, mode: str) -> Tuple[str, Optional[str]]:
    templates = vibe.get("response_templates")
    silence_prob = float(vibe.get("response_silence_probability", 0.0))
    if random.random() < silence_prob:
        return ("", None)

    lower = line.strip().lower()
    if "color" in lower:
        return ("I can set colors here. Use /color user red or /color assistant blue.", None)
    if "wikipedia" in lower and ("read" in lower or "crawl" in lower):
        return (
            "I can do that with explicit consent. "
            "Grant web modality with: /modality allow web yes, "
            "then tell me the exact pages or seeds.",
            None,
        )
    if lower in ("hi", "hello", "hey", "yo"):
        return ("Here. What do you want to build or understand?", None)
    if _is_question(line):
        return ("Say what you want, and I will shape it. Add constraints if you have them.", None)
    if mode == "compose":
        return ("Noted. If you want structure, add inputs, outputs, and steps.", None)
    if templates:
        return (random.choice(templates), None)
    return (_glitch_reply(line), "no_templates_loaded")


class LLMBackend:
    def __init__(self, kind: str, model_id: str, max_tokens: int = 256, temperature: float = 0.7):
        self.kind = kind
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.model = None
        self.tokenizer = None

        if kind == "mlx_lm":
            from mlx_lm import load, generate  # type: ignore
            self._load = load
            self._generate = generate
            self.model, self.tokenizer = self._load(model_id)
        else:
            raise ValueError(f"Unknown LLM backend: {kind}")

    def generate(self, messages: List[Message], system_prompt: str = "", mood: str = "") -> str:
        prompt_lines = []
        if system_prompt:
            prompt_lines.append(f"System: {system_prompt}")
        if mood:
            prompt_lines.append(f"Mood: {mood}")
        for m in messages[-12:]:
            role = "User" if m.role == "user" else ("Assistant" if m.role == "assistant" else "System")
            prompt_lines.append(f"{role}: {m.text}")
        prompt_lines.append("Assistant:")
        prompt = "\n".join(prompt_lines)
        output = None
        try:
            output = self._generate(
                self.model,
                self.tokenizer,
                prompt,
                max_tokens=self.max_tokens,
                temp=self.temperature,
            )
        except TypeError:
            try:
                output = self._generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
            except TypeError:
                output = self._generate(
                    self.model,
                    self.tokenizer,
                    prompt,
                    max_tokens=self.max_tokens,
                )
        if isinstance(output, str):
            return output.strip()
        if isinstance(output, dict) and "text" in output:
            return str(output["text"]).strip()
        return str(output).strip()


def _safe_addstr(stdscr, y: int, x: int, text: str, width: int):
    if width <= 0 or y < 0:
        return
    try:
        stdscr.addstr(y, x, text[:width])
    except Exception:
        pass


def wrap_lines(text: str, width: int) -> List[str]:
    if width <= 1:
        return [text]
    return textwrap.wrap(text, width=width, replace_whitespace=False) or [""]


def draw(
    stdscr,
    messages: List[Message],
    status: dict,
    input_buf: str,
    theme: dict,
):
    stdscr.clear()
    h, w = stdscr.getmaxyx()
    sidebar_w = min(36, w // 3) if w >= 90 else 0
    main_w = w - sidebar_w - (1 if sidebar_w else 0)
    header = "RIVIR Console  |  /help  /mode  /voice  /modality  /stats  /set  /quit"
    _safe_addstr(stdscr, 0, 0, header, main_w)
    stdscr.hline(1, 0, "-", main_w)

    # Chat area
    chat_h = h - 4
    lines: List[str] = []
    for m in messages:
        prefix = f"{m.role}: "
        wrapped = wrap_lines(m.text, max(10, main_w - len(prefix)))
        if wrapped:
            lines.append((m.role, prefix + wrapped[0]))
            for cont in wrapped[1:]:
                lines.append((m.role, " " * len(prefix) + cont))
    start = max(0, len(lines) - chat_h)
    y = 2
    for role, line in lines[start:]:
        if y >= h - 2:
            break
        color = theme.get(role)
        if color is not None:
            try:
                stdscr.addstr(y, 0, line[:main_w], curses.color_pair(color))
            except Exception:
                _safe_addstr(stdscr, y, 0, line, main_w)
        else:
            _safe_addstr(stdscr, y, 0, line, main_w)
        y += 1

    stdscr.hline(h - 3, 0, "-", main_w)
    _safe_addstr(stdscr, h - 2, 0, "You> " + input_buf, main_w)

    if sidebar_w:
        x0 = main_w + 1
        stdscr.vline(0, x0 - 1, "|", h)
        _safe_addstr(stdscr, 0, x0, "STATUS", sidebar_w)
        stdscr.hline(1, x0, "-", sidebar_w)
        _safe_addstr(stdscr, 2, x0, f"mode: {status['mode']}", sidebar_w)
        _safe_addstr(stdscr, 3, x0, f"voice: {'on' if status['voice'] else 'off'}", sidebar_w)
        _safe_addstr(stdscr, 4, x0, f"modalities: {status['modalities']}", sidebar_w)
        _safe_addstr(stdscr, 5, x0, f"si: {'on' if status['si'] else 'off'}", sidebar_w)
        _safe_addstr(stdscr, 6, x0, f"vibe: {'loaded' if status['vibe'] else 'none'}", sidebar_w)
        _safe_addstr(stdscr, 7, x0, f"llm: {'on' if status['llm'] else 'off'}", sidebar_w)
        _safe_addstr(stdscr, 8, x0, f"mood: {status['mood']}", sidebar_w)
        _safe_addstr(stdscr, 9, x0, f"receipts: {status['receipts']}", sidebar_w)

    stdscr.refresh()


def main(stdscr):
    curses.curs_set(1)
    stdscr.nodelay(False)
    messages: List[Message] = []
    input_buf = ""
    mode = "chat"

    vibe_path_env = os.environ.get("RIVIR_VIBE_SPEC")
    receipts_env = os.environ.get("RIVIR_RECEIPTS")
    si_db_env = os.environ.get("RIVIR_SI_DB")
    llm_kind = os.environ.get("RIVIR_LLM")
    llm_model = os.environ.get("RIVIR_LLM_MODEL")
    voice_enabled = str(os.environ.get("RIVIR_VOICE", "")).lower() in ("1", "true", "yes", "on")

    vibe_path = Path(vibe_path_env) if vibe_path_env else DEFAULT_VIBE
    receipts_path = Path(receipts_env) if receipts_env else DEFAULT_RECEIPTS
    si_db = si_db_env if si_db_env else DEFAULT_SI_DB

    vibe = load_vibe(vibe_path)
    if not vibe:
        messages.append(Message("system", f"vibe load failed: {vibe_path}"))
    affirmatives = vibe.get("affirmatives")
    if affirmatives and isinstance(affirmatives, list):
        gate = ConsentGate(log_path=None, affirmatives=set(a.strip().lower() for a in affirmatives if a))
    else:
        gate = ConsentGate(log_path=None)
    allowed_modalities = set()
    si = StructuralIntelligence(db_path=si_db)
    generator = WaveGenerator()

    llm = None
    mood = ""
    llm_kind = llm_kind or vibe.get("llm_backend") or DEFAULT_LLM_KIND
    llm_model = llm_model or vibe.get("llm_model") or DEFAULT_LLM_MODEL
    if llm_kind and llm_model:
        try:
            llm = LLMBackend(llm_kind, llm_model, max_tokens=256, temperature=0.7)
            mood = llm.generate([Message("user", "Pick a single-word mood for this session.")]).split()[0]
            messages.append(Message("system", f"llm: {llm_kind} loaded ({llm_model})"))
        except Exception as e:
            messages.append(Message("system", f"llm load failed: {e}"))

    if curses.has_colors():
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
    theme = {"assistant": 1, "user": 2, "system": 3}

    try:
        while True:
            status = {
                "mode": mode,
                "voice": voice_enabled,
                "modalities": sorted(allowed_modalities) if allowed_modalities else "none",
                "si": si is not None,
                "vibe": bool(vibe),
                "receipts": len(messages),
                "llm": llm is not None,
                "mood": mood or "none",
            }
            draw(stdscr, messages, status, input_buf, theme)

            ch = stdscr.get_wch()
            if isinstance(ch, str) and ch in ("\n", "\r"):
                line = input_buf.strip()
                input_buf = ""
                if not line:
                    continue

                if line.startswith("/"):
                    cmd = line[1:].strip()
                    if cmd in ("quit", "exit"):
                        return
                    if cmd == "help":
                        messages.append(Message("system", "Use /mode chat|compose, /voice on|off, /modality allow <name> <signal>, /stats, /set, /color, /export, /clear, /quit."))
                        continue
                    if cmd.startswith("mode"):
                        parts = cmd.split(" ", 1)
                        if len(parts) == 2 and parts[1] in ("chat", "compose"):
                            mode = parts[1]
                            messages.append(Message("system", f"mode: {mode}"))
                        else:
                            messages.append(Message("system", "usage: /mode chat|compose"))
                        continue
                    if cmd.startswith("voice"):
                        parts = cmd.split(" ", 1)
                        if len(parts) == 2 and parts[1] in ("on", "off"):
                            if parts[1] == "on" and "voice" not in allowed_modalities:
                                messages.append(Message("system", "voice modality not granted"))
                            else:
                                voice_enabled = parts[1] == "on"
                                messages.append(Message("system", f"voice: {parts[1]}"))
                        else:
                            messages.append(Message("system", "usage: /voice on|off"))
                        continue
                    if cmd.startswith("modality"):
                        parts = cmd.split(" ", 3)
                        if len(parts) >= 2 and parts[1] == "list":
                            messages.append(Message("system", f"modalities: {sorted(allowed_modalities) if allowed_modalities else 'none'}"))
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
                            messages.append(Message("system", f"modality {name}: {'granted' if allowed else 'denied'}"))
                            continue
                        messages.append(Message("system", "usage: /modality allow <name> <signal> | /modality list"))
                        continue
                    if cmd.startswith("set "):
                        parts = cmd.split(" ", 2)
                        if len(parts) < 3:
                            messages.append(Message("system", "usage: /set <vibe|receipts|si|llm> <path>"))
                            continue
                        _, key, value = parts
                        if key == "vibe":
                            vibe_path = Path(value)
                            vibe = load_vibe(vibe_path)
                            messages.append(Message("system", f"vibe: {'loaded' if vibe else 'failed'}"))
                            continue
                        if key == "receipts":
                            receipts_path = Path(value)
                            messages.append(Message("system", f"receipts: {receipts_path}"))
                            continue
                        if key == "si":
                            if si:
                                si.close()
                            si = StructuralIntelligence(db_path=value)
                            messages.append(Message("system", f"si: {value}"))
                            continue
                        if key == "llm":
                            if ":" not in value:
                                messages.append(Message("system", "usage: /set llm mlx_lm:<model_id>"))
                                continue
                            kind, model_id = value.split(":", 1)
                            try:
                                llm = LLMBackend(kind, model_id, max_tokens=256, temperature=0.7)
                                mood = llm.generate([Message("user", "Pick a single-word mood for this session.")]).split()[0]
                                messages.append(Message("system", f"llm: {kind} loaded ({model_id})"))
                            except Exception as e:
                                messages.append(Message("system", f"llm load failed: {e}"))
                            continue
                        messages.append(Message("system", "unknown set key"))
                        continue
                    if cmd.startswith("color "):
                        parts = cmd.split(" ", 2)
                        if len(parts) < 3:
                            messages.append(Message("system", "usage: /color user|assistant|system red|green|yellow|blue"))
                            continue
                        _, who, color = parts
                        color = color.lower()
                        palette = {
                            "red": curses.COLOR_RED,
                            "green": curses.COLOR_GREEN,
                            "yellow": curses.COLOR_YELLOW,
                            "blue": curses.COLOR_BLUE,
                            "cyan": curses.COLOR_CYAN,
                            "magenta": curses.COLOR_MAGENTA,
                            "white": curses.COLOR_WHITE,
                        }
                        if who not in ("user", "assistant", "system") or color not in palette:
                            messages.append(Message("system", "usage: /color user|assistant|system red|green|yellow|blue"))
                            continue
                        if curses.has_colors():
                            pair_id = {"assistant": 1, "user": 2, "system": 3}[who]
                            curses.init_pair(pair_id, palette[color], -1)
                            theme[who] = pair_id
                            messages.append(Message("system", f"{who} color: {color}"))
                        continue
                    if cmd.startswith("export "):
                        _, path = cmd.split(" ", 1)
                        try:
                            Path(path).write_text("\n".join(f"{m.role}: {m.text}" for m in messages))
                            messages.append(Message("system", f"exported: {path}"))
                        except Exception:
                            messages.append(Message("system", "export failed"))
                        continue
                    if cmd == "clear":
                        messages.clear()
                        messages.append(Message("system", "cleared"))
                        continue
                    if cmd == "stats":
                        if not si:
                            messages.append(Message("system", "si: off"))
                            continue
                        summary = si.get_invariant_summary()
                        total = sum(len(v) for v in summary.values())
                        trend = si.get_satisfaction_trend(5)
                        msg = f"invariants: {total}"
                        if trend:
                            msg += f", recent satisfaction: {sum(trend)/len(trend):.2f}"
                        messages.append(Message("system", msg))
                        continue

                if line.startswith("RIVIR_") or line.startswith("python3 "):
                    messages.append(Message("system", "That looks like a shell command. Run it in your terminal, not inside RIVIR."))
                    continue

                messages.append(Message("user", line))
                if llm:
                    reply = llm.generate(messages + [Message("user", line)], system_prompt=vibe.get("system_prompt", ""), mood=mood)
                    note = None
                else:
                    reply, note = rivir_reply(vibe, line, mode)
                if note == "no_templates_loaded":
                    messages.append(Message("system", "no templates loaded; using glitch reply"))
                if reply:
                    messages.append(Message("assistant", reply))
                    if voice_enabled and _should_speak(vibe, reply, line):
                        try:
                            subprocess.run(["say", reply], check=False)
                        except Exception:
                            messages.append(Message("system", "voice failed"))

                # Provide concrete wave hints when asked to build.
                if "build" in line.lower() or "browser" in line.lower():
                    try:
                        waves = generator.generate_waves(line, n=2)
                        hint = " | ".join(f"{w.description}: {w.implementation_style}" for w in waves)
                        messages.append(Message("assistant", f"Options: {hint}"))
                    except Exception:
                        pass

                if si:
                    deep = DeepSignal(
                        pressure="live chat",
                        urgency="now",
                        prior_blockers="",
                        stakeholders=[vibe.get("owner", "bruce_stagbrook")],
                        drives=vibe.get("drives", []),
                        commitments=vibe.get("commitments", []),
                        dreams=vibe.get("dreams", []),
                    )
                    si.record_interaction(
                        utterance=line,
                        shape_key=line,
                        satisfaction=0.7,
                        shape_dyck="",
                        deep_signals=deep,
                        full_disclosure=f"mode:{mode}",
                    )
                if receipts_path:
                    try:
                        receipts_path.write_text(json.dumps([m.__dict__ for m in messages], indent=2))
                    except Exception:
                        pass
                continue

            if ch == curses.KEY_BACKSPACE or ch == "\b" or ch == "\x7f":
                input_buf = input_buf[:-1]
            elif isinstance(ch, str):
                input_buf += ch
    finally:
        if si:
            si.close()


if __name__ == "__main__":
    curses.wrapper(main)
