#!/usr/bin/env python3
"""
Premium Speak-See-Experience Demo

High-end software generation with:
- Semantic shape equivalence
- Alpha-normalization
- Insta-software that WOWs
- Real catalog stacking
- Shape-shifter training

This is NOT cheap sketches. This is production-ready quality.
"""

import hashlib
import json
import time
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Core imports
from py_to_dyck import compile_source, run_dyck, parse_dyck, serialize_dyck
from weighted_state import from_shapes, normalize_l2

# Load asf_core2 for catalog and normalization
import importlib.util
def load_core2():
    path = Path("/Volumes/StagbrookField/stagbrook_field/.asf_core2.py")
    spec = importlib.util.spec_from_file_location("asf_core2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

core2 = load_core2()


# =============================================================================
# SEMANTIC EQUIVALENCE & ALPHA-NORMALIZATION
# =============================================================================

class ShapeCatalog:
    """
    Content-addressed catalog with semantic equivalence.
    Alpha-normalizes before hashing to recognize equivalent shapes.
    """

    def __init__(self, db_path: str = "premium_catalog.db"):
        self.db_path = db_path
        self.catalog = core2.PersistentCatalog(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._extend_schema()

    def _extend_schema(self):
        """Extend catalog with semantic equivalence tables."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS equivalence_classes (
                canonical_key TEXT PRIMARY KEY,
                members TEXT,
                description TEXT,
                created REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS high_quality_templates (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE,
                shape_key TEXT,
                dyck TEXT,
                template TEXT,
                quality_score REAL,
                usage_count INTEGER DEFAULT 0
            )
        """)
        self.conn.commit()

    def put(self, shape) -> str:
        """Store a shape, return content key."""
        return self.catalog.put(shape)

    def get(self, key: str):
        """Retrieve by content key."""
        return self.catalog.get(key)

    def normalize_alpha(self, shape) -> Tuple[Any, str]:
        """
        Alpha-normalize a shape (rename bound variables canonically).
        Returns (normalized_shape, canonical_key).
        """
        # Use core2's normalization
        normalized, status = core2.semantic_normalize(shape)
        key = core2.key(normalized)
        return normalized, key

    def find_equivalent(self, shape) -> Optional[str]:
        """Find an equivalent shape in the catalog."""
        _, canonical_key = self.normalize_alpha(shape)
        row = self.conn.execute(
            "SELECT canonical_key FROM equivalence_classes WHERE canonical_key = ?",
            (canonical_key,)
        ).fetchone()
        return row['canonical_key'] if row else None

    def register_equivalence(self, shape, description: str = ""):
        """Register a shape's equivalence class."""
        normalized, canonical_key = self.normalize_alpha(shape)
        shape_key = self.put(normalized)

        existing = self.conn.execute(
            "SELECT members FROM equivalence_classes WHERE canonical_key = ?",
            (canonical_key,)
        ).fetchone()

        if existing:
            members = json.loads(existing['members'])
            if shape_key not in members:
                members.append(shape_key)
                self.conn.execute(
                    "UPDATE equivalence_classes SET members = ? WHERE canonical_key = ?",
                    (json.dumps(members), canonical_key)
                )
        else:
            self.conn.execute(
                "INSERT INTO equivalence_classes (canonical_key, members, description, created) VALUES (?, ?, ?, ?)",
                (canonical_key, json.dumps([shape_key]), description, time.time())
            )
        self.conn.commit()
        return canonical_key

    def add_template(self, name: str, dyck: str, template: str, quality: float = 1.0):
        """Add a high-quality template to the catalog."""
        shape = parse_dyck(dyck)
        shape_key = self.put(shape)
        self.conn.execute("""
            INSERT OR REPLACE INTO high_quality_templates
            (name, shape_key, dyck, template, quality_score, usage_count)
            VALUES (?, ?, ?, ?, ?, 0)
        """, (name, shape_key, dyck, template, quality))
        self.conn.commit()

    def get_template(self, name: str) -> Optional[dict]:
        """Get a template by name."""
        row = self.conn.execute(
            "SELECT * FROM high_quality_templates WHERE name = ?",
            (name,)
        ).fetchone()
        if row:
            # Increment usage
            self.conn.execute(
                "UPDATE high_quality_templates SET usage_count = usage_count + 1 WHERE name = ?",
                (name,)
            )
            self.conn.commit()
            return dict(row)
        return None

    def search_templates(self, query: str, limit: int = 5) -> List[dict]:
        """Search templates by name/description."""
        rows = self.conn.execute("""
            SELECT * FROM high_quality_templates
            WHERE name LIKE ? OR template LIKE ?
            ORDER BY quality_score DESC, usage_count DESC
            LIMIT ?
        """, (f"%{query}%", f"%{query}%", limit)).fetchall()
        return [dict(r) for r in rows]

    def close(self):
        self.conn.close()


# =============================================================================
# HIGH-QUALITY SOFTWARE TEMPLATES
# =============================================================================

PREMIUM_TEMPLATES = {
    "todo": {
        "name": "Premium Todo App",
        "dyck": "(((())(()()()())))",
        "template": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --bg: #fafafa;
            --surface: #ffffff;
            --text: #1f2937;
            --text-muted: #6b7280;
            --border: #e5e7eb;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: 2rem;
        }
        .container {
            width: 100%;
            max-width: 600px;
            background: var(--surface);
            border-radius: 16px;
            box-shadow: var(--shadow);
            padding: 2rem;
        }
        h1 {
            font-size: 1.75rem;
            font-weight: 600;
            margin-bottom: 1.5rem;
            color: var(--primary);
        }
        .input-group {
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }
        input[type="text"] {
            flex: 1;
            padding: 0.75rem 1rem;
            border: 2px solid var(--border);
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.2s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: var(--primary);
        }
        button {
            padding: 0.75rem 1.5rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: background 0.2s;
        }
        button:hover { background: var(--primary-dark); }
        .task-list { list-style: none; }
        .task-item {
            display: flex;
            align-items: center;
            padding: 1rem;
            border-bottom: 1px solid var(--border);
            transition: background 0.2s;
        }
        .task-item:hover { background: #f9fafb; }
        .task-item.done .task-text { text-decoration: line-through; color: var(--text-muted); }
        .task-checkbox {
            width: 20px;
            height: 20px;
            margin-right: 1rem;
            accent-color: var(--primary);
        }
        .task-text { flex: 1; font-size: 1rem; }
        .task-delete {
            opacity: 0;
            background: none;
            border: none;
            color: #ef4444;
            cursor: pointer;
            padding: 0.5rem;
            font-size: 1.25rem;
            transition: opacity 0.2s;
        }
        .task-item:hover .task-delete { opacity: 1; }
        .empty-state {
            text-align: center;
            padding: 3rem;
            color: var(--text-muted);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>{{title}}</h1>
        <div class="input-group">
            <input type="text" id="taskInput" placeholder="What needs to be done?" autofocus>
            <button onclick="addTask()">Add</button>
        </div>
        <ul class="task-list" id="taskList">
            <li class="empty-state">No tasks yet. Add one above!</li>
        </ul>
    </div>
    <script>
        let tasks = [];

        function render() {
            const list = document.getElementById('taskList');
            if (tasks.length === 0) {
                list.innerHTML = '<li class="empty-state">No tasks yet. Add one above!</li>';
                return;
            }
            list.innerHTML = tasks.map((task, i) => `
                <li class="task-item ${task.done ? 'done' : ''}">
                    <input type="checkbox" class="task-checkbox"
                           ${task.done ? 'checked' : ''}
                           onchange="toggleTask(${i})">
                    <span class="task-text">${escapeHtml(task.text)}</span>
                    <button class="task-delete" onclick="deleteTask(${i})">×</button>
                </li>
            `).join('');
        }

        function addTask() {
            const input = document.getElementById('taskInput');
            const text = input.value.trim();
            if (!text) return;
            tasks.push({ text, done: false });
            input.value = '';
            render();
            save();
        }

        function toggleTask(i) {
            tasks[i].done = !tasks[i].done;
            render();
            save();
        }

        function deleteTask(i) {
            tasks.splice(i, 1);
            render();
            save();
        }

        function save() {
            localStorage.setItem('premium-tasks', JSON.stringify(tasks));
        }

        function load() {
            const saved = localStorage.getItem('premium-tasks');
            if (saved) tasks = JSON.parse(saved);
            render();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        document.getElementById('taskInput').addEventListener('keypress', e => {
            if (e.key === 'Enter') addTask();
        });

        load();
    </script>
</body>
</html>
""",
        "quality": 0.95
    },

    "timer": {
        "name": "Elegant Meditation Timer",
        "dyck": "(((())(()()())))",
        "template": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        :root {
            --bg: #0f0f23;
            --surface: #1a1a2e;
            --primary: #64b5f6;
            --primary-glow: rgba(100, 181, 246, 0.3);
            --text: #e8eaf6;
            --text-muted: #9e9e9e;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Inter', -apple-system, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .container {
            text-align: center;
            padding: 3rem;
        }
        h1 {
            font-size: 1.5rem;
            font-weight: 300;
            letter-spacing: 0.2em;
            margin-bottom: 3rem;
            color: var(--text-muted);
        }
        .timer-display {
            font-size: 6rem;
            font-weight: 200;
            font-family: 'SF Mono', 'Fira Code', monospace;
            margin-bottom: 3rem;
            text-shadow: 0 0 40px var(--primary-glow);
        }
        .timer-ring {
            width: 300px;
            height: 300px;
            margin: 0 auto 3rem;
            position: relative;
        }
        .timer-ring svg {
            transform: rotate(-90deg);
        }
        .timer-ring circle {
            fill: none;
            stroke-width: 4;
        }
        .timer-ring .bg { stroke: var(--surface); }
        .timer-ring .progress {
            stroke: var(--primary);
            stroke-linecap: round;
            transition: stroke-dashoffset 0.5s ease;
        }
        .timer-text {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 3.5rem;
            font-weight: 200;
            font-family: 'SF Mono', monospace;
        }
        .controls {
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2rem;
        }
        button {
            padding: 1rem 2rem;
            background: var(--surface);
            color: var(--text);
            border: 1px solid var(--primary);
            border-radius: 50px;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s;
        }
        button:hover {
            background: var(--primary);
            color: var(--bg);
        }
        button.active { background: var(--primary); color: var(--bg); }
        .presets {
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }
        .preset {
            padding: 0.5rem 1rem;
            background: var(--surface);
            border: none;
            border-radius: 20px;
            color: var(--text-muted);
            font-size: 0.875rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        .preset:hover { color: var(--primary); }
    </style>
</head>
<body>
    <div class="container">
        <h1>MEDITATION TIMER</h1>
        <div class="timer-ring">
            <svg width="300" height="300">
                <circle class="bg" cx="150" cy="150" r="140"/>
                <circle class="progress" cx="150" cy="150" r="140"
                        stroke-dasharray="879.6"
                        stroke-dashoffset="0" id="progress"/>
            </svg>
            <div class="timer-text" id="display">05:00</div>
        </div>
        <div class="controls">
            <button id="startBtn" onclick="toggle()">Start</button>
            <button onclick="reset()">Reset</button>
        </div>
        <div class="presets">
            <button class="preset" onclick="setTime(60)">1 min</button>
            <button class="preset" onclick="setTime(300)">5 min</button>
            <button class="preset" onclick="setTime(600)">10 min</button>
            <button class="preset" onclick="setTime(1200)">20 min</button>
        </div>
    </div>
    <script>
        let duration = 300;
        let remaining = 300;
        let running = false;
        let interval = null;
        const circumference = 2 * Math.PI * 140;

        function format(secs) {
            const m = Math.floor(secs / 60);
            const s = secs % 60;
            return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
        }

        function render() {
            document.getElementById('display').textContent = format(remaining);
            const offset = circumference * (1 - remaining / duration);
            document.getElementById('progress').style.strokeDashoffset = offset;
            document.getElementById('startBtn').textContent = running ? 'Pause' : 'Start';
            document.getElementById('startBtn').classList.toggle('active', running);
        }

        function toggle() {
            running = !running;
            if (running) {
                interval = setInterval(() => {
                    remaining--;
                    if (remaining <= 0) {
                        running = false;
                        clearInterval(interval);
                        playBell();
                    }
                    render();
                }, 1000);
            } else {
                clearInterval(interval);
            }
            render();
        }

        function reset() {
            running = false;
            clearInterval(interval);
            remaining = duration;
            render();
        }

        function setTime(secs) {
            duration = secs;
            remaining = secs;
            running = false;
            clearInterval(interval);
            render();
        }

        function playBell() {
            const ctx = new AudioContext();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.frequency.value = 440;
            osc.type = 'sine';
            gain.gain.setValueAtTime(0.5, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 2);
            osc.start();
            osc.stop(ctx.currentTime + 2);
        }

        render();
    </script>
</body>
</html>
""",
        "quality": 0.92
    },

    "notes": {
        "name": "Minimal Notes App",
        "dyck": "(((())(()()()())))",
        "template": """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{title}}</title>
    <style>
        :root {
            --bg: #fffef5;
            --surface: #fff;
            --text: #2d2a24;
            --text-muted: #8b8680;
            --accent: #e6b800;
            --border: #e8e6de;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Georgia', serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }
        .app {
            display: grid;
            grid-template-columns: 280px 1fr;
            min-height: 100vh;
        }
        .sidebar {
            background: var(--surface);
            border-right: 1px solid var(--border);
            padding: 1.5rem;
            overflow-y: auto;
        }
        .sidebar h1 {
            font-size: 1.25rem;
            font-weight: normal;
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        .note-list { list-style: none; }
        .note-item {
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.2s;
        }
        .note-item:hover { background: var(--bg); }
        .note-item.active { background: #fff8dc; border-left: 3px solid var(--accent); }
        .note-title {
            font-weight: 500;
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .note-preview {
            font-size: 0.875rem;
            color: var(--text-muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .editor {
            padding: 3rem;
            max-width: 700px;
            margin: 0 auto;
        }
        .editor-title {
            font-size: 2rem;
            font-weight: normal;
            border: none;
            outline: none;
            width: 100%;
            margin-bottom: 1.5rem;
            background: transparent;
        }
        .editor-content {
            font-size: 1.125rem;
            line-height: 1.8;
            border: none;
            outline: none;
            width: 100%;
            min-height: 60vh;
            resize: none;
            background: transparent;
        }
        .new-note {
            width: 100%;
            padding: 0.75rem;
            background: var(--accent);
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 1rem;
            cursor: pointer;
            margin-bottom: 1rem;
        }
        .new-note:hover { opacity: 0.9; }
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <h1>Notes</h1>
            <button class="new-note" onclick="newNote()">+ New Note</button>
            <ul class="note-list" id="noteList"></ul>
        </aside>
        <main class="editor">
            <input type="text" class="editor-title" id="title" placeholder="Untitled" oninput="save()">
            <textarea class="editor-content" id="content" placeholder="Start writing..." oninput="save()"></textarea>
        </main>
    </div>
    <script>
        let notes = [];
        let current = 0;

        function load() {
            const saved = localStorage.getItem('premium-notes');
            notes = saved ? JSON.parse(saved) : [{ title: 'Welcome', content: 'Start writing your thoughts...' }];
            render();
            select(0);
        }

        function save() {
            notes[current].title = document.getElementById('title').value || 'Untitled';
            notes[current].content = document.getElementById('content').value;
            localStorage.setItem('premium-notes', JSON.stringify(notes));
            render();
        }

        function render() {
            document.getElementById('noteList').innerHTML = notes.map((n, i) => `
                <li class="note-item ${i === current ? 'active' : ''}" onclick="select(${i})">
                    <div class="note-title">${n.title || 'Untitled'}</div>
                    <div class="note-preview">${(n.content || '').slice(0, 50)}</div>
                </li>
            `).join('');
        }

        function select(i) {
            current = i;
            document.getElementById('title').value = notes[i].title || '';
            document.getElementById('content').value = notes[i].content || '';
            render();
        }

        function newNote() {
            notes.unshift({ title: '', content: '' });
            select(0);
            save();
        }

        load();
    </script>
</body>
</html>
""",
        "quality": 0.90
    }
}


# =============================================================================
# SHAPE SHIFTER (Learning Engine)
# =============================================================================

class ShapeShifter:
    """
    The learning engine that:
    1. Maps NL → Shapes
    2. Learns satisfaction invariants
    3. Improves over time
    """

    def __init__(self, catalog: ShapeCatalog, db_path: str = "shifter.db"):
        self.catalog = catalog
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS mappings (
                id INTEGER PRIMARY KEY,
                nl_hash TEXT,
                nl_text TEXT,
                shape_key TEXT,
                template_name TEXT,
                satisfaction REAL,
                timestamp REAL
            )
        """)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS keywords (
                keyword TEXT PRIMARY KEY,
                best_template TEXT,
                avg_satisfaction REAL,
                count INTEGER DEFAULT 1
            )
        """)
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_nl_hash ON mappings(nl_hash)")
        self.conn.commit()

    def learn(self, nl_text: str, template_name: str, satisfaction: float):
        """Learn from a satisfaction signal."""
        nl_hash = hashlib.sha256(nl_text.encode()).hexdigest()[:16]
        template = self.catalog.get_template(template_name)
        shape_key = template['shape_key'] if template else ""

        self.conn.execute("""
            INSERT INTO mappings (nl_hash, nl_text, shape_key, template_name, satisfaction, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (nl_hash, nl_text, shape_key, template_name, satisfaction, time.time()))

        # Update keyword mappings
        for kw in self._keywords(nl_text):
            row = self.conn.execute("SELECT * FROM keywords WHERE keyword = ?", (kw,)).fetchone()
            if row:
                # Update if this is better
                if satisfaction > row['avg_satisfaction']:
                    self.conn.execute("""
                        UPDATE keywords SET best_template = ?, avg_satisfaction = ?, count = count + 1
                        WHERE keyword = ?
                    """, (template_name, satisfaction, kw))
            else:
                self.conn.execute("""
                    INSERT INTO keywords (keyword, best_template, avg_satisfaction, count)
                    VALUES (?, ?, ?, 1)
                """, (kw, template_name, satisfaction))

        self.conn.commit()

    def suggest(self, nl_text: str) -> List[Tuple[str, float]]:
        """Suggest templates for NL text. Returns [(template_name, confidence)]."""
        keywords = self._keywords(nl_text)
        suggestions = {}

        for kw in keywords:
            row = self.conn.execute("SELECT * FROM keywords WHERE keyword = ?", (kw,)).fetchone()
            if row:
                name = row['best_template']
                sat = row['avg_satisfaction']
                count = row['count']
                # Weight by satisfaction and frequency
                score = sat * (1 + count / 10)
                if name in suggestions:
                    suggestions[name] = max(suggestions[name], score)
                else:
                    suggestions[name] = score

        # Sort by score
        sorted_suggestions = sorted(suggestions.items(), key=lambda x: -x[1])
        return sorted_suggestions[:5]

    def _keywords(self, text: str) -> List[str]:
        stopwords = {'a', 'an', 'the', 'is', 'are', 'i', 'want', 'need', 'to', 'make', 'create', 'build'}
        return [w for w in text.lower().split() if w not in stopwords and len(w) > 2]

    def close(self):
        self.conn.close()


# =============================================================================
# PREMIUM DEMO
# =============================================================================

class PremiumDemo:
    """
    The premium speak-see-experience demo.

    Flow:
    1. User speaks natural language
    2. Shape Shifter suggests high-quality templates
    3. User sees generated software
    4. User experiences and rates
    5. Shifter learns, catalog grows
    """

    def __init__(self, db_prefix: str = "premium"):
        self.catalog = ShapeCatalog(f"{db_prefix}_catalog.db")
        self.shifter = ShapeShifter(self.catalog, f"{db_prefix}_shifter.db")
        self._seed_templates()

    def _seed_templates(self):
        """Seed catalog with premium templates."""
        for name, data in PREMIUM_TEMPLATES.items():
            self.catalog.add_template(name, data['dyck'], data['template'], data['quality'])
            # Register equivalence
            shape = parse_dyck(data['dyck'])
            self.catalog.register_equivalence(shape, data['name'])

    def speak(self, description: str) -> List[dict]:
        """Speak → See. Returns list of generated options."""
        options = []

        # Get suggestions from shifter
        suggestions = self.shifter.suggest(description)

        if suggestions:
            for name, score in suggestions[:3]:
                template = self.catalog.get_template(name)
                if template:
                    html = template['template'].replace('{{title}}', description.title())
                    options.append({
                        'name': name,
                        'template': template['template'],
                        'html': html,
                        'confidence': min(1.0, score),
                        'quality': template['quality_score']
                    })

        # Fall back to keyword matching in templates
        if not options:
            for name, data in PREMIUM_TEMPLATES.items():
                if any(kw in description.lower() for kw in [name, name[:-1]]):
                    html = data['template'].replace('{{title}}', description.title())
                    options.append({
                        'name': name,
                        'template': data['template'],
                        'html': html,
                        'confidence': 0.7,
                        'quality': data['quality']
                    })

        # Default to todo if nothing matches
        if not options:
            data = PREMIUM_TEMPLATES['todo']
            html = data['template'].replace('{{title}}', description.title())
            options.append({
                'name': 'todo',
                'template': data['template'],
                'html': html,
                'confidence': 0.5,
                'quality': data['quality']
            })

        return options

    def experience(self, description: str, template_name: str, satisfaction: float):
        """Experience → Learn."""
        self.shifter.learn(description, template_name, satisfaction)

    def run(self):
        """Run the interactive demo."""
        print("\n" + "=" * 70)
        print("  PREMIUM SPEAK-SEE-EXPERIENCE DEMO")
        print("  High-end software. Insta-delivery. Shape-first learning.")
        print("=" * 70)
        print("""
  Tell me what you want. I'll generate premium software.
  Rate what I give you. I learn YOUR preferences.

  Examples:
    "a calm todo list with elegant design"
    "meditation timer with peaceful sounds"
    "minimal notes app like paper"

  Type 'quit' to exit, 'catalog' to see templates.
""")

        while True:
            try:
                request = input("\n  What software do you want?\n  > ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if not request:
                continue

            if request.lower() in ('quit', 'exit', 'q'):
                break

            if request.lower() == 'catalog':
                print("\n  PREMIUM TEMPLATES:")
                for name, data in PREMIUM_TEMPLATES.items():
                    print(f"    {name}: {data['name']} (quality: {data['quality']:.0%})")
                continue

            # Generate options
            print("\n  Generating premium software...")
            options = self.speak(request)

            print(f"\n  Found {len(options)} option(s):\n")
            for i, opt in enumerate(options, 1):
                print(f"  [{i}] {opt['name']}")
                print(f"      Quality: {opt['quality']:.0%}")
                print(f"      Confidence: {opt['confidence']:.0%}")

            # Choose
            try:
                choice = input("\n  Which one? (1-{}, or 's' to save HTML): ".format(len(options))).strip()
            except (EOFError, KeyboardInterrupt):
                break

            if choice.lower() == 's':
                # Save all to files
                for opt in options:
                    filename = f"generated_{opt['name']}.html"
                    with open(filename, 'w') as f:
                        f.write(opt['html'])
                    print(f"  Saved: {filename}")
                continue

            try:
                idx = int(choice) - 1
                chosen = options[idx]
            except (ValueError, IndexError):
                print("  Invalid choice")
                continue

            # Save HTML
            filename = f"generated_{chosen['name']}.html"
            with open(filename, 'w') as f:
                f.write(chosen['html'])
            print(f"\n  Generated: {filename}")
            print(f"  Open in browser to experience it!")

            # Get satisfaction
            try:
                sat_str = input("  Rate your experience (0-10): ").strip()
                satisfaction = float(sat_str) / 10.0
            except (ValueError, EOFError):
                satisfaction = 0.7

            feedback = input("  Any feedback? (optional): ").strip()

            # Learn
            self.experience(request, chosen['name'], satisfaction)
            print(f"  Recorded: {satisfaction:.0%} satisfaction")

            # Show what was learned
            suggestions = self.shifter.suggest(request)
            if suggestions:
                print(f"  Next time, I'll suggest: {suggestions[0][0]} first")

        print("\n  Thank you for training me!")
        self.close()

    def close(self):
        self.catalog.close()
        self.shifter.close()


def main():
    demo = PremiumDemo()
    demo.run()


if __name__ == "__main__":
    main()
