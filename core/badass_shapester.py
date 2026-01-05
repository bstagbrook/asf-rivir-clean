#!/usr/bin/env python3
"""
Badass Shapester - Premium Software Generation Through Rich Description

The key insight: MORE DETAIL = BETTER SOFTWARE

This shapester:
1. Extracts rich features from detailed descriptions
2. Learns which features correlate with satisfaction
3. Generates increasingly badass software over time
4. Builds a catalog of premium patterns

The description IS the software. A detailed description creates detailed software.
"""

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set


# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

@dataclass
class DescriptionFeatures:
    """Rich features extracted from a description."""
    # Core intent
    intent: str = ""  # create, build, make, etc.
    target: str = ""  # what kind of thing

    # Style
    aesthetic: List[str] = field(default_factory=list)  # calm, elegant, bold, etc.
    mood: List[str] = field(default_factory=list)  # peaceful, energetic, focused

    # UX
    interactions: List[str] = field(default_factory=list)  # click, drag, swipe
    animations: List[str] = field(default_factory=list)  # smooth, bounce, fade

    # Visual
    colors: List[str] = field(default_factory=list)  # lavender, dark, warm
    typography: List[str] = field(default_factory=list)  # serif, modern, elegant
    layout: List[str] = field(default_factory=list)  # minimal, grid, card

    # Functional
    features: List[str] = field(default_factory=list)  # save, export, search
    data: List[str] = field(default_factory=list)  # tasks, notes, items

    # Quality signals
    quality_words: List[str] = field(default_factory=list)  # premium, polished, refined
    detail_level: int = 0  # How detailed the description is

    def to_dict(self) -> dict:
        return {
            'intent': self.intent,
            'target': self.target,
            'aesthetic': self.aesthetic,
            'mood': self.mood,
            'interactions': self.interactions,
            'animations': self.animations,
            'colors': self.colors,
            'typography': self.typography,
            'layout': self.layout,
            'features': self.features,
            'data': self.data,
            'quality_words': self.quality_words,
            'detail_level': self.detail_level
        }


class FeatureExtractor:
    """Extract rich features from natural language descriptions."""

    # Vocabulary for feature extraction
    INTENTS = ['create', 'build', 'make', 'design', 'craft', 'generate', 'develop']

    TARGETS = {
        'todo': ['todo', 'task', 'checklist', 'list'],
        'notes': ['note', 'journal', 'diary', 'writing', 'document'],
        'timer': ['timer', 'clock', 'countdown', 'pomodoro', 'meditation'],
        'calculator': ['calculator', 'math', 'compute', 'calc'],
        'dashboard': ['dashboard', 'panel', 'control', 'monitor'],
        'chat': ['chat', 'message', 'conversation', 'messenger'],
        'gallery': ['gallery', 'photo', 'image', 'portfolio'],
        'form': ['form', 'input', 'survey', 'questionnaire'],
        'calendar': ['calendar', 'schedule', 'planner', 'agenda'],
        'player': ['player', 'music', 'audio', 'video', 'media'],
    }

    AESTHETICS = {
        'minimal': ['minimal', 'minimalist', 'simple', 'clean', 'bare'],
        'elegant': ['elegant', 'refined', 'sophisticated', 'graceful'],
        'bold': ['bold', 'striking', 'dramatic', 'powerful'],
        'soft': ['soft', 'gentle', 'subtle', 'quiet'],
        'modern': ['modern', 'contemporary', 'sleek', 'fresh'],
        'classic': ['classic', 'traditional', 'timeless', 'vintage'],
        'playful': ['playful', 'fun', 'whimsical', 'cheerful'],
        'professional': ['professional', 'corporate', 'business', 'serious'],
    }

    MOODS = {
        'calm': ['calm', 'peaceful', 'serene', 'tranquil', 'zen'],
        'energetic': ['energetic', 'vibrant', 'lively', 'dynamic'],
        'focused': ['focused', 'concentrated', 'productive', 'efficient'],
        'creative': ['creative', 'artistic', 'expressive', 'inspired'],
        'cozy': ['cozy', 'warm', 'comfortable', 'inviting'],
        'sacred': ['sacred', 'spiritual', 'mindful', 'meditative'],
    }

    COLORS = {
        'dark': ['dark', 'black', 'night', 'shadow'],
        'light': ['light', 'white', 'bright', 'airy'],
        'warm': ['warm', 'golden', 'amber', 'orange', 'yellow', 'sunset'],
        'cool': ['cool', 'blue', 'teal', 'cyan', 'ocean'],
        'earth': ['earth', 'brown', 'beige', 'natural', 'organic'],
        'purple': ['purple', 'violet', 'lavender', 'indigo'],
        'green': ['green', 'forest', 'mint', 'sage', 'emerald'],
        'pink': ['pink', 'rose', 'coral', 'blush'],
    }

    ANIMATIONS = {
        'smooth': ['smooth', 'fluid', 'seamless', 'flowing'],
        'bounce': ['bounce', 'spring', 'elastic'],
        'fade': ['fade', 'dissolve', 'transition'],
        'slide': ['slide', 'glide', 'sweep'],
        'grow': ['grow', 'expand', 'scale', 'zoom'],
        'none': ['static', 'instant', 'no animation'],
    }

    LAYOUTS = {
        'grid': ['grid', 'tile', 'mosaic', 'card'],
        'list': ['list', 'stack', 'column', 'vertical'],
        'split': ['split', 'sidebar', 'panel', 'dual'],
        'centered': ['centered', 'focused', 'spotlight'],
        'fullscreen': ['fullscreen', 'immersive', 'full'],
    }

    QUALITY_WORDS = [
        'premium', 'polished', 'refined', 'beautiful', 'gorgeous',
        'stunning', 'exquisite', 'luxury', 'high-end', 'professional',
        'production-ready', 'pixel-perfect', 'crisp', 'sharp', 'detailed',
        'thoughtful', 'crafted', 'artisan', 'bespoke', 'custom'
    ]

    def extract(self, description: str) -> DescriptionFeatures:
        """Extract all features from a description."""
        desc = description.lower()
        words = set(desc.split())

        features = DescriptionFeatures()

        # Intent
        for intent in self.INTENTS:
            if intent in desc:
                features.intent = intent
                break

        # Target
        for target, keywords in self.TARGETS.items():
            if any(kw in desc for kw in keywords):
                features.target = target
                break

        # Aesthetic
        for aesthetic, keywords in self.AESTHETICS.items():
            if any(kw in desc for kw in keywords):
                features.aesthetic.append(aesthetic)

        # Mood
        for mood, keywords in self.MOODS.items():
            if any(kw in desc for kw in keywords):
                features.mood.append(mood)

        # Colors
        for color, keywords in self.COLORS.items():
            if any(kw in desc for kw in keywords):
                features.colors.append(color)

        # Animations
        for anim, keywords in self.ANIMATIONS.items():
            if any(kw in desc for kw in keywords):
                features.animations.append(anim)

        # Layout
        for layout, keywords in self.LAYOUTS.items():
            if any(kw in desc for kw in keywords):
                features.layout.append(layout)

        # Quality words
        for qw in self.QUALITY_WORDS:
            if qw in desc:
                features.quality_words.append(qw)

        # Detail level (more words and adjectives = more detail)
        features.detail_level = min(10, len(description.split()) // 5)

        return features


# =============================================================================
# TEMPLATE GENERATION
# =============================================================================

class TemplateBuilder:
    """Build HTML templates from features."""

    # Color schemes
    COLOR_SCHEMES = {
        'dark': {
            'bg': '#0f0f23', 'surface': '#1a1a2e', 'primary': '#64b5f6',
            'text': '#e8eaf6', 'muted': '#9e9e9e', 'border': '#2d2d44'
        },
        'light': {
            'bg': '#fafafa', 'surface': '#ffffff', 'primary': '#6366f1',
            'text': '#1f2937', 'muted': '#6b7280', 'border': '#e5e7eb'
        },
        'warm': {
            'bg': '#fffef5', 'surface': '#fff', 'primary': '#e6b800',
            'text': '#2d2a24', 'muted': '#8b8680', 'border': '#e8e6de'
        },
        'cool': {
            'bg': '#f0f9ff', 'surface': '#ffffff', 'primary': '#0ea5e9',
            'text': '#0c4a6e', 'muted': '#64748b', 'border': '#e0f2fe'
        },
        'purple': {
            'bg': '#faf5ff', 'surface': '#ffffff', 'primary': '#8b5cf6',
            'text': '#4c1d95', 'muted': '#7c3aed', 'border': '#e9d5ff'
        },
        'earth': {
            'bg': '#faf8f5', 'surface': '#ffffff', 'primary': '#a3826d',
            'text': '#44403c', 'muted': '#78716c', 'border': '#e7e5e4'
        },
    }

    # Font stacks
    FONTS = {
        'modern': "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
        'elegant': "'Georgia', 'Playfair Display', serif",
        'mono': "'SF Mono', 'Fira Code', 'Consolas', monospace",
        'playful': "'Nunito', 'Comic Sans MS', cursive",
    }

    # Animation timings
    ANIMATIONS = {
        'smooth': '0.3s ease',
        'bounce': '0.5s cubic-bezier(0.68, -0.55, 0.265, 1.55)',
        'fade': '0.2s ease-in-out',
        'none': '0s',
    }

    def build(self, features: DescriptionFeatures, title: str) -> str:
        """Build an HTML template from features."""
        # Select color scheme
        scheme_name = features.colors[0] if features.colors else 'light'
        if scheme_name not in self.COLOR_SCHEMES:
            scheme_name = 'light'
        colors = self.COLOR_SCHEMES[scheme_name]

        # Select font
        font_style = 'elegant' if 'elegant' in features.aesthetic else 'modern'
        font = self.FONTS[font_style]

        # Select animation
        anim_style = features.animations[0] if features.animations else 'smooth'
        if anim_style not in self.ANIMATIONS:
            anim_style = 'smooth'
        animation = self.ANIMATIONS[anim_style]

        # Build based on target
        builder_method = getattr(self, f'_build_{features.target}', self._build_generic)
        return builder_method(title, colors, font, animation, features)

    def _build_todo(self, title: str, colors: dict, font: str, animation: str, features: DescriptionFeatures) -> str:
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: {colors['bg']};
            --surface: {colors['surface']};
            --primary: {colors['primary']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --border: {colors['border']};
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --radius: {'16px' if 'soft' in features.aesthetic else '8px'};
            --transition: {animation};
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: {font};
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: flex-start;
            justify-content: center;
            padding: {'3rem' if 'minimal' in features.aesthetic else '2rem'};
        }}
        .container {{
            width: 100%;
            max-width: {'500px' if 'minimal' in features.aesthetic else '600px'};
            background: var(--surface);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: {'3rem' if 'elegant' in features.aesthetic else '2rem'};
            {'border: 1px solid var(--border);' if 'soft' in features.aesthetic else ''}
        }}
        h1 {{
            font-size: {'1.5rem' if 'minimal' in features.aesthetic else '1.75rem'};
            font-weight: {'300' if 'elegant' in features.aesthetic else '600'};
            margin-bottom: {'2rem' if 'elegant' in features.aesthetic else '1.5rem'};
            color: var(--primary);
            {'letter-spacing: 0.1em;' if 'elegant' in features.aesthetic else ''}
        }}
        .input-group {{
            display: flex;
            gap: 0.75rem;
            margin-bottom: 1.5rem;
        }}
        input[type="text"] {{
            flex: 1;
            padding: {'1rem' if 'elegant' in features.aesthetic else '0.75rem'} 1rem;
            border: 2px solid var(--border);
            border-radius: calc(var(--radius) / 2);
            font-size: 1rem;
            font-family: inherit;
            background: var(--bg);
            color: var(--text);
            transition: border-color var(--transition);
        }}
        input[type="text"]:focus {{
            outline: none;
            border-color: var(--primary);
        }}
        button {{
            padding: {'1rem 2rem' if 'elegant' in features.aesthetic else '0.75rem 1.5rem'};
            background: var(--primary);
            color: {'var(--bg)' if colors['bg'].startswith('#f') else 'white'};
            border: none;
            border-radius: {'50px' if 'soft' in features.aesthetic else 'calc(var(--radius) / 2)'};
            font-size: 1rem;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition);
            font-family: inherit;
        }}
        button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
        .task-list {{ list-style: none; }}
        .task-item {{
            display: flex;
            align-items: center;
            padding: {'1.25rem' if 'elegant' in features.aesthetic else '1rem'};
            border-bottom: 1px solid var(--border);
            transition: background var(--transition);
        }}
        .task-item:hover {{ background: var(--bg); }}
        .task-item.done .task-text {{
            text-decoration: line-through;
            color: var(--muted);
        }}
        .task-checkbox {{
            width: {'24px' if 'elegant' in features.aesthetic else '20px'};
            height: {'24px' if 'elegant' in features.aesthetic else '20px'};
            margin-right: 1rem;
            accent-color: var(--primary);
        }}
        .task-text {{ flex: 1; font-size: 1rem; }}
        .task-delete {{
            opacity: 0;
            background: none;
            border: none;
            color: #ef4444;
            cursor: pointer;
            padding: 0.5rem;
            font-size: 1.25rem;
            transition: opacity var(--transition);
        }}
        .task-item:hover .task-delete {{ opacity: 1; }}
        .empty-state {{
            text-align: center;
            padding: 3rem;
            color: var(--muted);
            {'font-style: italic;' if 'elegant' in features.aesthetic else ''}
        }}
        {'@media (prefers-color-scheme: dark) { :root { --bg: #0f0f23; --surface: #1a1a2e; --text: #e8eaf6; --border: #2d2d44; } }' if 'calm' in features.mood else ''}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="input-group">
            <input type="text" id="taskInput" placeholder="{'What would you like to accomplish?' if 'elegant' in features.aesthetic else 'What needs to be done?'}" autofocus>
            <button onclick="addTask()">{'Add Task' if 'professional' in features.aesthetic else 'Add'}</button>
        </div>
        <ul class="task-list" id="taskList">
            <li class="empty-state">{'Begin your journey...' if 'sacred' in features.mood else 'No tasks yet. Add one above!'}</li>
        </ul>
    </div>
    <script>
        let tasks = JSON.parse(localStorage.getItem('{title.replace(" ", "_").lower()}_tasks') || '[]');

        function render() {{
            const list = document.getElementById('taskList');
            if (tasks.length === 0) {{
                list.innerHTML = '<li class="empty-state">{"Begin your journey..." if "sacred" in features.mood else "No tasks yet. Add one above!"}</li>';
                return;
            }}
            list.innerHTML = tasks.map((task, i) => `
                <li class="task-item ${{task.done ? 'done' : ''}}">
                    <input type="checkbox" class="task-checkbox"
                           ${{task.done ? 'checked' : ''}}
                           onchange="toggleTask(${{i}})">
                    <span class="task-text">${{escapeHtml(task.text)}}</span>
                    <button class="task-delete" onclick="deleteTask(${{i}})">×</button>
                </li>
            `).join('');
        }}

        function addTask() {{
            const input = document.getElementById('taskInput');
            const text = input.value.trim();
            if (!text) return;
            tasks.push({{ text, done: false, created: Date.now() }});
            input.value = '';
            render();
            save();
        }}

        function toggleTask(i) {{
            tasks[i].done = !tasks[i].done;
            render();
            save();
        }}

        function deleteTask(i) {{
            tasks.splice(i, 1);
            render();
            save();
        }}

        function save() {{
            localStorage.setItem('{title.replace(" ", "_").lower()}_tasks', JSON.stringify(tasks));
        }}

        function escapeHtml(text) {{
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }}

        document.getElementById('taskInput').addEventListener('keypress', e => {{
            if (e.key === 'Enter') addTask();
        }});

        render();
    </script>
</body>
</html>'''

    def _build_timer(self, title: str, colors: dict, font: str, animation: str, features: DescriptionFeatures) -> str:
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: {colors['bg']};
            --surface: {colors['surface']};
            --primary: {colors['primary']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --glow: {'rgba(100, 181, 246, 0.3)' if 'dark' in features.colors else 'rgba(99, 102, 241, 0.2)'};
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: {font};
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .container {{
            text-align: center;
            padding: 3rem;
        }}
        h1 {{
            font-size: {'1.25rem' if 'minimal' in features.aesthetic else '1.5rem'};
            font-weight: {'200' if 'elegant' in features.aesthetic else '300'};
            letter-spacing: {'0.3em' if 'elegant' in features.aesthetic else '0.2em'};
            margin-bottom: 3rem;
            color: var(--muted);
            text-transform: uppercase;
        }}
        .timer-ring {{
            width: {'280px' if 'minimal' in features.aesthetic else '320px'};
            height: {'280px' if 'minimal' in features.aesthetic else '320px'};
            margin: 0 auto 3rem;
            position: relative;
        }}
        .timer-ring svg {{ transform: rotate(-90deg); }}
        .timer-ring circle {{ fill: none; stroke-width: {'3' if 'elegant' in features.aesthetic else '4'}; }}
        .timer-ring .bg {{ stroke: var(--surface); }}
        .timer-ring .progress {{
            stroke: var(--primary);
            stroke-linecap: round;
            transition: stroke-dashoffset {animation};
            {'filter: drop-shadow(0 0 10px var(--glow));' if 'calm' in features.mood else ''}
        }}
        .timer-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: {'4rem' if 'minimal' in features.aesthetic else '3.5rem'};
            font-weight: {'100' if 'elegant' in features.aesthetic else '200'};
            font-family: 'SF Mono', monospace;
            {'text-shadow: 0 0 30px var(--glow);' if 'calm' in features.mood else ''}
        }}
        .controls {{
            display: flex;
            gap: 1rem;
            justify-content: center;
            margin-bottom: 2rem;
        }}
        button {{
            padding: 1rem 2rem;
            background: {'transparent' if 'minimal' in features.aesthetic else 'var(--surface)'};
            color: var(--text);
            border: {'2px' if 'elegant' in features.aesthetic else '1px'} solid var(--primary);
            border-radius: 50px;
            font-size: 1rem;
            cursor: pointer;
            transition: all {animation};
            font-family: inherit;
        }}
        button:hover {{ background: var(--primary); color: var(--bg); }}
        button.active {{ background: var(--primary); color: var(--bg); }}
        .presets {{
            display: flex;
            gap: 0.5rem;
            justify-content: center;
            flex-wrap: wrap;
        }}
        .preset {{
            padding: 0.5rem 1rem;
            background: {'transparent' if 'minimal' in features.aesthetic else 'var(--surface)'};
            border: none;
            border-radius: 20px;
            color: var(--muted);
            font-size: 0.875rem;
            cursor: pointer;
            transition: color {animation};
        }}
        .preset:hover {{ color: var(--primary); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{'Meditation' if 'calm' in features.mood or 'sacred' in features.mood else title}</h1>
        <div class="timer-ring">
            <svg width="{'280' if 'minimal' in features.aesthetic else '320'}" height="{'280' if 'minimal' in features.aesthetic else '320'}">
                <circle class="bg" cx="{'140' if 'minimal' in features.aesthetic else '160'}" cy="{'140' if 'minimal' in features.aesthetic else '160'}" r="{'130' if 'minimal' in features.aesthetic else '150'}"/>
                <circle class="progress" cx="{'140' if 'minimal' in features.aesthetic else '160'}" cy="{'140' if 'minimal' in features.aesthetic else '160'}" r="{'130' if 'minimal' in features.aesthetic else '150'}"
                        stroke-dasharray="{'816.8' if 'minimal' in features.aesthetic else '942.5'}"
                        stroke-dashoffset="0" id="progress"/>
            </svg>
            <div class="timer-text" id="display">05:00</div>
        </div>
        <div class="controls">
            <button id="startBtn" onclick="toggle()">{'Begin' if 'sacred' in features.mood else 'Start'}</button>
            <button onclick="reset()">Reset</button>
        </div>
        <div class="presets">
            <button class="preset" onclick="setTime(60)">1 min</button>
            <button class="preset" onclick="setTime(300)">5 min</button>
            <button class="preset" onclick="setTime(600)">10 min</button>
            <button class="preset" onclick="setTime(1200)">20 min</button>
            {'<button class="preset" onclick="setTime(1800)">30 min</button>' if 'calm' in features.mood else ''}
        </div>
    </div>
    <script>
        const radius = {'130' if 'minimal' in features.aesthetic else '150'};
        const circumference = 2 * Math.PI * radius;
        let duration = 300, remaining = 300, running = false, interval = null;

        function format(secs) {{
            const m = Math.floor(secs / 60);
            const s = secs % 60;
            return `${{String(m).padStart(2, '0')}}:${{String(s).padStart(2, '0')}}`;
        }}

        function render() {{
            document.getElementById('display').textContent = format(remaining);
            const offset = circumference * (1 - remaining / duration);
            document.getElementById('progress').style.strokeDashoffset = offset;
            document.getElementById('startBtn').textContent = running ? 'Pause' : '{'Begin' if 'sacred' in features.mood else 'Start'}';
            document.getElementById('startBtn').classList.toggle('active', running);
        }}

        function toggle() {{
            running = !running;
            if (running) {{
                interval = setInterval(() => {{
                    remaining--;
                    if (remaining <= 0) {{
                        running = false;
                        clearInterval(interval);
                        playBell();
                    }}
                    render();
                }}, 1000);
            }} else {{
                clearInterval(interval);
            }}
            render();
        }}

        function reset() {{
            running = false;
            clearInterval(interval);
            remaining = duration;
            render();
        }}

        function setTime(secs) {{
            duration = secs;
            remaining = secs;
            running = false;
            clearInterval(interval);
            render();
        }}

        function playBell() {{
            const ctx = new AudioContext();
            const osc = ctx.createOscillator();
            const gain = ctx.createGain();
            osc.connect(gain);
            gain.connect(ctx.destination);
            osc.frequency.value = {'396' if 'sacred' in features.mood else '440'};
            osc.type = 'sine';
            gain.gain.setValueAtTime(0.3, ctx.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 3);
            osc.start();
            osc.stop(ctx.currentTime + 3);
        }}

        render();
    </script>
</body>
</html>'''

    def _build_notes(self, title: str, colors: dict, font: str, animation: str, features: DescriptionFeatures) -> str:
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: {colors['bg']};
            --surface: {colors['surface']};
            --primary: {colors['primary']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --border: {colors['border']};
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: {font};
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
        }}
        .app {{
            display: grid;
            grid-template-columns: {'240px' if 'minimal' in features.aesthetic else '280px'} 1fr;
            min-height: 100vh;
        }}
        .sidebar {{
            background: var(--surface);
            border-right: 1px solid var(--border);
            padding: {'1rem' if 'minimal' in features.aesthetic else '1.5rem'};
            overflow-y: auto;
        }}
        .sidebar h1 {{
            font-size: {'1rem' if 'minimal' in features.aesthetic else '1.25rem'};
            font-weight: {'300' if 'elegant' in features.aesthetic else 'normal'};
            margin-bottom: 1.5rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
            {'letter-spacing: 0.05em;' if 'elegant' in features.aesthetic else ''}
        }}
        .note-list {{ list-style: none; }}
        .note-item {{
            padding: 0.75rem;
            margin-bottom: 0.5rem;
            border-radius: {'8px' if 'soft' in features.aesthetic else '6px'};
            cursor: pointer;
            transition: background {animation};
        }}
        .note-item:hover {{ background: var(--bg); }}
        .note-item.active {{
            background: {'#fff8dc' if colors['bg'].startswith('#fff') else '#2d2d44'};
            border-left: 3px solid var(--primary);
        }}
        .note-title {{
            font-weight: 500;
            margin-bottom: 0.25rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .note-preview {{
            font-size: 0.875rem;
            color: var(--muted);
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        .editor {{
            padding: {'4rem' if 'elegant' in features.aesthetic else '3rem'};
            max-width: {'600px' if 'minimal' in features.aesthetic else '700px'};
            margin: 0 auto;
        }}
        .editor-title {{
            font-size: {'2.5rem' if 'elegant' in features.aesthetic else '2rem'};
            font-weight: {'200' if 'elegant' in features.aesthetic else 'normal'};
            border: none;
            outline: none;
            width: 100%;
            margin-bottom: {'2rem' if 'elegant' in features.aesthetic else '1.5rem'};
            background: transparent;
            color: var(--text);
            font-family: inherit;
        }}
        .editor-content {{
            font-size: {'1.25rem' if 'elegant' in features.aesthetic else '1.125rem'};
            line-height: {'2' if 'elegant' in features.aesthetic else '1.8'};
            border: none;
            outline: none;
            width: 100%;
            min-height: 60vh;
            resize: none;
            background: transparent;
            color: var(--text);
            font-family: inherit;
        }}
        .new-note {{
            width: 100%;
            padding: 0.75rem;
            background: var(--primary);
            color: {'var(--bg)' if colors['bg'].startswith('#f') else 'white'};
            border: none;
            border-radius: {'8px' if 'soft' in features.aesthetic else '6px'};
            font-size: 1rem;
            cursor: pointer;
            margin-bottom: 1rem;
            font-family: inherit;
            transition: opacity {animation};
        }}
        .new-note:hover {{ opacity: 0.9; }}
    </style>
</head>
<body>
    <div class="app">
        <aside class="sidebar">
            <h1>{'Notes' if 'minimal' in features.aesthetic else title}</h1>
            <button class="new-note" onclick="newNote()">+ {'New' if 'minimal' in features.aesthetic else 'New Note'}</button>
            <ul class="note-list" id="noteList"></ul>
        </aside>
        <main class="editor">
            <input type="text" class="editor-title" id="title" placeholder="{'Title' if 'minimal' in features.aesthetic else 'Untitled'}" oninput="save()">
            <textarea class="editor-content" id="content" placeholder="{'Write...' if 'minimal' in features.aesthetic else 'Start writing your thoughts...'}" oninput="save()"></textarea>
        </main>
    </div>
    <script>
        let notes = [];
        let current = 0;

        function load() {{
            const saved = localStorage.getItem('{title.replace(" ", "_").lower()}_notes');
            notes = saved ? JSON.parse(saved) : [{{ title: 'Welcome', content: '{'Begin your journey here...' if 'sacred' in features.mood else 'Start writing your thoughts...'}' }}];
            render();
            select(0);
        }}

        function save() {{
            notes[current].title = document.getElementById('title').value || 'Untitled';
            notes[current].content = document.getElementById('content').value;
            notes[current].updated = Date.now();
            localStorage.setItem('{title.replace(" ", "_").lower()}_notes', JSON.stringify(notes));
            render();
        }}

        function render() {{
            document.getElementById('noteList').innerHTML = notes.map((n, i) => `
                <li class="note-item ${{i === current ? 'active' : ''}}" onclick="select(${{i}})">
                    <div class="note-title">${{n.title || 'Untitled'}}</div>
                    <div class="note-preview">${{(n.content || '').slice(0, 50)}}</div>
                </li>
            `).join('');
        }}

        function select(i) {{
            current = i;
            document.getElementById('title').value = notes[i].title || '';
            document.getElementById('content').value = notes[i].content || '';
            render();
        }}

        function newNote() {{
            notes.unshift({{ title: '', content: '', created: Date.now() }});
            select(0);
            save();
        }}

        load();
    </script>
</body>
</html>'''

    def _build_generic(self, title: str, colors: dict, font: str, animation: str, features: DescriptionFeatures) -> str:
        """Generic template for unknown targets."""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --bg: {colors['bg']};
            --surface: {colors['surface']};
            --primary: {colors['primary']};
            --text: {colors['text']};
            --muted: {colors['muted']};
            --border: {colors['border']};
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: {font};
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 2rem;
        }}
        .container {{
            width: 100%;
            max-width: 600px;
            background: var(--surface);
            border-radius: 16px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            padding: 3rem;
            text-align: center;
        }}
        h1 {{
            font-size: 2rem;
            font-weight: 300;
            margin-bottom: 1rem;
            color: var(--primary);
        }}
        p {{
            color: var(--muted);
            margin-bottom: 2rem;
        }}
        .features {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .feature {{
            padding: 1.5rem 1rem;
            background: var(--bg);
            border-radius: 12px;
            font-size: 0.875rem;
        }}
        button {{
            padding: 1rem 2rem;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1rem;
            cursor: pointer;
            transition: all {animation};
        }}
        button:hover {{ opacity: 0.9; transform: translateY(-1px); }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <p>{'A thoughtfully crafted experience' if 'elegant' in features.aesthetic else 'Your custom application'}</p>
        <div class="features">
            {''.join(f'<div class="feature">{f.title()}</div>' for f in (features.aesthetic + features.mood)[:4]) or '<div class="feature">Custom</div>'}
        </div>
        <button onclick="alert('Coming soon!')">Get Started</button>
    </div>
</body>
</html>'''


# =============================================================================
# SHAPESTER
# =============================================================================

class BadassShapester:
    """
    The learning engine that gets really good at making badass software.
    """

    def __init__(self, db_path: str = "shapester.db"):
        self.extractor = FeatureExtractor()
        self.builder = TemplateBuilder()
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY,
                description TEXT,
                features TEXT,
                html_hash TEXT,
                satisfaction REAL,
                feedback TEXT,
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS feature_scores (
                feature TEXT PRIMARY KEY,
                total_satisfaction REAL DEFAULT 0,
                count INTEGER DEFAULT 0
            );

            CREATE INDEX IF NOT EXISTS idx_gen_satisfaction ON generations(satisfaction DESC);
        """)
        self.conn.commit()

    def generate(self, description: str) -> Tuple[str, DescriptionFeatures]:
        """Generate badass software from description."""
        features = self.extractor.extract(description)

        # Boost features that have high satisfaction scores
        self._boost_features(features)

        # Build the HTML
        title = description.split(',')[0].strip().title()
        html = self.builder.build(features, title)

        return html, features

    def _boost_features(self, features: DescriptionFeatures):
        """Boost features based on learned satisfaction scores."""
        all_features = (
            features.aesthetic + features.mood + features.colors +
            features.animations + features.layout
        )

        for f in all_features:
            row = self.conn.execute(
                "SELECT total_satisfaction, count FROM feature_scores WHERE feature = ?",
                (f,)
            ).fetchone()
            if row and row['count'] > 0:
                avg = row['total_satisfaction'] / row['count']
                # If this feature has low satisfaction, consider removing it
                # (In a more sophisticated system, we'd suggest alternatives)

    def learn(self, description: str, features: DescriptionFeatures, satisfaction: float, feedback: str = ""):
        """Learn from feedback."""
        html_hash = hashlib.sha256(description.encode()).hexdigest()[:16]

        self.conn.execute("""
            INSERT INTO generations (description, features, html_hash, satisfaction, feedback, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (description, json.dumps(features.to_dict()), html_hash, satisfaction, feedback, time.time()))

        # Update feature scores
        all_features = (
            features.aesthetic + features.mood + features.colors +
            features.animations + features.layout
        )

        for f in all_features:
            self.conn.execute("""
                INSERT INTO feature_scores (feature, total_satisfaction, count)
                VALUES (?, ?, 1)
                ON CONFLICT(feature) DO UPDATE SET
                    total_satisfaction = total_satisfaction + ?,
                    count = count + 1
            """, (f, satisfaction, satisfaction))

        self.conn.commit()

    def get_best_features(self, limit: int = 10) -> List[Tuple[str, float]]:
        """Get features with highest average satisfaction."""
        rows = self.conn.execute("""
            SELECT feature, total_satisfaction / count as avg
            FROM feature_scores
            WHERE count >= 2
            ORDER BY avg DESC
            LIMIT ?
        """, (limit,)).fetchall()
        return [(r['feature'], r['avg']) for r in rows]

    def suggest_improvements(self, description: str) -> str:
        """Suggest improvements to a description based on learned preferences."""
        best = self.get_best_features(5)
        if not best:
            return description

        suggestions = []
        for feature, score in best:
            if feature not in description.lower():
                suggestions.append(feature)

        if suggestions:
            return f"{description} with {', '.join(suggestions[:3])} aesthetic"
        return description

    def close(self):
        self.conn.close()


# =============================================================================
# CLI
# =============================================================================

def main():
    """Interactive demo of the badass shapester."""
    shapester = BadassShapester(":memory:")

    print("\n" + "=" * 70)
    print("  BADASS SHAPESTER")
    print("  The more you describe, the better it gets.")
    print("=" * 70)
    print("""
  Tips for BADASS software:

  BAD:  "make a todo app"
  GOOD: "create a calm, elegant todo list with soft lavender colors"
  BEST: "craft a mindful task tracker with warm earth tones, gentle
        animations, elegant serif typography, and a sacred, meditative
        mood that inspires focused productivity"

  The description IS the software. More detail = better results.

  Type 'quit' to exit, 'best' to see top features.
""")

    while True:
        try:
            description = input("\n  Describe your software:\n  > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not description:
            continue

        if description.lower() in ('quit', 'exit', 'q'):
            break

        if description.lower() == 'best':
            best = shapester.get_best_features()
            if best:
                print("\n  Top features by satisfaction:")
                for f, score in best:
                    print(f"    {f}: {score:.0%}")
            else:
                print("  No data yet. Keep training!")
            continue

        # Suggest improvements
        improved = shapester.suggest_improvements(description)
        if improved != description:
            print(f"\n  Suggestion: {improved}")
            use_improved = input("  Use improved? (y/n): ").strip().lower()
            if use_improved == 'y':
                description = improved

        # Generate
        print("\n  Generating badass software...")
        html, features = shapester.generate(description)

        # Show features
        print(f"\n  Detected features:")
        print(f"    Target:     {features.target or 'custom'}")
        print(f"    Aesthetic:  {', '.join(features.aesthetic) or 'default'}")
        print(f"    Mood:       {', '.join(features.mood) or 'neutral'}")
        print(f"    Colors:     {', '.join(features.colors) or 'auto'}")
        print(f"    Detail:     {'★' * features.detail_level}{'☆' * (10-features.detail_level)}")

        # Save
        filename = f"badass_{features.target or 'app'}.html"
        with open(filename, 'w') as f:
            f.write(html)
        print(f"\n  Generated: {filename}")
        print(f"  Open in browser to experience it!")

        # Get satisfaction
        try:
            sat_str = input("\n  Rate it (0-10): ").strip()
            satisfaction = float(sat_str) / 10.0
        except (ValueError, EOFError):
            satisfaction = 0.7

        feedback = input("  Feedback (optional): ").strip()

        # Learn
        shapester.learn(description, features, satisfaction, feedback)
        print(f"  Learned! Satisfaction: {satisfaction:.0%}")

        # Show improvement
        best = shapester.get_best_features(3)
        if best:
            print(f"\n  Your top features: {', '.join(f[0] for f in best)}")

    print("\n  Thanks for training the shapester!")
    shapester.close()


if __name__ == "__main__":
    main()
