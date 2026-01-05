#!/usr/bin/env python3
"""
RIVIR Expert Training & Silent Watcher

Two powerful capabilities:
1. Expert Training - Tell RIVIR to become an expert at something
2. Silent Watcher - Let RIVIR observe everything you do to learn

Usage:
    # Become an expert
    rivir.become_expert("meditation app design")
    rivir.become_expert("python optimization", depth="deep")
    
    # Start silent watching
    watcher = RIVIRWatcher()
    watcher.start_watching()  # Watches keystrokes, apps, files
    watcher.stop_watching()
"""

import os
import time
import json
import sqlite3
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import hashlib

try:
    from rivir_direct import RIVIRDirect
    from receipts import create_transition_receipt
except ImportError:
    from .rivir_direct import RIVIRDirect
    from .receipts import create_transition_receipt


# =============================================================================
# EXPERT TRAINING SYSTEM
# =============================================================================

@dataclass
class ExpertiseArea:
    """Defines an area of expertise to develop."""
    domain: str
    depth: str = "moderate"  # surface, moderate, deep, master
    focus_areas: List[str] = field(default_factory=list)
    learning_sources: List[str] = field(default_factory=list)
    practice_tasks: List[str] = field(default_factory=list)
    mastery_indicators: List[str] = field(default_factory=list)
    current_level: float = 0.0  # 0.0 to 1.0
    started: float = field(default_factory=time.time)
    last_practice: float = 0.0


class ExpertTrainer:
    """Trains RIVIR to become an expert in specific domains."""
    
    def __init__(self, rivir: RIVIRDirect, db_path: str = "expert_training.db"):
        self.rivir = rivir
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.active_expertise = {}
        
    def _init_db(self):
        """Initialize expertise tracking database."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS expertise_areas (
                id INTEGER PRIMARY KEY,
                domain TEXT UNIQUE,
                depth TEXT,
                focus_areas TEXT,
                learning_sources TEXT,
                practice_tasks TEXT,
                mastery_indicators TEXT,
                current_level REAL,
                started REAL,
                last_practice REAL
            );
            
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                timestamp REAL,
                activity TEXT,
                content TEXT,
                satisfaction REAL,
                progress_made REAL,
                insights TEXT
            );
            
            CREATE TABLE IF NOT EXISTS expertise_receipts (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                receipt_id TEXT,
                timestamp REAL,
                level_before REAL,
                level_after REAL,
                breakthrough INTEGER DEFAULT 0
            );
        """)
        self.conn.commit()
    
    def become_expert(
        self, 
        domain: str, 
        depth: str = "moderate",
        focus_areas: List[str] = None,
        learning_plan: Dict[str, Any] = None
    ) -> ExpertiseArea:
        """
        Tell RIVIR to become an expert in a domain.
        
        Args:
            domain: What to become expert in (e.g., "meditation app design")
            depth: How deep to go (surface, moderate, deep, master)
            focus_areas: Specific areas to focus on
            learning_plan: Custom learning approach
        """
        print(f"🎯 RIVIR becoming expert in: {domain}")
        print(f"   Depth: {depth}")
        
        # Create expertise area
        expertise = ExpertiseArea(
            domain=domain,
            depth=depth,
            focus_areas=focus_areas or self._generate_focus_areas(domain),
            learning_sources=self._generate_learning_sources(domain),
            practice_tasks=self._generate_practice_tasks(domain),
            mastery_indicators=self._generate_mastery_indicators(domain, depth)
        )
        
        # Store in database
        self.conn.execute("""
            INSERT OR REPLACE INTO expertise_areas 
            (domain, depth, focus_areas, learning_sources, practice_tasks, 
             mastery_indicators, current_level, started, last_practice)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            expertise.domain,
            expertise.depth,
            json.dumps(expertise.focus_areas),
            json.dumps(expertise.learning_sources),
            json.dumps(expertise.practice_tasks),
            json.dumps(expertise.mastery_indicators),
            expertise.current_level,
            expertise.started,
            expertise.last_practice
        ))
        self.conn.commit()
        
        # Activate for learning
        self.active_expertise[domain] = expertise
        
        # Create initial learning session
        self._start_learning_session(domain)
        
        print(f"✓ Expertise training started")
        print(f"   Focus areas: {', '.join(expertise.focus_areas[:3])}...")
        print(f"   Practice tasks: {len(expertise.practice_tasks)} tasks")
        
        return expertise
    
    def _generate_focus_areas(self, domain: str) -> List[str]:
        """Generate focus areas for a domain."""
        # Use RIVIR to generate focus areas
        prompt = f"List 5-7 key focus areas for becoming expert in {domain}. Format as simple bullet points."
        output = self.rivir.collapse_and_create(prompt)
        
        # Parse the output to extract focus areas
        areas = []
        text = output[0] if output and output[0] else ""
        
        for line in text.split('\n'):
            line = line.strip()
            if line and not line.startswith('<') and not line.startswith('{'): 
                # Remove bullet points and numbers
                clean_line = line.lstrip('1234567890.-•* ').strip()
                if clean_line and len(clean_line) > 5 and len(clean_line) < 100:
                    areas.append(clean_line)
        
        # Fallback if parsing failed
        if not areas:
            areas = [
                f"{domain} fundamentals",
                f"{domain} best practices", 
                f"{domain} user experience",
                f"{domain} technical implementation",
                f"{domain} design patterns"
            ]
        
        return areas[:7]
    
    def _generate_learning_sources(self, domain: str) -> List[str]:
        """Generate learning sources for a domain."""
        return [
            f"Documentation and guides for {domain}",
            f"Best practices in {domain}",
            f"Case studies and examples",
            f"Expert insights and patterns",
            f"Common pitfalls and solutions"
        ]
    
    def _generate_practice_tasks(self, domain: str) -> List[str]:
        """Generate practice tasks for a domain."""
        prompt = f"List 10 practice tasks to master {domain}, from beginner to advanced"
        output = self.rivir.collapse_and_create(prompt)
        
        tasks = []
        for line in output[0].split('\n'):
            if line.strip() and ('.' in line or '-' in line):
                task = line.strip().lstrip('1234567890.-• ')
                if task:
                    tasks.append(task)
        
        return tasks[:10] if tasks else [f"Practice basic {domain}", f"Build advanced {domain} project"]
    
    def _generate_mastery_indicators(self, domain: str, depth: str) -> List[str]:
        """Generate mastery indicators for a domain and depth."""
        depth_indicators = {
            "surface": ["Can explain basics", "Can identify key concepts", "Can follow tutorials"],
            "moderate": ["Can solve common problems", "Can adapt examples", "Can teach basics"],
            "deep": ["Can design solutions", "Can optimize performance", "Can handle edge cases"],
            "master": ["Can innovate new approaches", "Can mentor others", "Can set best practices"]
        }
        
        base_indicators = depth_indicators.get(depth, depth_indicators["moderate"])
        domain_specific = [f"Can create high-quality {domain}", f"Understands {domain} deeply"]
        
        return base_indicators + domain_specific
    
    def _start_learning_session(self, domain: str):
        """Start a learning session for a domain."""
        expertise = self.active_expertise.get(domain)
        if not expertise:
            return
        
        # Generate learning content
        focus_area = expertise.focus_areas[0] if expertise.focus_areas else domain
        prompt = f"Teach me about {focus_area} in {domain}. Be comprehensive but practical."
        
        output = self.rivir.collapse_and_create(prompt)
        
        # Record learning session
        self.conn.execute("""
            INSERT INTO learning_sessions 
            (domain, timestamp, activity, content, satisfaction, progress_made)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            domain,
            time.time(),
            f"Learning about {focus_area}",
            output[0][:1000],  # Truncate for storage
            0.8,  # Default satisfaction
            0.1   # Small progress increment
        ))
        self.conn.commit()
        
        # Update expertise level
        expertise.current_level = min(1.0, expertise.current_level + 0.1)
        expertise.last_practice = time.time()
    
    def practice_skill(self, domain: str, task_description: str = None) -> str:
        """Practice a skill in the domain."""
        expertise = self.active_expertise.get(domain)
        if not expertise:
            return "Domain not found in active expertise"
        
        # Choose practice task
        if not task_description and expertise.practice_tasks:
            # Choose task based on current level
            task_index = min(len(expertise.practice_tasks) - 1, 
                           int(expertise.current_level * len(expertise.practice_tasks)))
            task_description = expertise.practice_tasks[task_index]
        
        if not task_description:
            task_description = f"Practice {domain} skills"
        
        print(f"🏋️ Practicing: {task_description}")
        
        # Generate practice content
        prompt = f"As an expert in {domain}, complete this task: {task_description}"
        output = self.rivir.collapse_and_create(prompt)
        
        # Record practice session
        progress = 0.05 + (0.1 if "advanced" in task_description.lower() else 0.0)
        
        self.conn.execute("""
            INSERT INTO learning_sessions 
            (domain, timestamp, activity, content, progress_made)
            VALUES (?, ?, ?, ?, ?)
        """, (
            domain,
            time.time(),
            f"Practice: {task_description}",
            output[0][:1000],
            progress
        ))
        self.conn.commit()
        
        # Update expertise
        expertise.current_level = min(1.0, expertise.current_level + progress)
        expertise.last_practice = time.time()
        
        print(f"✓ Practice complete. Level: {expertise.current_level:.1%}")
        
        return output[0]
    
    def get_expertise_status(self, domain: str = None) -> Dict[str, Any]:
        """Get current expertise status."""
        if domain:
            expertise = self.active_expertise.get(domain)
            if expertise:
                return {
                    'domain': expertise.domain,
                    'level': expertise.current_level,
                    'depth': expertise.depth,
                    'focus_areas': expertise.focus_areas,
                    'days_training': (time.time() - expertise.started) / 86400,
                    'last_practice': expertise.last_practice
                }
            return {}
        else:
            # All expertise areas
            return {
                domain: {
                    'level': exp.current_level,
                    'depth': exp.depth,
                    'days_training': (time.time() - exp.started) / 86400
                }
                for domain, exp in self.active_expertise.items()
            }


# =============================================================================
# SILENT WATCHER SYSTEM
# =============================================================================

@dataclass
class WatchEvent:
    """An observed event from the user."""
    timestamp: float
    event_type: str  # keystroke, app_switch, file_open, command, etc.
    content: str
    context: Dict[str, Any] = field(default_factory=dict)
    app_name: str = ""
    file_path: str = ""


class RIVIRWatcher:
    """Silently watches user activity to learn patterns and preferences."""
    
    def __init__(self, rivir: RIVIRDirect, db_path: str = "rivir_watcher.db"):
        self.rivir = rivir
        self.db_path = db_path
        self.watching = False
        self.watch_thread = None
        self.events_queue = []
        self.patterns_learned = {}
        
        # Initialize DB connection in main thread
        self._init_connection()
        
    def _init_connection(self):
        """Initialize database connection."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        
    def _init_db(self):
        """Initialize watcher database."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS watch_events (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                event_type TEXT,
                content TEXT,
                context TEXT,
                app_name TEXT,
                file_path TEXT
            );
            
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                last_seen REAL,
                frequency INTEGER DEFAULT 1
            );
            
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY,
                category TEXT,
                preference TEXT,
                strength REAL,
                evidence TEXT,
                learned_at REAL
            );
        """)
        self.conn.commit()
    
    def start_watching(self, watch_types: List[str] = None):
        """
        Start silently watching user activity.
        
        Args:
            watch_types: Types of events to watch 
                        ['keystrokes', 'apps', 'files', 'commands', 'clipboard']
        """
        if self.watching:
            print("Already watching")
            return
        
        watch_types = watch_types or ['apps', 'files', 'commands']
        
        print(f"👁️  Starting silent watch...")
        print(f"   Watching: {', '.join(watch_types)}")
        print(f"   Learning patterns and preferences")
        print(f"   (Press Ctrl+C to stop)")
        
        self.watching = True
        self.watch_thread = threading.Thread(
            target=self._watch_loop, 
            args=(watch_types,),
            daemon=True
        )
        self.watch_thread.start()
    
    def _watch_loop(self, watch_types: List[str]):
        """Main watching loop."""
        last_app = ""
        last_file = ""
        
        while self.watching:
            try:
                # Watch active application
                if 'apps' in watch_types:
                    current_app = self._get_active_app()
                    if current_app and current_app != last_app:
                        self._record_event(WatchEvent(
                            timestamp=time.time(),
                            event_type="app_switch",
                            content=current_app,
                            app_name=current_app
                        ))
                        last_app = current_app
                
                # Watch file operations (recent files)
                if 'files' in watch_types:
                    recent_files = self._get_recent_files()
                    for file_path in recent_files:
                        if file_path != last_file:
                            self._record_event(WatchEvent(
                                timestamp=time.time(),
                                event_type="file_access",
                                content=file_path,
                                file_path=file_path,
                                app_name=last_app
                            ))
                            last_file = file_path
                
                # Watch commands (if terminal is active)
                if 'commands' in watch_types and 'terminal' in last_app.lower():
                    # This would need more sophisticated monitoring
                    pass
                
                # Process events periodically
                if len(self.events_queue) > 10:
                    self._process_events()
                
                time.sleep(2)  # Check every 2 seconds
                
            except Exception as e:
                print(f"Watch error: {e}")
                time.sleep(5)
    
    def _get_active_app(self) -> str:
        """Get currently active application."""
        try:
            # macOS
            result = subprocess.run([
                'osascript', '-e', 
                'tell application "System Events" to get name of first application process whose frontmost is true'
            ], capture_output=True, text=True, timeout=1)
            
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        
        return ""
    
    def _get_recent_files(self) -> List[str]:
        """Get recently accessed files."""
        try:
            # Check recent files in various ways
            recent = []
            
            # macOS recent files
            result = subprocess.run([
                'mdfind', '-onlyin', os.path.expanduser('~'), 
                'kMDItemLastUsedDate >= $time.now(-3600)'  # Last hour
            ], capture_output=True, text=True, timeout=2)
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                recent.extend([f for f in files if f and not f.startswith('.')])
            
            return recent[:5]  # Limit to 5 most recent
        except:
            return []
    
    def _record_event(self, event: WatchEvent):
        """Record a watch event."""
        self.events_queue.append(event)
        
        # Store in database
        self.conn.execute("""
            INSERT INTO watch_events 
            (timestamp, event_type, content, context, app_name, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event.timestamp,
            event.event_type,
            event.content,
            json.dumps(event.context),
            event.app_name,
            event.file_path
        ))
        self.conn.commit()
    
    def _process_events(self):
        """Process accumulated events to learn patterns."""
        if not self.events_queue:
            return
        
        # Analyze patterns
        app_usage = {}
        file_patterns = {}
        time_patterns = {}
        
        for event in self.events_queue:
            # App usage patterns
            if event.event_type == "app_switch":
                app_usage[event.app_name] = app_usage.get(event.app_name, 0) + 1
            
            # File type patterns
            if event.event_type == "file_access" and event.file_path:
                ext = Path(event.file_path).suffix.lower()
                if ext:
                    file_patterns[ext] = file_patterns.get(ext, 0) + 1
            
            # Time patterns
            hour = int(time.strftime('%H', time.localtime(event.timestamp)))
            time_patterns[hour] = time_patterns.get(hour, 0) + 1
        
        # Learn from patterns
        self._learn_app_preferences(app_usage)
        self._learn_file_preferences(file_patterns)
        self._learn_time_preferences(time_patterns)
        
        # Clear processed events
        self.events_queue.clear()
    
    def _learn_app_preferences(self, app_usage: Dict[str, int]):
        """Learn application preferences."""
        total_switches = sum(app_usage.values())
        
        for app, count in app_usage.items():
            preference_strength = count / total_switches
            
            if preference_strength > 0.2:  # Significant usage
                self.conn.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (category, preference, strength, evidence, learned_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    "app_preference",
                    app,
                    preference_strength,
                    f"Used {count}/{total_switches} times",
                    time.time()
                ))
        
        self.conn.commit()
    
    def _learn_file_preferences(self, file_patterns: Dict[str, int]):
        """Learn file type preferences."""
        total_files = sum(file_patterns.values())
        
        for ext, count in file_patterns.items():
            preference_strength = count / total_files
            
            if preference_strength > 0.1:
                self.conn.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (category, preference, strength, evidence, learned_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    "file_type_preference",
                    ext,
                    preference_strength,
                    f"Accessed {count}/{total_files} files",
                    time.time()
                ))
        
        self.conn.commit()
    
    def _learn_time_preferences(self, time_patterns: Dict[int, int]):
        """Learn time-based activity patterns."""
        total_activity = sum(time_patterns.values())
        
        for hour, count in time_patterns.items():
            activity_level = count / total_activity
            
            if activity_level > 0.1:  # Significant activity
                self.conn.execute("""
                    INSERT OR REPLACE INTO user_preferences 
                    (category, preference, strength, evidence, learned_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    "time_preference",
                    f"hour_{hour}",
                    activity_level,
                    f"Active {count}/{total_activity} times",
                    time.time()
                ))
        
        self.conn.commit()
    
    def stop_watching(self):
        """Stop watching user activity."""
        if not self.watching:
            return
        
        print("🛑 Stopping silent watch...")
        self.watching = False
        
        if self.watch_thread:
            self.watch_thread.join(timeout=2)
        
        # Process remaining events
        self._process_events()
        
        # Generate insights
        insights = self.get_learned_insights()
        print(f"✓ Watch stopped. Learned {len(insights)} insights")
    
    def get_learned_insights(self) -> Dict[str, Any]:
        """Get insights learned from watching."""
        preferences = self.conn.execute("""
            SELECT category, preference, strength, evidence 
            FROM user_preferences 
            ORDER BY strength DESC
        """).fetchall()
        
        insights = {
            'app_preferences': [],
            'file_preferences': [],
            'time_preferences': [],
            'patterns': []
        }
        
        for pref in preferences:
            category = pref['category']
            if category == 'app_preference':
                insights['app_preferences'].append({
                    'app': pref['preference'],
                    'strength': pref['strength'],
                    'evidence': pref['evidence']
                })
            elif category == 'file_type_preference':
                insights['file_preferences'].append({
                    'file_type': pref['preference'],
                    'strength': pref['strength'],
                    'evidence': pref['evidence']
                })
            elif category == 'time_preference':
                insights['time_preferences'].append({
                    'time': pref['preference'],
                    'strength': pref['strength'],
                    'evidence': pref['evidence']
                })
        
        return insights
    
    def apply_learned_preferences(self):
        """Apply learned preferences to RIVIR generation."""
        insights = self.get_learned_insights()
        
        # Create preference context for RIVIR
        context = {
            'user_prefers_apps': [p['app'] for p in insights['app_preferences'][:3]],
            'user_prefers_files': [p['file_type'] for p in insights['file_preferences'][:3]],
            'user_active_times': [p['time'] for p in insights['time_preferences'][:3]],
            'learned_from_watching': True
        }
        
        # This context can be used in RIVIR generation
        return context


# =============================================================================
# INTEGRATED EXPERT + WATCHER
# =============================================================================

class RIVIRLearningSystem:
    """Combined expert training and silent watching system."""
    
    def __init__(self, rivir: RIVIRDirect):
        self.rivir = rivir
        self.trainer = ExpertTrainer(rivir)
        self.watcher = RIVIRWatcher(rivir)
    
    def become_expert_and_watch(self, domain: str, depth: str = "moderate"):
        """Become expert in domain while watching user behavior."""
        print(f"🎯👁️  Starting expert training + silent watching for: {domain}")
        
        # Start expertise training
        expertise = self.trainer.become_expert(domain, depth)
        
        # Start watching to learn user preferences
        self.watcher.start_watching(['apps', 'files'])
        
        return expertise
    
    def practice_with_context(self, domain: str, task: str = None):
        """Practice skills using learned user context."""
        # Get learned preferences
        context = self.watcher.apply_learned_preferences()
        
        # Practice with context
        result = self.trainer.practice_skill(domain, task)
        
        return result, context
    
    def get_full_status(self):
        """Get status of both expert training and watching."""
        return {
            'expertise': self.trainer.get_expertise_status(),
            'insights': self.watcher.get_learned_insights(),
            'watching': self.watcher.watching
        }


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Interactive CLI for expert training and watching."""
    print("🎯👁️  RIVIR Expert Training & Silent Watcher")
    print("=" * 50)
    
    rivir = RIVIRDirect()
    system = RIVIRLearningSystem(rivir)
    
    print("""
Commands:
  expert <domain> [depth]     - Become expert in domain
  practice <domain> [task]    - Practice skills in domain
  watch                       - Start silent watching
  stop                        - Stop watching
  status                      - Show learning status
  insights                    - Show learned insights
  quit                        - Exit
    """)
    
    try:
        while True:
            try:
                cmd = input("\nRIVIR> ").strip().split()
                if not cmd:
                    continue
                
                if cmd[0] == 'quit':
                    break
                elif cmd[0] == 'expert':
                    domain = ' '.join(cmd[1:]) if len(cmd) > 1 else input("Domain: ")
                    depth = input("Depth (surface/moderate/deep/master) [moderate]: ") or "moderate"
                    system.trainer.become_expert(domain, depth)
                
                elif cmd[0] == 'practice':
                    domain = ' '.join(cmd[1:]) if len(cmd) > 1 else input("Domain: ")
                    task = input("Task (optional): ") or None
                    result, context = system.practice_with_context(domain, task)
                    print(f"Practice result: {result[:200]}...")
                
                elif cmd[0] == 'watch':
                    system.watcher.start_watching()
                
                elif cmd[0] == 'stop':
                    system.watcher.stop_watching()
                
                elif cmd[0] == 'status':
                    status = system.get_full_status()
                    print(f"Expertise areas: {len(status['expertise'])}")
                    for domain, info in status['expertise'].items():
                        print(f"  {domain}: {info['level']:.1%} ({info['depth']})")
                    print(f"Watching: {status['watching']}")
                
                elif cmd[0] == 'insights':
                    insights = system.watcher.get_learned_insights()
                    print("Learned insights:")
                    for category, items in insights.items():
                        if items:
                            print(f"  {category}: {len(items)} items")
                            for item in items[:3]:
                                print(f"    {item}")
                
                else:
                    print("Unknown command")
                    
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    finally:
        system.watcher.stop_watching()
        rivir.close()


if __name__ == "__main__":
    main()