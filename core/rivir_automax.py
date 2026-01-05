#!/usr/bin/env python3
"""
RIVIR Auto-Max-Learning Script

This is the ultimate self-improvement system for RIVIR:
- Continuously learns new domains
- Auto-practices skills based on usage patterns
- Adapts learning strategy based on success rates
- Watches user behavior to prioritize learning
- Self-optimizes generation quality
- Builds expertise networks across domains

Usage:
    python3 rivir_automax.py --domains "web dev,ai,design" --watch --continuous
    python3 rivir_automax.py --boost-domain "python" --intensity high
"""

import argparse
import asyncio
import json
import time
import threading
from typing import Dict, List, Set, Optional
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

try:
    from rivir_llm import RIVIRModel, GenerationConfig
    from rivir_learning import RIVIRLearningSystem, ExpertTrainer, RIVIRWatcher
    from rivir_direct import RIVIRDirect
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all RIVIR modules are available")
    sys.exit(1)


@dataclass
class LearningGoal:
    """A learning goal with adaptive strategy."""
    domain: str
    target_level: float = 0.8
    priority: float = 1.0
    learning_rate: float = 0.1
    success_rate: float = 0.0
    last_practice: float = 0.0
    practice_count: int = 0
    adaptive_strategy: str = "balanced"  # aggressive, balanced, conservative
    
    def should_practice(self, current_time: float, min_interval: float = 300) -> bool:
        """Determine if this goal should be practiced now."""
        time_since_practice = current_time - self.last_practice
        
        # Higher priority = more frequent practice
        adjusted_interval = min_interval / self.priority
        
        # Lower success rate = more frequent practice
        if self.success_rate < 0.5:
            adjusted_interval *= 0.5
        
        return time_since_practice >= adjusted_interval


class RIVIRAutoMaxLearner:
    """
    The ultimate RIVIR auto-learning system.
    
    This system:
    1. Continuously learns new domains based on user patterns
    2. Auto-practices skills with adaptive scheduling
    3. Optimizes learning strategies based on success rates
    4. Builds cross-domain expertise networks
    5. Self-tunes generation parameters
    """
    
    def __init__(self, db_prefix: str = "rivir_automax"):
        # Core RIVIR system
        self.rivir = RIVIRDirect(db_prefix)
        self.model = RIVIRModel(db_prefix)
        self.learning_system = RIVIRLearningSystem(self.rivir)
        
        # Auto-learning state
        self.learning_goals: Dict[str, LearningGoal] = {}
        self.domain_network: Dict[str, Set[str]] = {}  # Related domains
        self.generation_history: List[Dict] = []
        self.optimization_metrics: Dict[str, float] = {}
        
        # Control flags
        self.running = False
        self.continuous_mode = False
        self.watch_mode = False
        
        # Learning parameters
        self.max_concurrent_domains = 5
        self.practice_interval = 300  # 5 minutes
        self.optimization_interval = 1800  # 30 minutes
        self.discovery_interval = 3600  # 1 hour
        
        # Performance tracking
        self.session_stats = {
            'domains_learned': 0,
            'practices_completed': 0,
            'optimizations_applied': 0,
            'discoveries_made': 0,
            'avg_satisfaction': 0.0,
            'learning_velocity': 0.0
        }
    
    def add_learning_goal(
        self, 
        domain: str, 
        target_level: float = 0.8,
        priority: float = 1.0,
        strategy: str = "balanced"
    ):
        """Add a new learning goal."""
        print(f"🎯 Adding learning goal: {domain} (target: {target_level:.0%})")
        
        goal = LearningGoal(
            domain=domain,
            target_level=target_level,
            priority=priority,
            adaptive_strategy=strategy
        )
        
        self.learning_goals[domain] = goal
        
        # Start expertise training
        self.learning_system.trainer.become_expert(domain, "deep")
        self.session_stats['domains_learned'] += 1
        
        # Discover related domains
        self._discover_related_domains(domain)
    
    def _discover_related_domains(self, domain: str):
        """Discover domains related to the given domain."""
        prompt = f"List 3-5 domains closely related to {domain}. One word per domain."
        output = self.model.generate(prompt, max_length=100)
        
        related = set()
        for line in output.text.split('\n'):
            line = line.strip().lower()
            if line and len(line.split()) <= 2 and line != domain.lower():
                related.add(line)
        
        self.domain_network[domain] = related
        print(f"🔗 Discovered related domains for {domain}: {', '.join(related)}")
        self.session_stats['discoveries_made'] += 1
    
    def _adaptive_practice(self, goal: LearningGoal) -> bool:
        """Practice a domain with adaptive strategy."""
        domain = goal.domain
        
        # Get current expertise level
        expertise = self.learning_system.trainer.active_expertise.get(domain)
        if not expertise:
            return False
        
        current_level = expertise.current_level
        
        # Choose practice intensity based on strategy and success rate
        if goal.adaptive_strategy == "aggressive" or goal.success_rate < 0.3:
            # High intensity practice
            practice_count = 3
            difficulty = "advanced"
        elif goal.adaptive_strategy == "conservative" or goal.success_rate > 0.8:
            # Light practice
            practice_count = 1
            difficulty = "basic"
        else:
            # Balanced practice
            practice_count = 2
            difficulty = "intermediate"
        
        print(f"🏋️ Auto-practicing {domain} ({practice_count}x {difficulty})")
        
        success_count = 0
        for i in range(practice_count):
            try:
                # Generate practice task
                task_prompt = f"Create a {difficulty} {domain} practice exercise"
                task_output = self.model.generate(task_prompt, max_length=200)
                task = task_output.text[:100]  # Truncate for practice
                
                # Practice the skill
                result = self.learning_system.trainer.practice_skill(domain, task)
                
                # Evaluate success (simple heuristic)
                if len(result) > 100 and "error" not in result.lower():
                    success_count += 1
                
                time.sleep(1)  # Brief pause between practices
                
            except Exception as e:
                print(f"Practice error: {e}")
        
        # Update goal metrics
        goal.practice_count += practice_count
        goal.success_rate = (goal.success_rate * (goal.practice_count - practice_count) + success_count) / goal.practice_count
        goal.last_practice = time.time()
        
        # Adapt strategy based on performance
        if goal.success_rate > 0.8 and goal.adaptive_strategy == "aggressive":
            goal.adaptive_strategy = "balanced"
            print(f"📈 {domain}: Switching to balanced strategy (high success)")
        elif goal.success_rate < 0.3 and goal.adaptive_strategy != "aggressive":
            goal.adaptive_strategy = "aggressive"
            print(f"📉 {domain}: Switching to aggressive strategy (low success)")
        
        self.session_stats['practices_completed'] += practice_count
        return success_count > 0
    
    def _optimize_generation(self):
        """Optimize generation parameters based on recent performance."""
        print("⚙️ Optimizing generation parameters...")
        
        # Analyze recent generation history
        if len(self.generation_history) < 5:
            return
        
        recent = self.generation_history[-20:]  # Last 20 generations
        
        # Calculate metrics
        avg_satisfaction = sum(g.get('satisfaction', 0.5) for g in recent) / len(recent)
        avg_alignment = sum(g.get('alignment', 0.5) for g in recent) / len(recent)
        avg_length = sum(len(g.get('text', '')) for g in recent) / len(recent)
        
        # Optimize based on metrics
        config_updates = {}
        
        if avg_satisfaction < 0.6:
            # Low satisfaction - increase quality focus
            config_updates['temperature'] = max(0.3, self.model.config.temperature - 0.1)
            config_updates['amplify_quality'] = True
            print("🔧 Reducing temperature for higher quality")
        
        elif avg_satisfaction > 0.9:
            # High satisfaction - can increase creativity
            config_updates['temperature'] = min(0.9, self.model.config.temperature + 0.1)
            print("🔧 Increasing temperature for more creativity")
        
        if avg_alignment < 0.7:
            # Low alignment - strengthen ethical checks
            config_updates['white_hat_check'] = True
            print("🔧 Strengthening ethical alignment")
        
        if avg_length < 200:
            # Too short - increase max length
            config_updates['max_length'] = min(2000, self.model.config.max_length + 200)
            print("🔧 Increasing max generation length")
        
        # Apply updates
        for key, value in config_updates.items():
            if hasattr(self.model.config, key):
                setattr(self.model.config, key, value)
        
        self.optimization_metrics.update({
            'avg_satisfaction': avg_satisfaction,
            'avg_alignment': avg_alignment,
            'avg_length': avg_length
        })
        
        self.session_stats['optimizations_applied'] += len(config_updates)
    
    def _auto_discover_domains(self):
        """Automatically discover new domains to learn based on user patterns."""
        if not self.learning_system.watcher.watching:
            return
        
        print("🔍 Auto-discovering new learning domains...")
        
        # Get learned insights
        insights = self.learning_system.watcher.get_learned_insights()
        
        # Analyze app preferences for domain hints
        app_domains = {
            'code': ['vscode', 'pycharm', 'sublime', 'atom', 'vim'],
            'design': ['figma', 'sketch', 'photoshop', 'illustrator'],
            'productivity': ['notion', 'obsidian', 'todoist', 'trello'],
            'data': ['jupyter', 'tableau', 'excel', 'r studio'],
            'web': ['chrome', 'firefox', 'safari', 'postman']
        }
        
        user_apps = [p['app'].lower() for p in insights.get('app_preferences', [])]
        
        suggested_domains = set()
        for domain, apps in app_domains.items():
            if any(app in ' '.join(user_apps) for app in apps):
                suggested_domains.add(domain)
        
        # Add domains that aren't already being learned
        for domain in suggested_domains:
            if domain not in self.learning_goals and len(self.learning_goals) < self.max_concurrent_domains:
                self.add_learning_goal(domain, priority=0.8, strategy="balanced")
    
    def _learning_loop(self):
        """Main learning loop."""
        last_optimization = 0
        last_discovery = 0
        
        while self.running:
            try:
                current_time = time.time()
                
                # Practice skills that need attention
                for goal in self.learning_goals.values():
                    if goal.should_practice(current_time, self.practice_interval):
                        self._adaptive_practice(goal)
                
                # Periodic optimization
                if current_time - last_optimization > self.optimization_interval:
                    self._optimize_generation()
                    last_optimization = current_time
                
                # Periodic domain discovery
                if current_time - last_discovery > self.discovery_interval:
                    self._auto_discover_domains()
                    last_discovery = current_time
                
                # Update learning velocity
                practices_per_hour = self.session_stats['practices_completed'] / max(1, (current_time - self.start_time) / 3600)
                self.session_stats['learning_velocity'] = practices_per_hour
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                print(f"Learning loop error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def start_auto_learning(
        self, 
        domains: List[str] = None,
        watch: bool = True,
        continuous: bool = True
    ):
        """Start the auto-learning system."""
        print("🚀 Starting RIVIR Auto-Max-Learning System")
        print("=" * 50)
        
        self.running = True
        self.continuous_mode = continuous
        self.watch_mode = watch
        self.start_time = time.time()
        
        # Add initial domains
        if domains:
            for domain in domains:
                self.add_learning_goal(domain)
        else:
            # Default domains
            default_domains = ["web development", "ai assistance", "user experience"]
            for domain in default_domains:
                self.add_learning_goal(domain)
        
        # Start watching if requested
        if watch:
            print("👁️ Starting user behavior watching...")
            self.learning_system.watcher.start_watching(['apps', 'files'])
        
        # Start learning loop
        if continuous:
            print("🔄 Starting continuous learning loop...")
            self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
            self.learning_thread.start()
        
        print("✅ Auto-learning system active!")
        self._print_status()
    
    def stop_auto_learning(self):
        """Stop the auto-learning system."""
        print("\n🛑 Stopping auto-learning system...")
        
        self.running = False
        
        if self.watch_mode:
            self.learning_system.watcher.stop_watching()
        
        # Final optimization
        if self.generation_history:
            self._optimize_generation()
        
        self._print_final_stats()
        
        # Cleanup
        self.model.close()
        self.rivir.close()
    
    def boost_domain(self, domain: str, intensity: str = "high"):
        """Intensively boost learning in a specific domain."""
        print(f"🚀 Boosting {domain} learning (intensity: {intensity})")
        
        if domain not in self.learning_goals:
            self.add_learning_goal(domain)
        
        goal = self.learning_goals[domain]
        
        # Set aggressive parameters
        if intensity == "high":
            goal.priority = 3.0
            goal.adaptive_strategy = "aggressive"
            practice_rounds = 10
        elif intensity == "medium":
            goal.priority = 2.0
            goal.adaptive_strategy = "balanced"
            practice_rounds = 5
        else:
            goal.priority = 1.5
            practice_rounds = 3
        
        # Intensive practice session
        print(f"🏋️ Starting {practice_rounds}-round intensive practice...")
        for i in range(practice_rounds):
            print(f"Round {i+1}/{practice_rounds}")
            success = self._adaptive_practice(goal)
            if success:
                print("✅ Practice successful")
            else:
                print("⚠️ Practice had issues")
            time.sleep(2)
        
        expertise = self.learning_system.trainer.active_expertise.get(domain)
        if expertise:
            print(f"🎯 {domain} expertise level: {expertise.current_level:.1%}")
    
    def generate_with_learning(self, prompt: str, learn_from_result: bool = True) -> str:
        """Generate with automatic learning from the result."""
        # Generate using current expertise
        output = self.model.generate(prompt)
        
        # Record for learning
        generation_record = {
            'prompt': prompt,
            'text': output.text,
            'alignment': output.alignment_score,
            'timestamp': time.time(),
            'expertise_used': list(self.learning_goals.keys())
        }
        
        self.generation_history.append(generation_record)
        
        if learn_from_result:
            # Auto-feedback based on output quality heuristics
            quality_score = self._estimate_quality(output.text, output.alignment_score)
            self.model.learn_from_feedback(output.response_id, quality_score)
            generation_record['satisfaction'] = quality_score
        
        return output.text
    
    def _estimate_quality(self, text: str, alignment: float) -> float:
        """Estimate quality score for auto-feedback."""
        score = alignment * 0.4  # Base on alignment
        
        # Length factor
        if 100 <= len(text) <= 1000:
            score += 0.2
        elif len(text) > 1000:
            score += 0.1
        
        # Content quality heuristics
        if any(word in text.lower() for word in ['error', 'failed', 'broken']):
            score -= 0.3
        
        if any(word in text.lower() for word in ['elegant', 'beautiful', 'clean', 'professional']):
            score += 0.2
        
        # HTML/code quality
        if '<' in text and '>' in text:
            if text.count('<') == text.count('>'):
                score += 0.1  # Balanced tags
        
        return max(0.0, min(1.0, score))
    
    def _print_status(self):
        """Print current learning status."""
        print(f"\n📊 Learning Status:")
        print(f"   Active domains: {len(self.learning_goals)}")
        for domain, goal in self.learning_goals.items():
            expertise = self.learning_system.trainer.active_expertise.get(domain)
            level = expertise.current_level if expertise else 0.0
            print(f"   • {domain}: {level:.1%} (success: {goal.success_rate:.1%})")
        
        print(f"   Watching: {self.learning_system.watcher.watching}")
        print(f"   Continuous: {self.continuous_mode}")
    
    def _print_final_stats(self):
        """Print final session statistics."""
        duration = time.time() - self.start_time
        
        print(f"\n📈 Session Complete ({duration/60:.1f} minutes)")
        print(f"   Domains learned: {self.session_stats['domains_learned']}")
        print(f"   Practices completed: {self.session_stats['practices_completed']}")
        print(f"   Optimizations applied: {self.session_stats['optimizations_applied']}")
        print(f"   Discoveries made: {self.session_stats['discoveries_made']}")
        print(f"   Learning velocity: {self.session_stats['learning_velocity']:.1f} practices/hour")
        
        if self.optimization_metrics:
            print(f"   Final satisfaction: {self.optimization_metrics.get('avg_satisfaction', 0):.1%}")


def main():
    """CLI interface for auto-max-learning."""
    parser = argparse.ArgumentParser(description="RIVIR Auto-Max-Learning System")
    parser.add_argument("--domains", type=str, help="Comma-separated domains to learn")
    parser.add_argument("--watch", action="store_true", help="Enable user behavior watching")
    parser.add_argument("--continuous", action="store_true", default=True, help="Enable continuous learning")
    parser.add_argument("--boost-domain", type=str, help="Intensively boost a specific domain")
    parser.add_argument("--intensity", choices=["low", "medium", "high"], default="high", help="Boost intensity")
    parser.add_argument("--duration", type=int, default=0, help="Run duration in minutes (0 = indefinite)")
    
    args = parser.parse_args()
    
    # Create learner
    learner = RIVIRAutoMaxLearner()
    
    try:
        if args.boost_domain:
            # Boost mode
            learner.boost_domain(args.boost_domain, args.intensity)
        else:
            # Normal auto-learning mode
            domains = args.domains.split(",") if args.domains else None
            learner.start_auto_learning(domains, args.watch, args.continuous)
            
            if args.duration > 0:
                print(f"⏰ Running for {args.duration} minutes...")
                time.sleep(args.duration * 60)
            else:
                print("⏰ Running indefinitely (Ctrl+C to stop)...")
                while True:
                    time.sleep(60)
                    learner._print_status()
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user")
    
    finally:
        learner.stop_auto_learning()


if __name__ == "__main__":
    main()