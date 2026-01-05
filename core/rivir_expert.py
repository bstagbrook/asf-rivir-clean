#!/usr/bin/env python3
"""
RIVIR Expertise Development System

This allows you to tell RIVIR to "go get good at something" and it will:
1. Break down the domain into learnable components
2. Create practice scenarios and exercises
3. Learn from feedback and iteration
4. Build up expertise over time
5. Track progress and mastery levels

Usage:
    rivir = RIVIRExpert()
    rivir.learn_domain("web development")
    rivir.learn_domain("meditation app design") 
    rivir.learn_domain("quantum computing")
    
    # Check expertise
    level = rivir.get_expertise_level("web development")
    
    # Practice and improve
    rivir.practice_domain("web development", iterations=10)
"""

import json
import sqlite3
import time
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

try:
    from rivir_direct import RIVIRDirect
    from deep_listener import DeepListener, DeepListening
    from white_hat_guru import WhiteHatGuru
except ImportError:
    from .rivir_direct import RIVIRDirect
    from .deep_listener import DeepListener, DeepListening
    from .white_hat_guru import WhiteHatGuru


# =============================================================================
# EXPERTISE TRACKING
# =============================================================================

@dataclass
class ExpertiseDomain:
    """A domain of expertise to develop."""
    name: str
    description: str
    components: List[str] = field(default_factory=list)
    mastery_level: float = 0.0  # 0.0 to 1.0
    practice_count: int = 0
    success_rate: float = 0.0
    last_practiced: float = 0.0
    learning_path: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'components': self.components,
            'mastery_level': self.mastery_level,
            'practice_count': self.practice_count,
            'success_rate': self.success_rate,
            'last_practiced': self.last_practiced,
            'learning_path': self.learning_path,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses
        }


@dataclass
class PracticeSession:
    """A practice session for developing expertise."""
    domain: str
    challenge: str
    approach: str
    outcome: str
    success: bool
    feedback: str
    lessons_learned: List[str]
    timestamp: float
    
    def to_dict(self) -> dict:
        return {
            'domain': self.domain,
            'challenge': self.challenge,
            'approach': self.approach,
            'outcome': self.outcome,
            'success': self.success,
            'feedback': self.feedback,
            'lessons_learned': self.lessons_learned,
            'timestamp': self.timestamp
        }


# =============================================================================
# EXPERTISE DEVELOPMENT ENGINE
# =============================================================================

class RIVIRExpert:
    """
    RIVIR with expertise development capabilities.
    
    This system can:
    - Learn new domains from scratch
    - Break down complex skills into components
    - Practice and iterate to improve
    - Track mastery levels and progress
    - Adapt learning based on feedback
    """
    
    def __init__(self, db_prefix: str = "rivir_expert"):
        # Core RIVIR system
        self.rivir = RIVIRDirect(db_prefix)
        
        # Expertise tracking
        self.domains: Dict[str, ExpertiseDomain] = {}
        self.practice_history: List[PracticeSession] = []
        
        # Database for persistence
        self.conn = sqlite3.connect(f"{db_prefix}_expertise.db")
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_domains()
        
        # Learning strategies
        self.learning_strategies = {
            'decomposition': self._decompose_domain,
            'practice': self._generate_practice_scenarios,
            'reflection': self._reflect_on_practice,
            'adaptation': self._adapt_learning_path
        }
    
    def _init_db(self):
        """Initialize expertise tracking database."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS domains (
                name TEXT PRIMARY KEY,
                description TEXT,
                components TEXT,
                mastery_level REAL,
                practice_count INTEGER,
                success_rate REAL,
                last_practiced REAL,
                learning_path TEXT,
                strengths TEXT,
                weaknesses TEXT
            );
            
            CREATE TABLE IF NOT EXISTS practice_sessions (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                challenge TEXT,
                approach TEXT,
                outcome TEXT,
                success INTEGER,
                feedback TEXT,
                lessons_learned TEXT,
                timestamp REAL
            );
            
            CREATE TABLE IF NOT EXISTS learning_insights (
                id INTEGER PRIMARY KEY,
                domain TEXT,
                insight TEXT,
                confidence REAL,
                timestamp REAL
            );
        """)
        self.conn.commit()
    
    def _load_domains(self):
        """Load existing domains from database."""
        rows = self.conn.execute("SELECT * FROM domains").fetchall()
        for row in rows:
            domain = ExpertiseDomain(
                name=row['name'],
                description=row['description'],
                components=json.loads(row['components'] or '[]'),
                mastery_level=row['mastery_level'],
                practice_count=row['practice_count'],
                success_rate=row['success_rate'],
                last_practiced=row['last_practiced'],
                learning_path=json.loads(row['learning_path'] or '[]'),
                strengths=json.loads(row['strengths'] or '[]'),
                weaknesses=json.loads(row['weaknesses'] or '[]')
            )
            self.domains[domain.name] = domain
    
    def _save_domain(self, domain: ExpertiseDomain):
        """Save domain to database."""
        self.conn.execute("""
            INSERT OR REPLACE INTO domains 
            (name, description, components, mastery_level, practice_count, success_rate, 
             last_practiced, learning_path, strengths, weaknesses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            domain.name, domain.description, json.dumps(domain.components),
            domain.mastery_level, domain.practice_count, domain.success_rate,
            domain.last_practiced, json.dumps(domain.learning_path),
            json.dumps(domain.strengths), json.dumps(domain.weaknesses)
        ))
        self.conn.commit()
    
    # -------------------------------------------------------------------------
    # DOMAIN LEARNING
    # -------------------------------------------------------------------------
    
    def learn_domain(self, domain_name: str, description: str = "") -> ExpertiseDomain:
        """
        Start learning a new domain or enhance existing knowledge.
        
        This is the main entry point for "go get good at X".
        """
        print(f"🎯 Learning domain: {domain_name}")
        
        # Check if domain exists
        if domain_name in self.domains:
            domain = self.domains[domain_name]
            print(f"   Continuing existing domain (mastery: {domain.mastery_level:.1%})")
        else:
            # Create new domain
            domain = ExpertiseDomain(
                name=domain_name,
                description=description or f"Expertise in {domain_name}"
            )
            self.domains[domain_name] = domain
            print(f"   Created new domain")
        
        # Decompose the domain into learnable components
        print(f"🔍 Analyzing domain structure...")
        self._decompose_domain(domain)
        
        # Create initial learning path
        print(f"🗺️  Creating learning path...")
        self._create_learning_path(domain)
        
        # Save progress
        self._save_domain(domain)
        
        print(f"✅ Domain setup complete!")
        print(f"   Components: {len(domain.components)}")
        print(f"   Learning path: {len(domain.learning_path)} steps")
        
        return domain
    
    def _decompose_domain(self, domain: ExpertiseDomain):
        """Break down a domain into learnable components."""
        # Use RIVIR to analyze the domain
        analysis_prompt = f"""
        Analyze the domain "{domain.name}" and break it down into key learnable components.
        
        Domain: {domain.name}
        Description: {domain.description}
        
        Please identify:
        1. Core concepts and principles
        2. Practical skills needed
        3. Tools and technologies
        4. Common patterns and approaches
        5. Advanced techniques
        
        Format as a structured breakdown.
        """
        
        creation, data = self.rivir.collapse_and_create(analysis_prompt)
        
        # Extract components from the analysis
        # This is a simplified extraction - could be enhanced with NLP
        components = []
        lines = creation.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                component = line.lstrip('-•* ').strip()
                if component and len(component) > 3:
                    components.append(component)
        
        # Update domain
        domain.components = components[:20]  # Limit to top 20 components
        
        print(f"   Identified {len(domain.components)} key components")
    
    def _create_learning_path(self, domain: ExpertiseDomain):
        """Create a structured learning path for the domain."""
        if not domain.components:
            return
        
        # Use RIVIR to create learning sequence
        path_prompt = f"""
        Create an optimal learning path for mastering "{domain.name}".
        
        Available components: {', '.join(domain.components)}
        
        Design a progressive learning sequence that:
        1. Starts with fundamentals
        2. Builds complexity gradually  
        3. Includes practical exercises
        4. Reinforces key concepts
        
        Format as numbered steps.
        """
        
        creation, data = self.rivir.collapse_and_create(path_prompt)
        
        # Extract learning steps
        steps = []
        lines = creation.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('Step')):
                step = line.split('.', 1)[-1].strip()
                if step and len(step) > 5:
                    steps.append(step)
        
        domain.learning_path = steps[:15]  # Limit to 15 steps
        
        print(f"   Created {len(domain.learning_path)} learning steps")
    
    # -------------------------------------------------------------------------
    # PRACTICE AND IMPROVEMENT
    # -------------------------------------------------------------------------
    
    def practice_domain(self, domain_name: str, iterations: int = 5) -> List[PracticeSession]:
        """
        Practice a domain through generated challenges and scenarios.
        
        This is where RIVIR actively improves its expertise.
        """
        if domain_name not in self.domains:
            raise ValueError(f"Domain '{domain_name}' not found. Use learn_domain() first.")
        
        domain = self.domains[domain_name]
        sessions = []
        
        print(f"🏋️  Practicing {domain_name} ({iterations} iterations)")
        
        for i in range(iterations):
            print(f"   Session {i+1}/{iterations}...")
            
            # Generate practice challenge
            challenge = self._generate_practice_challenge(domain)
            
            # Attempt the challenge
            approach, outcome = self._attempt_challenge(domain, challenge)
            
            # Evaluate success
            success, feedback = self._evaluate_attempt(domain, challenge, approach, outcome)
            
            # Learn from the attempt
            lessons = self._extract_lessons(domain, challenge, approach, outcome, success, feedback)
            
            # Create session record
            session = PracticeSession(
                domain=domain_name,
                challenge=challenge,
                approach=approach,
                outcome=outcome,
                success=success,
                feedback=feedback,
                lessons_learned=lessons,
                timestamp=time.time()
            )
            
            sessions.append(session)
            self.practice_history.append(session)
            
            # Save session to database
            self.conn.execute("""
                INSERT INTO practice_sessions 
                (domain, challenge, approach, outcome, success, feedback, lessons_learned, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session.domain, session.challenge, session.approach, session.outcome,
                int(session.success), session.feedback, json.dumps(session.lessons_learned),
                session.timestamp
            ))
            
            # Update domain statistics
            domain.practice_count += 1
            domain.last_practiced = time.time()
            
            print(f"      Challenge: {challenge[:50]}...")
            print(f"      Success: {'✓' if success else '✗'}")
        
        # Update mastery level based on recent performance
        self._update_mastery_level(domain, sessions)
        
        # Adapt learning based on performance
        self._adapt_learning_path(domain, sessions)
        
        # Save updated domain
        self._save_domain(domain)
        
        print(f"✅ Practice complete! New mastery: {domain.mastery_level:.1%}")
        
        return sessions
    
    def _generate_practice_challenge(self, domain: ExpertiseDomain) -> str:
        """Generate a practice challenge for the domain."""
        # Focus on areas that need improvement
        focus_areas = domain.weaknesses if domain.weaknesses else domain.components[:3]
        
        challenge_prompt = f"""
        Create a practical challenge for developing expertise in "{domain.name}".
        
        Focus areas: {', '.join(focus_areas)}
        Current mastery level: {domain.mastery_level:.1%}
        
        The challenge should:
        1. Be specific and actionable
        2. Test practical skills
        3. Be appropriate for current skill level
        4. Allow for measurable success/failure
        
        Describe the challenge clearly.
        """
        
        creation, data = self.rivir.collapse_and_create(challenge_prompt)
        
        # Extract the main challenge description
        lines = creation.split('\n')
        challenge = ""
        for line in lines:
            line = line.strip()
            if line and len(line) > 20 and not line.startswith('Challenge:'):
                challenge = line
                break
        
        return challenge or f"Practice core concepts of {domain.name}"
    
    def _attempt_challenge(self, domain: ExpertiseDomain, challenge: str) -> Tuple[str, str]:
        """Attempt to solve a practice challenge."""
        attempt_prompt = f"""
        Solve this {domain.name} challenge using your current expertise:
        
        Challenge: {challenge}
        
        Available knowledge:
        - Components: {', '.join(domain.components)}
        - Strengths: {', '.join(domain.strengths)}
        - Learning path: {', '.join(domain.learning_path[:5])}
        
        Provide:
        1. Your approach/strategy
        2. Step-by-step solution
        3. Expected outcome
        
        Be specific and practical.
        """
        
        creation, data = self.rivir.collapse_and_create(attempt_prompt)
        
        # Split into approach and outcome
        parts = creation.split('\n\n')
        if len(parts) >= 2:
            approach = parts[0]
            outcome = '\n\n'.join(parts[1:])
        else:
            approach = creation[:len(creation)//2]
            outcome = creation[len(creation)//2:]
        
        return approach.strip(), outcome.strip()
    
    def _evaluate_attempt(self, domain: ExpertiseDomain, challenge: str, approach: str, outcome: str) -> Tuple[bool, str]:
        """Evaluate the success of a challenge attempt."""
        eval_prompt = f"""
        Evaluate this attempt at a {domain.name} challenge:
        
        Challenge: {challenge}
        Approach: {approach}
        Outcome: {outcome}
        
        Rate the attempt:
        1. Is the approach sound? (technical correctness)
        2. Is the solution practical? (real-world applicability)  
        3. Does it demonstrate expertise? (depth of understanding)
        
        Provide:
        - SUCCESS or FAILURE
        - Specific feedback for improvement
        """
        
        creation, data = self.rivir.collapse_and_create(eval_prompt)
        
        # Determine success
        success = 'SUCCESS' in creation.upper() or 'SUCCESSFUL' in creation.upper()
        
        # Extract feedback
        feedback = creation.replace('SUCCESS', '').replace('FAILURE', '').strip()
        
        return success, feedback
    
    def _extract_lessons(self, domain: ExpertiseDomain, challenge: str, approach: str, 
                        outcome: str, success: bool, feedback: str) -> List[str]:
        """Extract lessons learned from a practice session."""
        lessons_prompt = f"""
        Extract key lessons from this {domain.name} practice session:
        
        Challenge: {challenge}
        Success: {success}
        Feedback: {feedback}
        
        What can be learned to improve future performance?
        List specific, actionable insights.
        """
        
        creation, data = self.rivir.collapse_and_create(lessons_prompt)
        
        # Extract lessons as list
        lessons = []
        lines = creation.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('•') or line.startswith('*')):
                lesson = line.lstrip('-•* ').strip()
                if lesson and len(lesson) > 10:
                    lessons.append(lesson)
        
        return lessons[:5]  # Top 5 lessons
    
    def _update_mastery_level(self, domain: ExpertiseDomain, recent_sessions: List[PracticeSession]):
        """Update mastery level based on recent performance."""
        if not recent_sessions:
            return
        
        # Calculate success rate from recent sessions
        successes = sum(1 for s in recent_sessions if s.success)
        recent_success_rate = successes / len(recent_sessions)
        
        # Update overall success rate (weighted average)
        if domain.practice_count > len(recent_sessions):
            weight = 0.3  # Weight recent sessions at 30%
            domain.success_rate = (domain.success_rate * (1 - weight) + 
                                 recent_success_rate * weight)
        else:
            domain.success_rate = recent_success_rate
        
        # Update mastery level based on success rate and practice count
        practice_factor = min(1.0, domain.practice_count / 50)  # Max benefit at 50 practices
        domain.mastery_level = domain.success_rate * practice_factor
        
        # Bonus for consistency
        if domain.practice_count > 10 and domain.success_rate > 0.8:
            domain.mastery_level = min(1.0, domain.mastery_level * 1.1)
    
    def _adapt_learning_path(self, domain: ExpertiseDomain, recent_sessions: List[PracticeSession]):
        """Adapt learning path based on performance patterns."""
        if not recent_sessions:
            return
        
        # Identify weak areas from failed sessions
        weak_areas = []
        strong_areas = []
        
        for session in recent_sessions:
            if not session.success:
                # Extract topics from challenge
                challenge_words = session.challenge.lower().split()
                for component in domain.components:
                    if any(word in component.lower() for word in challenge_words):
                        weak_areas.append(component)
            else:
                # Extract topics from successful challenges
                challenge_words = session.challenge.lower().split()
                for component in domain.components:
                    if any(word in component.lower() for word in challenge_words):
                        strong_areas.append(component)
        
        # Update strengths and weaknesses
        domain.weaknesses = list(set(weak_areas))[:5]
        domain.strengths = list(set(strong_areas))[:5]
    
    # -------------------------------------------------------------------------
    # EXPERTISE QUERIES
    # -------------------------------------------------------------------------
    
    def get_expertise_level(self, domain_name: str) -> float:
        """Get current expertise level for a domain (0.0 to 1.0)."""
        if domain_name not in self.domains:
            return 0.0
        return self.domains[domain_name].mastery_level
    
    def list_domains(self) -> List[Dict[str, Any]]:
        """List all domains and their expertise levels."""
        return [
            {
                'name': domain.name,
                'mastery_level': domain.mastery_level,
                'practice_count': domain.practice_count,
                'success_rate': domain.success_rate,
                'components_count': len(domain.components)
            }
            for domain in self.domains.values()
        ]
    
    def get_domain_status(self, domain_name: str) -> Dict[str, Any]:
        """Get detailed status for a specific domain."""
        if domain_name not in self.domains:
            return {}
        
        domain = self.domains[domain_name]
        recent_sessions = [s for s in self.practice_history 
                          if s.domain == domain_name and s.timestamp > time.time() - 86400]
        
        return {
            'domain': domain.to_dict(),
            'recent_practice_count': len(recent_sessions),
            'recent_success_rate': sum(1 for s in recent_sessions if s.success) / len(recent_sessions) if recent_sessions else 0,
            'next_focus_areas': domain.weaknesses[:3] if domain.weaknesses else domain.components[:3],
            'mastery_level_text': self._mastery_level_text(domain.mastery_level)
        }
    
    def _mastery_level_text(self, level: float) -> str:
        """Convert mastery level to descriptive text."""
        if level < 0.2:
            return "Beginner"
        elif level < 0.4:
            return "Novice"
        elif level < 0.6:
            return "Intermediate"
        elif level < 0.8:
            return "Advanced"
        else:
            return "Expert"
    
    def close(self):
        """Clean shutdown."""
        self.conn.close()
        self.rivir.close()


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Interactive CLI for RIVIR expertise development."""
    print("🎓 RIVIR Expertise Development System")
    print("Tell RIVIR to 'go get good at something' and watch it learn!")
    print("Type 'help' for commands, 'quit' to exit\n")
    
    expert = RIVIRExpert()
    
    try:
        while True:
            try:
                user_input = input("Expert> ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if user_input.lower() == 'help':
                    print("""
Commands:
  learn <domain>           - Start learning a new domain
  practice <domain> [n]    - Practice domain (n iterations, default 5)
  status <domain>          - Show detailed domain status
  list                     - List all domains and expertise levels
  expertise <domain>       - Get expertise level for domain
  help                     - Show this help
  quit                     - Exit

Examples:
  Expert> learn web development
  Expert> learn meditation app design
  Expert> practice web development 10
  Expert> status web development
                    """)
                    continue
                    
                if user_input.lower() == 'list':
                    domains = expert.list_domains()
                    if domains:
                        print("\n📚 Current Domains:")
                        for domain in domains:
                            mastery_text = expert._mastery_level_text(domain['mastery_level'])
                            print(f"  {domain['name']}: {domain['mastery_level']:.1%} ({mastery_text})")
                            print(f"    Practice: {domain['practice_count']} sessions, {domain['success_rate']:.1%} success")
                    else:
                        print("No domains learned yet. Use 'learn <domain>' to start!")
                    continue
                
                parts = user_input.split()
                command = parts[0].lower()
                
                if command == 'learn' and len(parts) > 1:
                    domain_name = ' '.join(parts[1:])
                    domain = expert.learn_domain(domain_name)
                    
                elif command == 'practice' and len(parts) > 1:
                    domain_name = ' '.join(parts[1:-1]) if len(parts) > 2 and parts[-1].isdigit() else ' '.join(parts[1:])
                    iterations = int(parts[-1]) if len(parts) > 2 and parts[-1].isdigit() else 5
                    
                    try:
                        sessions = expert.practice_domain(domain_name, iterations)
                        successes = sum(1 for s in sessions if s.success)
                        print(f"📈 Practice results: {successes}/{len(sessions)} successful")
                    except ValueError as e:
                        print(f"❌ {e}")
                
                elif command == 'status' and len(parts) > 1:
                    domain_name = ' '.join(parts[1:])
                    status = expert.get_domain_status(domain_name)
                    if status:
                        domain = status['domain']
                        print(f"\n📊 {domain['name']} Status:")
                        print(f"  Mastery: {domain['mastery_level']:.1%} ({status['mastery_level_text']})")
                        print(f"  Practice: {domain['practice_count']} sessions, {domain['success_rate']:.1%} success")
                        print(f"  Components: {len(domain['components'])}")
                        print(f"  Strengths: {', '.join(domain['strengths'][:3])}")
                        print(f"  Focus areas: {', '.join(status['next_focus_areas'])}")
                    else:
                        print(f"❌ Domain '{domain_name}' not found")
                
                elif command == 'expertise' and len(parts) > 1:
                    domain_name = ' '.join(parts[1:])
                    level = expert.get_expertise_level(domain_name)
                    mastery_text = expert._mastery_level_text(level)
                    print(f"{domain_name}: {level:.1%} ({mastery_text})")
                
                else:
                    print("❓ Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted")
                break
            except Exception as e:
                print(f"Error: {e}")
                
    finally:
        print("\n🎓 Closing expertise system...")
        expert.close()
        print("All learning progress saved. Goodbye!")


if __name__ == "__main__":
    main()