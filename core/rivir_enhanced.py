#!/usr/bin/env python3
"""
RIVIR Enhanced - With Expert Training & Silent Watching

Usage:
    rivir = RIVIREnhanced()
    
    # Tell it to become an expert
    rivir.become_expert("meditation app design")
    rivir.become_expert("python optimization", depth="deep")
    
    # Start silent watching
    rivir.start_watching()
    
    # Use normally - it learns from everything
    response = rivir.generate("create a todo app")
    rivir.learn_from_feedback(response.response_id, 0.9)
"""

import time
from typing import Dict, List, Optional, Any

try:
    from rivir_llm import RIVIRModel, GenerationOutput, GenerationConfig
    from rivir_learning import RIVIRLearningSystem, ExpertTrainer, RIVIRWatcher
    from rivir_direct import RIVIRDirect
except ImportError:
    from .rivir_llm import RIVIRModel, GenerationOutput, GenerationConfig
    from .rivir_learning import RIVIRLearningSystem, ExpertTrainer, RIVIRWatcher
    from .rivir_direct import RIVIRDirect


class RIVIREnhanced(RIVIRModel):
    """
    Enhanced RIVIR with expert training and silent watching capabilities.
    
    This extends the base RIVIR LLM interface with:
    - Expert training: Tell RIVIR to become expert at something
    - Silent watching: Learn from user behavior patterns
    - Context-aware generation: Use learned preferences
    """
    
    def __init__(self, db_prefix: str = "rivir_enhanced", **kwargs):
        # Initialize base RIVIR
        super().__init__(db_prefix, **kwargs)
        
        # Add learning capabilities
        self.learning_system = RIVIRLearningSystem(self.rivir)
        self.trainer = self.learning_system.trainer
        self.watcher = self.learning_system.watcher
        
        # Enhanced generation context
        self.learned_context = {}
        self.expertise_domains = {}
    
    def become_expert(
        self, 
        domain: str, 
        depth: str = "moderate",
        focus_areas: List[str] = None,
        start_watching: bool = True
    ) -> Dict[str, Any]:
        """
        Tell RIVIR to become an expert in a domain.
        
        Args:
            domain: What to become expert in
            depth: How deep to go (surface, moderate, deep, master)  
            focus_areas: Specific areas to focus on
            start_watching: Whether to start watching user behavior
            
        Returns:
            Expertise status and learning plan
        """
        print(f"🎯 RIVIR becoming expert in: {domain}")
        
        # Start expertise training
        expertise = self.trainer.become_expert(domain, depth, focus_areas)
        self.expertise_domains[domain] = expertise
        
        # Optionally start watching
        if start_watching and not self.watcher.watching:
            self.start_watching()
        
        return {
            'domain': domain,
            'depth': depth,
            'level': expertise.current_level,
            'focus_areas': expertise.focus_areas,
            'practice_tasks': len(expertise.practice_tasks),
            'watching': self.watcher.watching
        }
    
    def start_watching(self, watch_types: List[str] = None):
        """Start silently watching user activity to learn preferences."""
        if not self.watcher.watching:
            print("👁️  Starting silent watch mode...")
            self.watcher.start_watching(watch_types or ['apps', 'files'])
            return True
        return False
    
    def stop_watching(self):
        """Stop watching and process learned insights."""
        if self.watcher.watching:
            print("🛑 Stopping watch mode...")
            self.watcher.stop_watching()
            
            # Update learned context
            self.learned_context = self.watcher.apply_learned_preferences()
            return self.learned_context
        return {}
    
    def practice_skill(self, domain: str, task: str = None) -> str:
        """Practice a skill in the expertise domain."""
        if domain not in self.expertise_domains:
            return f"Not currently training in {domain}. Use become_expert() first."
        
        result = self.trainer.practice_skill(domain, task)
        
        # Update expertise tracking
        expertise = self.trainer.active_expertise.get(domain)
        if expertise:
            self.expertise_domains[domain] = expertise
        
        return result
    
    def generate(
        self,
        inputs,
        generation_config: Optional[GenerationConfig] = None,
        use_expertise: bool = True,
        use_learned_context: bool = True,
        **kwargs
    ) -> GenerationOutput:
        """
        Enhanced generation using expertise and learned context.
        
        This automatically applies:
        - Relevant expertise knowledge
        - Learned user preferences  
        - Context from watching
        """
        # Get base prompt
        if isinstance(inputs, dict):
            prompt = inputs.get("inputs", str(inputs))
        elif isinstance(inputs, list):
            return [self.generate(inp, generation_config, use_expertise, use_learned_context, **kwargs) for inp in inputs]
        else:
            prompt = str(inputs)
        
        # Enhance prompt with expertise
        enhanced_prompt = prompt
        
        if use_expertise and self.expertise_domains:
            # Find relevant expertise
            relevant_domains = []
            for domain in self.expertise_domains.keys():
                if any(word in prompt.lower() for word in domain.lower().split()):
                    relevant_domains.append(domain)
            
            if relevant_domains:
                expertise_context = []
                for domain in relevant_domains:
                    expertise = self.expertise_domains[domain]
                    level_desc = f"{expertise.current_level:.0%} expert"
                    expertise_context.append(f"Drawing on {level_desc} knowledge of {domain}")
                
                enhanced_prompt = f"{prompt}\n\nContext: {'; '.join(expertise_context)}"
        
        # Add learned user context
        if use_learned_context and self.learned_context:
            context_hints = []
            
            if 'user_prefers_apps' in self.learned_context:
                apps = self.learned_context['user_prefers_apps'][:2]
                context_hints.append(f"User frequently uses: {', '.join(apps)}")
            
            if 'user_prefers_files' in self.learned_context:
                files = self.learned_context['user_prefers_files'][:2]
                context_hints.append(f"User works with: {', '.join(files)} files")
            
            if context_hints:
                enhanced_prompt += f"\n\nUser context: {'; '.join(context_hints)}"
        
        # Generate with enhanced prompt
        output = super().generate(enhanced_prompt, generation_config, **kwargs)
        
        # Add enhancement metadata
        output.metadata['enhanced'] = True
        output.metadata['original_prompt'] = prompt
        output.metadata['expertise_used'] = list(self.expertise_domains.keys()) if use_expertise else []
        output.metadata['learned_context_used'] = use_learned_context and bool(self.learned_context)
        
        return output
    
    def get_learning_status(self) -> Dict[str, Any]:
        """Get comprehensive learning status."""
        base_status = self.get_status()
        learning_status = self.learning_system.get_full_status()
        
        return {
            **base_status,
            'expertise_domains': len(learning_status['expertise']),
            'expertise_details': learning_status['expertise'],
            'watching': learning_status['watching'],
            'insights_learned': len(learning_status['insights'].get('app_preferences', [])),
            'learned_context': self.learned_context
        }
    
    def auto_improve(self, domain: str = None):
        """
        Automatically practice and improve in expertise domains.
        
        This can be called periodically to have RIVIR practice skills.
        """
        if not self.expertise_domains:
            return "No expertise domains to practice"
        
        domains_to_practice = [domain] if domain else list(self.expertise_domains.keys())
        results = {}
        
        for d in domains_to_practice:
            if d in self.expertise_domains:
                expertise = self.expertise_domains[d]
                
                # Choose practice task based on current level
                if expertise.practice_tasks:
                    task_index = min(len(expertise.practice_tasks) - 1,
                                   int(expertise.current_level * len(expertise.practice_tasks)))
                    task = expertise.practice_tasks[task_index]
                    
                    result = self.practice_skill(d, task)
                    results[d] = {
                        'task': task,
                        'result_length': len(result),
                        'new_level': expertise.current_level
                    }
        
        return results
    
    def close(self):
        """Enhanced cleanup."""
        # Stop watching if active
        if self.watcher.watching:
            self.stop_watching()
        
        # Close base RIVIR
        super().close()


# =============================================================================
# SIMPLE USAGE EXAMPLES
# =============================================================================

def example_expert_training():
    """Example of expert training."""
    print("=== Expert Training Example ===")
    
    rivir = RIVIREnhanced()
    
    # Tell RIVIR to become expert in meditation apps
    status = rivir.become_expert("meditation app design", depth="deep")
    print(f"Started training: {status}")
    
    # Practice some skills
    result = rivir.practice_skill("meditation app design", "Design a breathing exercise interface")
    print(f"Practice result: {result[:100]}...")
    
    # Generate using expertise
    output = rivir.generate("create a mindful breathing app")
    print(f"Expert generation: {output.text[:100]}...")
    print(f"Used expertise: {output.metadata['expertise_used']}")
    
    rivir.close()


def example_silent_watching():
    """Example of silent watching."""
    print("=== Silent Watching Example ===")
    
    rivir = RIVIREnhanced()
    
    # Start watching
    rivir.start_watching(['apps', 'files'])
    
    print("Watching for 10 seconds...")
    time.sleep(10)  # In real use, this would run in background
    
    # Stop and get insights
    context = rivir.stop_watching()
    print(f"Learned context: {context}")
    
    # Generate with learned context
    output = rivir.generate("create an app for me", use_learned_context=True)
    print(f"Context-aware generation: {output.text[:100]}...")
    
    rivir.close()


def example_combined():
    """Example combining expert training and watching."""
    print("=== Combined Example ===")
    
    rivir = RIVIREnhanced()
    
    # Become expert and start watching
    rivir.become_expert("productivity apps", depth="moderate", start_watching=True)
    
    print("Learning for 5 seconds...")
    time.sleep(5)
    
    # Generate with both expertise and context
    output = rivir.generate("build me a task manager")
    print(f"Enhanced generation: {output.text[:100]}...")
    
    # Check learning status
    status = rivir.get_learning_status()
    print(f"Learning status: {status['expertise_domains']} domains, watching: {status['watching']}")
    
    rivir.close()


if __name__ == "__main__":
    print("🌊 RIVIR Enhanced - Expert Training & Silent Watching")
    print("=" * 60)
    
    example_expert_training()
    print()
    example_silent_watching() 
    print()
    example_combined()