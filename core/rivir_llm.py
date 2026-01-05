#!/usr/bin/env python3
"""
RIVIR LLM Interface - Plug RIVIR into any LLM system

This makes RIVIR appear as a standard language model that can be used with:
- mlx_lm
- OpenAI Codex
- Hugging Face transformers
- Any system expecting an LLM interface

Internally it uses:
- Quantum affordances (superposition → collapse)
- Deep listening (surface → sacred drive)
- White hat guru (ethical amplification)
- Badass shapester (quality generation)
- Receipt chain (learning from feedback)

Usage:
    from rivir_llm import RIVIRModel
    
    model = RIVIRModel()
    response = model.generate("create a todo app", max_length=1000)
    model.learn_from_feedback(response_id, satisfaction=0.9)
"""

import json
import time
import hashlib
from typing import Dict, List, Optional, Union, Any, Iterator
from dataclasses import dataclass
import threading
import queue

try:
    from rivir_direct import RIVIRDirect, WaveformState
except ImportError:
    from .rivir_direct import RIVIRDirect, WaveformState


# =============================================================================
# LLM-COMPATIBLE INTERFACES
# =============================================================================

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_length: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    use_cache: bool = True
    quantum_seed: Optional[int] = None
    amplify_quality: bool = True
    deep_listen: bool = True
    white_hat_check: bool = True


@dataclass
class GenerationOutput:
    """Output from RIVIR generation."""
    text: str
    response_id: str
    metadata: Dict[str, Any]
    listening_data: Dict[str, Any]
    alignment_score: float
    possibilities_explored: int
    chosen_path: str
    receipt_id: str


class RIVIRTokenizer:
    """Minimal tokenizer interface for compatibility."""
    
    def __init__(self):
        self.vocab_size = 50000  # Fake vocab size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def encode(self, text: str, **kwargs) -> List[int]:
        """Encode text to token IDs (fake implementation)."""
        # Simple hash-based encoding for compatibility
        tokens = []
        for word in text.split():
            token_id = hash(word) % self.vocab_size
            tokens.append(abs(token_id))
        return tokens
    
    def decode(self, token_ids: List[int], **kwargs) -> str:
        """Decode token IDs to text (fake implementation)."""
        # This is just for interface compatibility
        return f"[RIVIR_DECODED_{len(token_ids)}_TOKENS]"
    
    def __call__(self, text: Union[str, List[str]], **kwargs):
        """Tokenizer call interface."""
        if isinstance(text, str):
            return {"input_ids": [self.encode(text)]}
        else:
            return {"input_ids": [self.encode(t) for t in text]}


class RIVIRModel:
    """
    RIVIR as an LLM-compatible model.
    
    This class provides standard LLM interfaces while internally using
    the full RIVIR system with quantum affordances and ethical checks.
    """
    
    def __init__(self, db_prefix: str = "rivir_llm", **kwargs):
        # Core RIVIR system
        self.rivir = RIVIRDirect(db_prefix)
        
        # LLM compatibility
        self.tokenizer = RIVIRTokenizer()
        self.config = GenerationConfig()
        self.device = "cpu"  # RIVIR runs on CPU
        
        # Generation tracking
        self.generation_history = {}
        self.response_counter = 0
        
        # Model metadata for compatibility
        self.model_name = "RIVIR-Direct-v1"
        self.model_type = "rivir"
        
    @property
    def vocab_size(self):
        """Vocabulary size for compatibility."""
        return self.tokenizer.vocab_size
    
    def generate(
        self,
        inputs: Union[str, List[str], Dict],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Union[GenerationOutput, List[GenerationOutput]]:
        """
        Generate text using RIVIR system.
        
        This is the main interface that other systems will call.
        """
        # Handle different input formats
        if isinstance(inputs, dict):
            if "input_ids" in inputs:
                # Decode from token IDs (fake)
                prompt = f"[DECODED_FROM_{len(inputs['input_ids'])}_TOKENS]"
            elif "inputs" in inputs:
                prompt = inputs["inputs"]
            else:
                prompt = str(inputs)
        elif isinstance(inputs, list):
            # Batch generation
            return [self.generate(inp, generation_config, **kwargs) for inp in inputs]
        else:
            prompt = str(inputs)
        
        # Merge configs
        config = generation_config or self.config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        # Generate using RIVIR
        start_time = time.time()
        
        try:
            # Use RIVIR's collapse_and_create method
            creation, data = self.rivir.collapse_and_create(
                prompt, 
                seed=config.quantum_seed
            )
            
            if data.get('declined'):
                # Handle declined generation
                creation = f"[DECLINED: {data.get('reason', 'Alignment check failed')}]"
                alignment_score = data.get('alignment', 0.0)
                listening_data = data.get('listening', {})
                possibilities_explored = 0
                chosen_path = "declined"
                receipt_id = "declined"
            else:
                alignment_score = data['alignment']
                listening_data = data['listening']
                possibilities_explored = data['total_possibilities']
                chosen_path = data['chosen_possibility']
                receipt_id = data['receipt_id']
            
            # Truncate to max_length if needed
            if len(creation) > config.max_length:
                creation = creation[:config.max_length] + "..."
            
            # Create response ID
            self.response_counter += 1
            response_id = f"rivir_{self.response_counter}_{int(time.time())}"
            
            # Create output
            output = GenerationOutput(
                text=creation,
                response_id=response_id,
                metadata={
                    'generation_time': time.time() - start_time,
                    'model_name': self.model_name,
                    'config': config.__dict__,
                    'prompt': prompt,
                    'amplified_intent': data.get('amplified_intent', ''),
                    'stayed_hidden': data.get('stayed_hidden', True)
                },
                listening_data=listening_data,
                alignment_score=alignment_score,
                possibilities_explored=possibilities_explored,
                chosen_path=chosen_path,
                receipt_id=receipt_id
            )
            
            # Store for feedback
            self.generation_history[response_id] = {
                'output': output,
                'receipt_id': receipt_id,
                'timestamp': time.time()
            }
            
            return output
            
        except Exception as e:
            # Fallback generation
            fallback_text = f"[RIVIR_ERROR: {str(e)}]"
            response_id = f"error_{self.response_counter}_{int(time.time())}"
            
            return GenerationOutput(
                text=fallback_text,
                response_id=response_id,
                metadata={'error': str(e), 'generation_time': time.time() - start_time},
                listening_data={},
                alignment_score=0.0,
                possibilities_explored=0,
                chosen_path="error",
                receipt_id="error"
            )
    
    def __call__(self, *args, **kwargs):
        """Make model callable like standard LLMs."""
        return self.generate(*args, **kwargs)
    
    def learn_from_feedback(
        self, 
        response_id: str, 
        satisfaction: float, 
        notes: str = ""
    ):
        """
        Learn from user feedback on generated responses.
        
        This is how the system improves over time.
        """
        if response_id in self.generation_history:
            history_item = self.generation_history[response_id]
            receipt_id = history_item['receipt_id']
            
            if receipt_id != "declined" and receipt_id != "error":
                self.rivir.learn_from_feedback(receipt_id, satisfaction, notes)
                
                # Update history
                history_item['feedback'] = {
                    'satisfaction': satisfaction,
                    'notes': notes,
                    'timestamp': time.time()
                }
                
                return True
        return False
    
    def batch_generate(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> List[GenerationOutput]:
        """Generate responses for multiple prompts."""
        return [self.generate(prompt, generation_config, **kwargs) for prompt in prompts]
    
    def stream_generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs
    ) -> Iterator[str]:
        """
        Stream generation (simulated for RIVIR).
        
        RIVIR generates complete responses, so we simulate streaming
        by yielding chunks of the final response.
        """
        output = self.generate(prompt, generation_config, **kwargs)
        text = output.text
        
        # Simulate streaming by yielding chunks
        chunk_size = max(1, len(text) // 20)  # 20 chunks
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield chunk
            time.sleep(0.05)  # Simulate generation delay
    
    def get_status(self) -> Dict[str, Any]:
        """Get model status and statistics."""
        rivir_status = self.rivir.status()
        
        return {
            'model_name': self.model_name,
            'model_type': self.model_type,
            'generations_count': self.response_counter,
            'avg_alignment': rivir_status['avg_alignment'],
            'avg_satisfaction': rivir_status['avg_satisfaction'],
            'session_duration': rivir_status['session_duration'],
            'receipts_created': rivir_status['receipts_created'],
            'has_quantum': rivir_status['has_quantum'],
            'feedback_count': len([h for h in self.generation_history.values() if 'feedback' in h])
        }
    
    def save_pretrained(self, path: str):
        """Save model state (for compatibility)."""
        import os
        os.makedirs(path, exist_ok=True)
        
        # Save generation history and config
        with open(f"{path}/generation_history.json", 'w') as f:
            # Convert to serializable format
            history = {}
            for k, v in self.generation_history.items():
                history[k] = {
                    'receipt_id': v['receipt_id'],
                    'timestamp': v['timestamp'],
                    'feedback': v.get('feedback', {})
                }
            json.dump(history, f, indent=2)
        
        # Save config
        with open(f"{path}/config.json", 'w') as f:
            json.dump({
                'model_name': self.model_name,
                'model_type': self.model_type,
                'response_counter': self.response_counter
            }, f, indent=2)
        
        print(f"RIVIR model saved to {path}")
    
    @classmethod
    def from_pretrained(cls, path: str, **kwargs):
        """Load model state (for compatibility)."""
        model = cls(**kwargs)
        
        try:
            # Load generation history
            with open(f"{path}/generation_history.json", 'r') as f:
                history = json.load(f)
                # Note: We only load metadata, not full outputs
                for k, v in history.items():
                    model.generation_history[k] = v
            
            # Load config
            with open(f"{path}/config.json", 'r') as f:
                config = json.load(f)
                model.model_name = config.get('model_name', model.model_name)
                model.response_counter = config.get('response_counter', 0)
            
            print(f"RIVIR model loaded from {path}")
        except FileNotFoundError:
            print(f"No saved model found at {path}, using fresh model")
        
        return model
    
    def close(self):
        """Clean shutdown."""
        self.rivir.close()


# =============================================================================
# MLX_LM COMPATIBILITY LAYER
# =============================================================================

class RIVIRForMLX:
    """
    RIVIR wrapper for mlx_lm compatibility.
    
    Usage:
        from rivir_llm import RIVIRForMLX
        model, tokenizer = RIVIRForMLX.from_pretrained("rivir-direct")
    """
    
    def __init__(self, rivir_model: RIVIRModel):
        self.model = rivir_model
        self.tokenizer = rivir_model.tokenizer
    
    @classmethod
    def from_pretrained(cls, model_name: str = "rivir-direct", **kwargs):
        """Load RIVIR model in mlx_lm format."""
        rivir_model = RIVIRModel(**kwargs)
        wrapper = cls(rivir_model)
        return wrapper.model, wrapper.tokenizer
    
    def generate(self, *args, **kwargs):
        """Generate method for mlx_lm compatibility."""
        return self.model.generate(*args, **kwargs)


# =============================================================================
# OPENAI API COMPATIBILITY
# =============================================================================

class RIVIROpenAICompat:
    """
    OpenAI API compatible interface for RIVIR.
    
    Usage:
        client = RIVIROpenAICompat()
        response = client.chat.completions.create(
            model="rivir-direct",
            messages=[{"role": "user", "content": "create a todo app"}]
        )
    """
    
    def __init__(self):
        self.model = RIVIRModel()
        self.chat = self.ChatCompletions(self.model)
    
    class ChatCompletions:
        def __init__(self, model: RIVIRModel):
            self.model = model
        
        def create(
            self,
            model: str = "rivir-direct",
            messages: List[Dict[str, str]] = None,
            max_tokens: int = 1000,
            temperature: float = 0.7,
            **kwargs
        ):
            """Create chat completion using RIVIR."""
            # Extract prompt from messages
            if messages:
                prompt = messages[-1].get("content", "")
            else:
                prompt = kwargs.get("prompt", "")
            
            # Generate using RIVIR
            config = GenerationConfig(
                max_length=max_tokens,
                temperature=temperature
            )
            
            output = self.model.generate(prompt, config)
            
            # Format as OpenAI response
            return {
                "id": output.response_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": output.text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(output.text.split()),
                    "total_tokens": len(prompt.split()) + len(output.text.split())
                },
                "rivir_metadata": {
                    "alignment_score": output.alignment_score,
                    "listening_data": output.listening_data,
                    "possibilities_explored": output.possibilities_explored,
                    "receipt_id": output.receipt_id
                }
            }


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

def example_basic_usage():
    """Basic RIVIR LLM usage example."""
    print("=== Basic RIVIR LLM Usage ===")
    
    # Initialize model
    model = RIVIRModel()
    
    # Generate response
    output = model.generate("create a beautiful todo app with calm colors")
    
    print(f"Generated: {output.text[:100]}...")
    print(f"Alignment: {output.alignment_score:.2f}")
    print(f"Listening: {output.listening_data.get('deeper_need', 'N/A')}")
    
    # Provide feedback
    model.learn_from_feedback(output.response_id, satisfaction=0.9, notes="Great!")
    
    # Check status
    status = model.get_status()
    print(f"Status: {status['generations_count']} generations, {status['avg_satisfaction']:.2f} avg satisfaction")
    
    model.close()


def example_mlx_usage():
    """Example using mlx_lm compatible interface."""
    print("=== MLX_LM Compatible Usage ===")
    
    # Load model in mlx format
    model, tokenizer = RIVIRForMLX.from_pretrained("rivir-direct")
    
    # Generate
    output = model.generate("build a meditation app")
    print(f"MLX Generated: {output.text[:100]}...")
    
    model.close()


def example_openai_usage():
    """Example using OpenAI API compatible interface."""
    print("=== OpenAI API Compatible Usage ===")
    
    # Initialize client
    client = RIVIROpenAICompat()
    
    # Chat completion
    response = client.chat.completions.create(
        model="rivir-direct",
        messages=[
            {"role": "user", "content": "create a mindful task manager"}
        ],
        max_tokens=500
    )
    
    print(f"OpenAI Format: {response['choices'][0]['message']['content'][:100]}...")
    print(f"RIVIR Alignment: {response['rivir_metadata']['alignment_score']:.2f}")


if __name__ == "__main__":
    example_basic_usage()
    print()
    example_mlx_usage()
    print()
    example_openai_usage()