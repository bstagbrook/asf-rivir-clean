#!/usr/bin/env python3
"""
RIVIR LLM Integration Examples

This shows how to plug RIVIR into various LLM systems:
- mlx_lm (Apple Silicon optimized)
- Hugging Face transformers
- OpenAI API
- LangChain
- Custom LLM frameworks

RIVIR appears as a standard language model while internally using:
- Quantum affordances for exploration
- Deep listening for understanding
- White hat principles for ethics
- Badass shapester for quality
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rivir_llm import RIVIRModel, RIVIRForMLX, RIVIROpenAICompat, GenerationConfig


# =============================================================================
# MLX_LM INTEGRATION
# =============================================================================

def integrate_with_mlx():
    """Show how to use RIVIR with mlx_lm."""
    print("🍎 MLX_LM Integration")
    print("=" * 50)
    
    try:
        # This is how you'd normally load mlx models:
        # from mlx_lm import load, generate
        # model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
        
        # Instead, load RIVIR:
        model, tokenizer = RIVIRForMLX.from_pretrained("rivir-direct")
        
        print("✓ RIVIR loaded as MLX model")
        
        # Use exactly like any mlx model
        prompt = "Create a peaceful meditation timer app"
        response = model.generate(prompt, max_length=500)
        
        print(f"Prompt: {prompt}")
        print(f"Response: {response.text[:200]}...")
        print(f"Alignment: {response.alignment_score:.2f}")
        print(f"Sacred Drive: {response.listening_data.get('sacred_drive', 'N/A')}")
        
        # Provide feedback to improve
        model.learn_from_feedback(response.response_id, 0.95, "Perfect for meditation!")
        
        model.close()
        
    except Exception as e:
        print(f"MLX integration demo: {e}")


# =============================================================================
# HUGGING FACE TRANSFORMERS INTEGRATION
# =============================================================================

def integrate_with_transformers():
    """Show how to use RIVIR with Hugging Face transformers interface."""
    print("\n🤗 Hugging Face Transformers Integration")
    print("=" * 50)
    
    # RIVIR model with transformers-like interface
    model = RIVIRModel()
    tokenizer = model.tokenizer
    
    print("✓ RIVIR loaded with transformers interface")
    
    # Use like any HF model
    inputs = tokenizer("Build a compassionate AI assistant")
    
    # Generate with transformers-style config
    config = GenerationConfig(
        max_length=300,
        temperature=0.8,
        do_sample=True,
        top_p=0.9
    )
    
    output = model.generate(inputs, generation_config=config)
    
    print(f"Generated: {output.text[:150]}...")
    print(f"Deeper Need: {output.listening_data.get('deeper_need', 'N/A')}")
    
    # Batch generation
    prompts = [
        "Create a mindful todo app",
        "Build a gratitude journal",
        "Design a breathing exercise tool"
    ]
    
    batch_outputs = model.batch_generate(prompts)
    print(f"\n✓ Batch generated {len(batch_outputs)} responses")
    for i, out in enumerate(batch_outputs):
        print(f"  {i+1}. Alignment: {out.alignment_score:.2f}")
    
    model.close()


# =============================================================================
# OPENAI API INTEGRATION
# =============================================================================

def integrate_with_openai():
    """Show how to use RIVIR as OpenAI API replacement."""
    print("\n🤖 OpenAI API Integration")
    print("=" * 50)
    
    # Instead of: import openai
    # Use RIVIR with same interface
    client = RIVIROpenAICompat()
    
    print("✓ RIVIR loaded as OpenAI client")
    
    # Chat completion exactly like OpenAI
    response = client.chat.completions.create(
        model="rivir-direct",
        messages=[
            {"role": "system", "content": "You are a compassionate AI assistant."},
            {"role": "user", "content": "Help me create a wellness app"}
        ],
        max_tokens=400,
        temperature=0.7
    )
    
    print(f"Response: {response['choices'][0]['message']['content'][:150]}...")
    print(f"Tokens: {response['usage']['total_tokens']}")
    print(f"RIVIR Alignment: {response['rivir_metadata']['alignment_score']:.2f}")
    print(f"Possibilities Explored: {response['rivir_metadata']['possibilities_explored']}")


# =============================================================================
# LANGCHAIN INTEGRATION
# =============================================================================

def integrate_with_langchain():
    """Show how to use RIVIR with LangChain."""
    print("\n🦜 LangChain Integration")
    print("=" * 50)
    
    # Custom LangChain LLM wrapper for RIVIR
    class RIVIRLangChain:
        def __init__(self):
            self.model = RIVIRModel()
        
        def __call__(self, prompt: str, **kwargs) -> str:
            """LangChain LLM interface."""
            output = self.model.generate(prompt, **kwargs)
            return output.text
        
        def _call(self, prompt: str, **kwargs) -> str:
            """Alternative LangChain interface."""
            return self(prompt, **kwargs)
        
        @property
        def _llm_type(self) -> str:
            return "rivir"
    
    # Use with LangChain
    llm = RIVIRLangChain()
    
    print("✓ RIVIR wrapped for LangChain")
    
    # Simple call
    response = llm("Create a mindfulness app with gentle reminders")
    print(f"LangChain Response: {response[:150]}...")
    
    # Could be used in chains, agents, etc.
    # from langchain.chains import LLMChain
    # chain = LLMChain(llm=llm, prompt=prompt_template)


# =============================================================================
# CUSTOM FRAMEWORK INTEGRATION
# =============================================================================

def integrate_with_custom_framework():
    """Show how to integrate RIVIR with any custom LLM framework."""
    print("\n⚙️  Custom Framework Integration")
    print("=" * 50)
    
    class CustomLLMFramework:
        """Example custom framework that expects LLM interface."""
        
        def __init__(self, model):
            self.model = model
        
        def process_request(self, user_input: str) -> dict:
            """Process user request through the framework."""
            # Pre-processing
            enhanced_prompt = f"User request: {user_input}\nPlease respond helpfully:"
            
            # Generate using any LLM
            if hasattr(self.model, 'generate'):
                # RIVIR interface
                output = self.model.generate(enhanced_prompt)
                response_text = output.text
                metadata = {
                    'alignment': output.alignment_score,
                    'listening': output.listening_data,
                    'receipt_id': output.receipt_id
                }
            else:
                # Standard LLM interface
                response_text = self.model(enhanced_prompt)
                metadata = {}
            
            # Post-processing
            return {
                'response': response_text,
                'metadata': metadata,
                'processed_at': 'custom_framework'
            }
    
    # Use RIVIR with custom framework
    rivir_model = RIVIRModel()
    framework = CustomLLMFramework(rivir_model)
    
    print("✓ RIVIR integrated with custom framework")
    
    result = framework.process_request("I need help building a meditation app")
    
    print(f"Framework Response: {result['response'][:150]}...")
    print(f"Alignment Score: {result['metadata'].get('alignment', 'N/A')}")
    
    rivir_model.close()


# =============================================================================
# STREAMING INTEGRATION
# =============================================================================

def integrate_streaming():
    """Show how to use RIVIR with streaming interfaces."""
    print("\n🌊 Streaming Integration")
    print("=" * 50)
    
    model = RIVIRModel()
    
    print("✓ Starting streaming generation...")
    
    prompt = "Create a beautiful, calming todo application"
    
    # Stream the response
    full_response = ""
    for chunk in model.stream_generate(prompt):
        print(chunk, end='', flush=True)
        full_response += chunk
    
    print(f"\n\n✓ Streamed {len(full_response)} characters")
    
    model.close()


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    """Run all integration examples."""
    print("🌊 RIVIR LLM Integration Examples")
    print("=" * 60)
    print("Showing how to plug RIVIR into various LLM systems...")
    print()
    
    try:
        integrate_with_mlx()
        integrate_with_transformers()
        integrate_with_openai()
        integrate_with_langchain()
        integrate_with_custom_framework()
        integrate_streaming()
        
        print("\n" + "=" * 60)
        print("✅ All integrations demonstrated successfully!")
        print("\nKey Benefits of RIVIR as LLM:")
        print("  • Quantum affordances for creative exploration")
        print("  • Deep listening for understanding user intent")
        print("  • White hat principles for ethical responses")
        print("  • Learning from feedback for continuous improvement")
        print("  • Compatible with any LLM framework")
        
    except Exception as e:
        print(f"\n❌ Integration error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()