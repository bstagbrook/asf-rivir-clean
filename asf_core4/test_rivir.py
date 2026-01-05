#!/usr/bin/env python3
"""
Quick test of RIVIR expert training and watching
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

def test_expert_training():
    """Test the expert training system."""
    print("🎯 Testing Expert Training")
    
    try:
        from rivir_learning import RIVIRLearningSystem
        from rivir_direct import RIVIRDirect
        
        # Create system
        rivir = RIVIRDirect(":memory:")
        system = RIVIRLearningSystem(rivir)
        
        # Test becoming expert
        expertise = system.trainer.become_expert("web development", "moderate")
        print(f"✓ Created expertise: {expertise.name}")
        print(f"  Focus areas: {len(expertise.focus_areas)}")
        print(f"  Sample: {expertise.focus_areas[0] if expertise.focus_areas else 'None'}")
        
        # Test practice
        result = system.trainer.practice_skill("web development")
        print(f"✓ Practice result: {len(result)} chars")
        print(f"  Level after practice: {expertise.current_level:.1%}")
        
        # Test status
        status = system.trainer.get_expertise_status()
        print(f"✓ Status: {len(status)} domains")
        
        rivir.close()
        return True
        
    except Exception as e:
        print(f"✗ Expert training failed: {e}")
        return False

def test_llm_interface():
    """Test the LLM interface."""
    print("\n🤖 Testing LLM Interface")
    
    try:
        from rivir_llm import RIVIRModel
        
        # Create model
        model = RIVIRModel(":memory:")
        print("✓ Model created")
        
        # Test generation
        output = model.generate("create a simple app")
        print(f"✓ Generated: {len(output.text)} chars")
        print(f"  Alignment: {output.alignment_score:.2f}")
        
        # Test feedback
        success = model.learn_from_feedback(output.response_id, 0.9)
        print(f"✓ Feedback: {success}")
        
        # Test status
        status = model.get_status()
        print(f"✓ Status: {status['generations_count']} generations")
        
        model.close()
        return True
        
    except Exception as e:
        print(f"✗ LLM interface failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 RIVIR Quick Tests")
    print("=" * 40)
    
    results = []
    results.append(test_expert_training())
    results.append(test_llm_interface())
    
    print("\n" + "=" * 40)
    if all(results):
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
        
    print(f"Results: {sum(results)}/{len(results)} passed")

if __name__ == "__main__":
    main()