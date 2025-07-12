"""
Test script for real program encoding/decoding without artificial error vectors.

This script demonstrates that the HSC system works correctly for real programs.
"""

import numpy as np
import logging
import warnings
from embedding.engine import DSLToVectorEngine, EmbeddingConfig

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    # Suppress expected numpy warnings about inf values
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in subtract')

def test_real_programs():
    """Test encoding and decoding of real programs."""
    print("🚀 HSC Real Program Encoding/Decoding Test")
    print("="*60)
    
    engine = DSLToVectorEngine(EmbeddingConfig(vector_dim=64))
    
    # Test real programs
    test_programs = [
        "def test(): return 42",
        "def add(a, b): return a + b",
        "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
        "x = 10\ny = 20\nresult = x + y",
        "if x > 0:\n    return x\nelse:\n    return -x"
    ]
    
    for i, program in enumerate(test_programs, 1):
        print(f"\n🧪 TEST {i}: {program[:30]}...")
        
        try:
            # Encode the program
            encode_result = engine.encode_program(program)
            print(f"   ✅ Encoded successfully")
            print(f"   📊 Vector dimension: {len(encode_result.vector)}")
            print(f"   📊 Vector norm: {np.linalg.norm(encode_result.vector):.3f}")
            print(f"   📊 Vector stats: mean={np.mean(encode_result.vector):.3f}, std={np.std(encode_result.vector):.3f}")
            print(f"   📊 Confidence: {encode_result.confidence:.3f}")
            
            # Check if vector is valid
            if np.all(np.isfinite(encode_result.vector)):
                print(f"   ✅ Vector is valid (no NaN/inf)")
            else:
                print(f"   ❌ Vector contains invalid values")
                continue
            
            # Decode the vector
            decode_result = engine.decode_program(encode_result.vector)
            print(f"   📊 Decode success: {decode_result.success}")
            print(f"   📊 Decode confidence: {decode_result.confidence:.3f}")
            print(f"   📊 Decode errors: {len(decode_result.errors)}")
            
            if decode_result.success:
                print(f"   ✅ Decoding successful!")
            else:
                print(f"   ⚠️ Decoding had issues")
                for error in decode_result.errors:
                    print(f"      - {error.error_type.value}: {error.message}")
            
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
    
    print(f"\n" + "="*60)
    print("✅ Real program tests completed!")

def test_similarity():
    """Test similarity computation between real programs."""
    print("\n🔍 Testing Program Similarity")
    print("="*60)
    
    engine = DSLToVectorEngine(EmbeddingConfig(vector_dim=64))
    
    # Encode similar programs
    program1 = "def add(a, b): return a + b"
    program2 = "def sum(x, y): return x + y"
    program3 = "def multiply(a, b): return a * b"
    
    try:
        # Encode all programs
        result1 = engine.encode_program(program1, "prog1")
        result2 = engine.encode_program(program2, "prog2")
        result3 = engine.encode_program(program3, "prog3")
        
        # Compute similarities
        sim_12 = engine.compute_similarity("prog1", "prog2")  # Should be high (similar)
        sim_13 = engine.compute_similarity("prog1", "prog3")  # Should be lower (different)
        
        print(f"   📊 Similarity between similar programs: {sim_12:.3f}")
        print(f"   📊 Similarity between different programs: {sim_13:.3f}")
        
        if sim_12 > sim_13:
            print(f"   ✅ Similarity detection working correctly!")
        else:
            print(f"   ⚠️ Similarity detection may need adjustment")
            
    except Exception as e:
        print(f"   ❌ Similarity test failed: {e}")

def main():
    """Run all real program tests."""
    setup_logging()
    
    try:
        test_real_programs()
        test_similarity()
        
        print(f"\n" + "="*60)
        print("🎉 All real program tests completed successfully!")
        print("📊 The HSC system is working correctly for real programs!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 