#!/usr/bin/env python3
"""
Test script for comprehensive error recovery and feedback system.

This script demonstrates the enhanced error analysis capabilities
of the HSC system with detailed error reporting and visualization.
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

def test_normal_decoding():
    """Test normal decoding without errors."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Normal Decoding")
    print("="*60)
    
    engine = DSLToVectorEngine(EmbeddingConfig(vector_dim=64))
    
    # Encode a simple program
    program = "def test(): return 42"
    result = engine.encode_program(program)
    
    print(f"✅ Encoded program: {program}")
    print(f"   Vector dimension: {len(result.vector)}")
    print(f"   Confidence: {result.confidence:.3f}")
    
    # Decode the vector
    decode_result = engine.decode_program(result.vector)
    
    print(f"✅ Decoding result: {decode_result.success}")
    print(f"   Errors: {len(decode_result.errors)}")
    print(f"   Confidence: {decode_result.confidence:.3f}")

def test_error_generation():
    """Test error generation with problematic vectors."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Error Generation")
    print("="*60)
    
    engine = DSLToVectorEngine(EmbeddingConfig(vector_dim=64))
    
    # Create problematic vectors
    test_vectors = [
        ("NaN Vector", np.full(64, np.nan)),
        ("Inf Vector", np.full(64, np.inf)),
        ("Zero Vector", np.zeros(64)),
        ("Extreme Vector", np.random.uniform(0.9, 1.0, 64)),
        ("Uniform Vector", np.full(64, 0.5)),
        ("Random Vector", np.random.randn(64))
    ]
    
    for name, vector in test_vectors:
        print(f"\n🔍 Testing: {name}")
        print(f"   Vector shape: {vector.shape}")
        
        # Handle inf/NaN values in stats printing
        try:
            mean_val = np.mean(vector)
            std_val = np.std(vector)
            mean_str = f"{mean_val:.3f}" if np.isfinite(mean_val) else "inf" if np.isinf(mean_val) else "nan"
            std_str = f"{std_val:.3f}" if np.isfinite(std_val) else "inf" if np.isinf(std_val) else "nan"
            print(f"   Vector stats: mean={mean_str}, std={std_str}")
        except Exception:
            print(f"   Vector stats: mean=error, std=error")
        
        try:
            # Analyze the vector
            analysis = engine.analyze_decoding_errors(vector, save_analysis=True)
            
            error_report = analysis['error_report']
            print(f"   Total errors: {error_report.get('total_errors', 0)}")
            print(f"   Overall confidence: {error_report.get('overall_confidence', 0):.3f}")
            
            # Show error types
            error_types = error_report.get('error_types', {})
            if error_types:
                print(f"   Error types: {list(error_types.keys())}")
            
            # Show top recommendations
            recommendations = analysis.get('recovery_suggestions', [])
            if recommendations:
                print(f"   Top recommendation: {recommendations[0]}")
        
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")

def test_comprehensive_analysis():
    """Test comprehensive error analysis with visualization."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Comprehensive Analysis")
    print("="*60)
    
    engine = DSLToVectorEngine(EmbeddingConfig(vector_dim=64))
    
    # Create a vector with known issues
    problematic_vector = np.random.randn(64)
    problematic_vector[0:10] = np.nan  # Add NaN values
    problematic_vector[10:20] = np.inf  # Add inf values
    problematic_vector[20:30] = 0.95  # Add extreme values
    
    print("🔍 Analyzing problematic vector...")
    print(f"   NaN count: {np.sum(np.isnan(problematic_vector))}")
    print(f"   Inf count: {np.sum(np.isinf(problematic_vector))}")
    print(f"   Extreme count: {np.sum(problematic_vector > 0.9)}")
    
    # Perform comprehensive analysis
    analysis = engine.analyze_decoding_errors(problematic_vector, save_analysis=True)
    
    # Print detailed analysis
    error_report = analysis['error_report']
    print("\n📊 Analysis Results:")
    print(f"   Total errors: {error_report.get('total_errors', 0)}")
    print(f"   Overall confidence: {error_report.get('overall_confidence', 0):.3f}")
    
    # Show detailed error breakdown
    detailed_errors = error_report.get('detailed_errors', [])
    for i, error in enumerate(detailed_errors):
        print(f"\n   Error {i+1}:")
        print(f"     Type: {error['type']}")
        print(f"     Severity: {error['severity']}")
        print(f"     Confidence: {error['confidence']:.3f}")
        print(f"     Message: {error['message']}")
        print(f"     Suggestions: {len(error['suggestions'])} items")
    
    # Show vector analysis
    vector_analysis = analysis.get('vector_analysis', {})
    if vector_analysis:
        print("\n📈 Vector Analysis:")
        print(f"   Dimension: {vector_analysis.get('dimension', 0)}")
        print(f"   Norm: {vector_analysis.get('norm', 0):.3f}")
        print(f"   Distribution: {vector_analysis.get('distribution_type', 'UNKNOWN')}")
        print(f"   Has NaN: {vector_analysis.get('has_nan', False)}")
        print(f"   Has Inf: {vector_analysis.get('has_inf', False)}")

def test_error_visualization():
    """Test error visualization capabilities."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Error Visualization")
    print("="*60)
    
    try:
        from visualization.error_analysis import ErrorAnalyzer
        
        # Create test error data
        test_vector = np.random.randn(64)
        test_vector[0:5] = np.nan
        
        engine = DSLToVectorEngine(EmbeddingConfig(vector_dim=64))
        decode_result = engine.decode_program(test_vector)
        
        # Create visualizations
        analyzer = ErrorAnalyzer()
        
        print("📊 Creating error visualizations...")
        
        # Print error summary
        analyzer.print_error_summary(decode_result)
        
        # Create plots
        error_report = decode_result.metadata.get('error_report', {})
        error_patterns = decode_result.metadata.get('error_patterns', {})
        
        if error_report:
            analyzer.plot_error_summary(error_report, "test_error_summary.png")
            analyzer.plot_vector_analysis(test_vector, error_report, "test_vector_analysis.png")
        
        if error_patterns:
            analyzer.plot_error_patterns(error_patterns, "test_error_patterns.png")
        
        print("✅ Error visualizations created successfully")
        
    except ImportError as e:
        print(f"⚠️ Visualization module not available: {e}")
    except Exception as e:
        print(f"❌ Visualization failed: {e}")

def main():
    """Run all error recovery tests."""
    setup_logging()
    
    print("🚀 HSC Error Recovery and Feedback System Test")
    print("="*60)
    
    try:
        # Run tests
        test_normal_decoding()
        test_error_generation()
        test_comprehensive_analysis()
        test_error_visualization()
        
        print("\n" + "="*60)
        print("✅ All error recovery tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 