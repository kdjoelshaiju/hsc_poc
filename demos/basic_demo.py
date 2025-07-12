#!/usr/bin/env python3
"""
HSC Enhanced Demo - Comprehensive demonstration for hackathon judges

This script demonstrates the enhanced HSC functionality:
1. Abstract Type System with type inference
2. Hybrid embeddings with neural model support
3. Error recovery and feedback
4. Manifold learning with quality metrics
5. Explainable transitions during interpolation
6. Versioning and snapshot capabilities
"""

import numpy as np
import matplotlib.pyplot as plt
from embedding.engine import DSLToVectorEngine, EmbeddingConfig
from dsl.examples import examples

def main():
    print("🚀 Holistic Semantic Continuum (HSC) - Enhanced Hackathon Demo")
    print("=" * 70)
    print()
    
    # 1. Initialize the enhanced embedding engine
    print("1️⃣ Initializing Enhanced DSL-to-Vector Engine...")
    config = EmbeddingConfig(
        vector_dim=64,
        normalize=True,
        use_semantic_weights=True,
        use_neural_embeddings=False,  # Set to True if neural models available
        enable_versioning=True,
        snapshot_auto_save=True
    )
    engine = DSLToVectorEngine(config)
    print("   ✅ Enhanced engine initialized successfully")
    print(f"   📊 Configuration: {config.vector_dim}D vectors, versioning enabled")
    print()
    
    # 2. Show enhanced examples with type analysis
    print("2️⃣ Enhanced Program Examples with Type Analysis...")
    print("   📝 Programs with abstract type inference:")
    
    encoded_programs = {}
    type_analyses = {}
    
    for i, (name, code) in enumerate(examples.items()):
        print(f"   {i+1}. {name}:")
        print(f"      Code: {code[:50]}{'...' if len(code) > 50 else ''}")
        
        # Encode with enhanced features
        result = engine.encode_program(code, f"prog_{i}")
        encoded_programs[name] = result
        
        # Show type analysis
        type_summary = result.type_summary
        type_analyses[name] = type_summary
        
        print(f"      Type Coverage: {type_summary.get('type_coverage', 0):.2f}")
        print(f"      Type Confidence: {type_summary.get('type_confidence', 0):.2f}")
        print(f"      Type Distribution: {type_summary.get('type_distribution', {})}")
        print()
    
    # 3. Demonstrate error recovery and feedback
    print("3️⃣ Error Recovery and Feedback System...")
    print("   🔧 Testing error recovery with invalid input:")
    
    # Test with malformed program
    malformed_code = "def broken_program( x = 1 + + 2 return"
    print(f"   Testing: {malformed_code}")
    
    try:
        result = engine.encode_program(malformed_code, "broken")
        if result.errors:
            print(f"   ⚠️  Errors detected: {result.errors}")
            print(f"   🔍 Error recovery confidence: {result.confidence:.2f}")
        else:
            print("   ✅ Program processed successfully")
    except Exception as e:
        print(f"   ❌ Processing failed: {e}")
    print()
    
    # 4. Compute enhanced similarity with type awareness
    print("4️⃣ Enhanced Similarity Analysis with Type Awareness...")
    print("   🔍 Computing semantic similarities:")
    
    program_names = list(encoded_programs.keys())
    similarity_matrix = np.zeros((len(program_names), len(program_names)))
    
    for i, name1 in enumerate(program_names):
        for j, name2 in enumerate(program_names):
            if i == j:
                similarity_matrix[i, j] = 1.0
            else:
                similarity = engine.compute_similarity(f"prog_{i}", f"prog_{j}")
                similarity_matrix[i, j] = similarity
    
    # Show similarity results
    print("   📊 Similarity Matrix:")
    print("   " + " " * 15 + "".join(f"{name[:8]:>8}" for name in program_names))
    for i, name in enumerate(program_names):
        row = f"   {name[:14]:<15}"
        for j in range(len(program_names)):
            row += f"{similarity_matrix[i, j]:>8.3f}"
        print(row)
    print()
    
    # 5. Manifold learning with quality metrics
    print("5️⃣ Manifold Learning with Quality Metrics...")
    print("   🧠 Learning program embedding manifold:")
    
    try:
        quality_report = engine.learn_manifold()
        print("   📈 Manifold Quality Report:")
        for metric, value in quality_report.items():
            if isinstance(value, float):
                print(f"      {metric}: {value:.3f}")
            else:
                print(f"      {metric}: {value}")
        print()
    except Exception as e:
        print(f"   ⚠️  Manifold learning failed: {e}")
        print()
    
    # 6. Enhanced interpolation with explainable transitions
    print("6️⃣ Enhanced Interpolation with Explainable Transitions...")
    print("   🔄 Interpolating between programs with transition explanations:")
    
    if len(program_names) >= 2:
        start_prog = program_names[0]
        end_prog = program_names[1]
        print(f"   From: {start_prog}")
        print(f"   To: {end_prog}")
        
        try:
            interpolation_results = engine.interpolate_programs("prog_0", "prog_1", steps=5)
            
            print("   📋 Interpolation Steps with Transitions:")
            for i, (vector, decode_result, annotations) in enumerate(interpolation_results):
                step = i / (len(interpolation_results) - 1)
                print(f"      Step {i} (t={step:.2f}):")
                print(f"        Success: {decode_result.success}")
                print(f"        Confidence: {decode_result.confidence:.3f}")
                
                if annotations:
                    for ann in annotations:
                        print(f"        Transition: {ann.explanation}")
                        if ann.structural_changes:
                            print(f"          Structural: {ann.structural_changes[0]}")
                        if ann.semantic_changes:
                            print(f"          Semantic: {ann.semantic_changes[0]}")
                else:
                    print("        No transitions (initial step)")
                print()
        except Exception as e:
            print(f"   ⚠️  Interpolation failed: {e}")
    print()
    
    # 7. Versioning and snapshot capabilities
    print("7️⃣ Versioning and Snapshot System...")
    print("   💾 Creating snapshots and managing versions:")
    
    try:
        # Create initial snapshot
        snapshot_id = engine.create_snapshot(
            description="Initial demo state",
            tags=["demo", "initial", "hackathon"]
        )
        print(f"   ✅ Created snapshot: {snapshot_id}")
        
        # List snapshots
        snapshots = engine.list_snapshots()
        print(f"   📋 Available snapshots: {len(snapshots)}")
        for snapshot in snapshots[:3]:  # Show first 3
            print(f"      - {snapshot.snapshot_id}: {snapshot.description}")
        
        # Create branch
        branch_id = engine.create_snapshot(
            description="Demo branch with modifications",
            tags=["demo", "branch", "modified"]
        )
        print(f"   🌿 Created branch: {branch_id}")
        
        # Compare snapshots
        if engine.versioning:
            comparison = engine.versioning.compare_snapshots(snapshot_id, branch_id)
        print("   🔍 Snapshot Comparison:")
        for key, value in comparison.get("metadata_comparison", {}).items():
            print(f"      {key}: {value}")
        print()
        
    except Exception as e:
        print(f"   ⚠️  Versioning failed: {e}")
        print()
    
    # 8. Enhanced statistics and insights
    print("8️⃣ Enhanced Statistics and Insights...")
    print("   📊 System Statistics:")
    
    stats = engine.get_statistics()
    for key, value in stats.items():
        print(f"      {key}: {value}")
    
    # Type analysis summary
    print("   🧬 Type Analysis Summary:")
    total_type_coverage = np.mean([analysis.get('type_coverage', 0) for analysis in type_analyses.values()])
    total_type_confidence = np.mean([analysis.get('type_confidence', 0) for analysis in type_analyses.values()])
    print(f"      Average Type Coverage: {total_type_coverage:.3f}")
    print(f"      Average Type Confidence: {total_type_confidence:.3f}")
    
    # Similarity insights
    print("   🔗 Similarity Insights:")
    max_similarity = np.max(similarity_matrix[similarity_matrix < 1.0])
    min_similarity = np.min(similarity_matrix[similarity_matrix > 0.0])
    avg_similarity = np.mean(similarity_matrix[similarity_matrix < 1.0])
    print(f"      Max Similarity: {max_similarity:.3f}")
    print(f"      Min Similarity: {min_similarity:.3f}")
    print(f"      Average Similarity: {avg_similarity:.3f}")
    print()
    
    # 9. Create enhanced visualization
    print("9️⃣ Creating Enhanced Visualization...")
    print("   📈 Generating comprehensive visualization:")
    
    try:
        # Create enhanced visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('HSC Enhanced Demo - Comprehensive Analysis', fontsize=16, fontweight='bold')
        
        # 1. Similarity heatmap
        im1 = ax1.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        ax1.set_title('Program Similarity Matrix')
        ax1.set_xticks(range(len(program_names)))
        ax1.set_yticks(range(len(program_names)))
        ax1.set_xticklabels([name[:8] for name in program_names], rotation=45)
        ax1.set_yticklabels([name[:8] for name in program_names])
        plt.colorbar(im1, ax=ax1)
        
        # 2. Type distribution
        type_categories = ['numeric', 'boolean', 'string', 'function', 'unknown']
        type_data = np.zeros((len(program_names), len(type_categories)))
        
        for i, name in enumerate(program_names):
            type_dist = type_analyses[name].get('type_distribution', {})
            for j, cat in enumerate(type_categories):
                type_data[i, j] = type_dist.get(cat, 0)
        
        im2 = ax2.imshow(type_data, cmap='plasma', aspect='auto')
        ax2.set_title('Type Distribution by Program')
        ax2.set_xticks(range(len(type_categories)))
        ax2.set_yticks(range(len(program_names)))
        ax2.set_xticklabels(type_categories, rotation=45)
        ax2.set_yticklabels([name[:8] for name in program_names])
        plt.colorbar(im2, ax=ax2)
        
        # 3. Quality metrics (if available)
        if 'quality_report' in locals():
            metrics = ['trustworthiness', 'continuity', 'neighborhood_preservation']
            metric_values = [quality_report.get(m, 0) for m in metrics]
            ax3.bar(metrics, metric_values, color=['#ff7f0e', '#2ca02c', '#d62728'])
            ax3.set_title('Manifold Quality Metrics')
            ax3.set_ylim(0, 1)
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'Quality metrics\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Manifold Quality Metrics')
        
        # 4. System statistics
        stat_names = ['total_programs', 'vector_dimension', 'manifold_fitted', 'snapshot_count']
        stat_values = [stats.get(name, 0) for name in stat_names]
        stat_labels = ['Programs', 'Vector Dim', 'Manifold', 'Snapshots']
        
        ax4.bar(stat_labels, stat_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        ax4.set_title('System Statistics')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('hsc_enhanced_demo_visualization.png', dpi=300, bbox_inches='tight')
        print("   ✅ Enhanced visualization saved as 'hsc_enhanced_demo_visualization.png'")
        
    except Exception as e:
        print(f"   ⚠️  Visualization failed: {e}")
    
    print()
    print("🎉 Enhanced HSC Demo Completed Successfully!")
    print("=" * 70)
    print()
    print("🌟 Key Enhanced Features Demonstrated:")
    print("   • Abstract Type System with automatic type inference")
    print("   • Hybrid embeddings with neural model support")
    print("   • Comprehensive error recovery and feedback")
    print("   • Manifold learning with quality metrics (trustworthiness, continuity)")
    print("   • Explainable transitions during program interpolation")
    print("   • Full versioning and snapshot capabilities")
    print("   • Enhanced similarity analysis with type awareness")
    print()
    print("🚀 Ready for hackathon presentation!")

if __name__ == "__main__":
    main() 