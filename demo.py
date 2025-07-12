#!/usr/bin/env python3
"""
HSC Demo - Simple demonstration for hackathon judges

This script demonstrates the core HSC functionality:
1. Program embedding
2. Semantic similarity
3. Basic interpolation
4. Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from embedding.engine import DSLToVectorEngine, EmbeddingConfig
from dsl.examples import examples

def main():
    print("🚀 Holistic Semantic Continuum (HSC) - Hackathon Demo")
    print("=" * 60)
    print()
    
    # 1. Initialize the embedding engine
    print("1️⃣ Initializing DSL-to-Vector Engine...")
    config = EmbeddingConfig(vector_dim=64, normalize=True, use_semantic_weights=True)
    engine = DSLToVectorEngine(config)
    print("   ✅ Engine initialized successfully")
    print()
    
    # 2. Show example programs
    print("2️⃣ Example Programs in our DSL:")
    for name, code in examples.items():
        print(f"   📝 {name}:")
        print(f"      {code.strip()}")
        print()
    
    # 3. Encode programs
    print("3️⃣ Encoding programs into semantic vectors...")
    program_codes = list(examples.values())
    program_names = list(examples.keys())
    
    embeddings = engine.encode_batch([(code, name) for code, name in zip(program_codes, program_names)])
    print(f"   ✅ Encoded {len(embeddings)} programs into {embeddings.shape[1]}-dimensional vectors")
    print(f"   📊 Vector shape: {embeddings.shape}")
    print()
    
    # 4. Show semantic similarities
    print("4️⃣ Computing semantic similarities...")
    similarity_matrix = engine.get_similarity_matrix(program_codes)
    
    print("   📈 Similarity Matrix:")
    print("      " + " ".join(f"{name:>8}" for name in program_names))
    for i, name in enumerate(program_names):
        similarities = similarity_matrix[i]
        print(f"   {name:>8} " + " ".join(f"{sim:>8.3f}" for sim in similarities))
    print()
    
    # 5. Find most similar programs
    print("5️⃣ Finding most similar programs...")
    query_program = examples["square"]
    similar_programs = engine.find_most_similar(query_program, program_codes, top_k=3)
    
    print(f"   🔍 Most similar to 'square':")
    for idx, similarity in similar_programs:
        print(f"      - {program_names[idx]}: {similarity:.3f}")
    print()
    
    # 6. Demonstrate interpolation
    print("6️⃣ Demonstrating semantic interpolation...")
    start_program = examples["square"]
    end_program = examples["factorial"]
    
    # Simple linear interpolation in vector space
    start_vec = engine.encode_program(start_program)
    end_vec = engine.encode_program(end_program)
    
    print(f"   🔄 Interpolating between 'square' and 'factorial'...")
    steps = 5
    for i in range(steps + 1):
        t = i / steps
        interpolated_vec = (1 - t) * start_vec + t * end_vec
        
        # Find closest program to interpolated vector
        similarities = []
        for j, code in enumerate(program_codes):
            vec = engine.encode_program(code)
            similarity = np.dot(interpolated_vec, vec) / (np.linalg.norm(interpolated_vec) * np.linalg.norm(vec))
            similarities.append((j, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        closest_program = program_names[similarities[0][0]]
        print(f"      Step {i}: {closest_program} (t={t:.1f})")
    print()
    
    # 7. Create visualization
    print("7️⃣ Creating visualization...")
    try:
        # Reduce to 2D for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100, alpha=0.7)
        
        # Add labels
        for i, name in enumerate(program_names):
            plt.annotate(name, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        # Draw interpolation path
        start_2d = pca.transform(start_vec.reshape(1, -1))[0]
        end_2d = pca.transform(end_vec.reshape(1, -1))[0]
        
        plt.plot([start_2d[0], end_2d[0]], [start_2d[1], end_2d[1]], 'r--', alpha=0.7, linewidth=2)
        plt.scatter([start_2d[0], end_2d[0]], [start_2d[1], end_2d[1]], 
                   c=['green', 'red'], s=150, marker='*', edgecolors='black')
        
        plt.title('HSC: Program Embeddings in Semantic Space', fontsize=14, fontweight='bold')
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('hsc_demo_visualization.png', dpi=300, bbox_inches='tight')
        print("   ✅ Visualization saved as 'hsc_demo_visualization.png'")
        
    except ImportError:
        print("   ⚠️  Visualization skipped (scikit-learn not available)")
    except Exception as e:
        print(f"   ⚠️  Visualization error: {e}")
    print()
    
    # 8. Summary
    print("8️⃣ Summary of HSC Capabilities:")
    print("   🎯 Core Innovation: Programs as points in continuous semantic space")
    print("   🔍 Semantic Understanding: Automatic similarity detection")
    print("   🔄 Continuous Interpolation: Smooth transitions between programs")
    print("   🚀 Future Potential: Algorithm synthesis and optimization")
    print()
    
    print("🎉 Demo completed successfully!")
    print("📊 Check 'hsc_demo_visualization.png' for the semantic space visualization")
    print()
    print("💡 Key Innovation: HSC treats code as a continuous manifold rather than discrete text,")
    print("   enabling mathematical operations on programs themselves!")

if __name__ == "__main__":
    main() 