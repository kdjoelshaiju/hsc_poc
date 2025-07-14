"""
Semantic Search & Transformation Demo

This demo showcases the HSC semantic code search and transformation engine,
demonstrating how to search, compare, and transform code by meaning.
"""

import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.benchmarks import SemanticSearchEngine, BenchmarkSuite
from visualization.semantic_search import SemanticSearchVisualizer

def print_section_header(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"🚀 {title}")
    print(f"{'='*60}")

def print_subsection_header(title: str):
    """Print a formatted subsection header."""
    print(f"\n{'─'*50}")
    print(f"📋 {title}")
    print(f"{'─'*50}")

def demo_search_functionality(engine: SemanticSearchEngine, visualizer: SemanticSearchVisualizer):
    """Demonstrate semantic search functionality."""
    print_subsection_header("Semantic Code Search")
    
    # Test queries
    test_queries = [
        "def search(graph, start):",
        "def sort(array):",
        "def find_element(list, target):",
        "class DataStructure:",
        "def recursive_function(n):",
        "def iterative_loop(data):"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Search {i}: '{query}'")
        print("-" * 40)
        
        start_time = time.time()
        results = engine.search(query, top_k=3)
        search_time = time.time() - start_time
        
        print(f"⏱️ Search time: {search_time:.3f}s")
        
        for result in results:
            print(f"   {result.rank}. {result.snippet.name} ({result.similarity_score:.3f})")
            print(f"      Category: {result.snippet.category}")
            print(f"      Description: {result.snippet.description}")
        
        # Visualize search results
        visualizer.visualize_search_results(
            results, query, 
            save_path=f"demo_outputs/search_results_{i}.png"
        )

def demo_transformation_functionality(engine: SemanticSearchEngine, visualizer: SemanticSearchVisualizer):
    """Demonstrate code transformation functionality."""
    print_subsection_header("Code Transformation")
    
    # Test transformations
    transformations = [
        ("dfs_recursive", "dfs_iterative", "Recursive to Iterative DFS"),
        ("bubble_sort", "quick_sort", "Bubble Sort to Quick Sort"),
        ("linked_list", "stack", "Linked List to Stack"),
        ("binary_search", "linear_search", "Binary Search to Linear Search")
    ]
    
    for i, (start, end, description) in enumerate(transformations, 1):
        print(f"\n🔄 Transformation {i}: {description}")
        print(f"   From: {start} → To: {end}")
        print("-" * 40)
        
        start_time = time.time()
        path = engine.transform(start, end, steps=5)
        transform_time = time.time() - start_time
        
        if path:
            print(f"✅ Transformation successful!")
            print(f"⏱️ Transform time: {transform_time:.3f}s")
            print(f"📊 Average confidence: {sum(path.confidence_scores) / len(path.confidence_scores):.3f}")
            print(f"📊 Steps: {len(path.intermediate_steps)}")
            
            # Show intermediate steps
            for j, step in enumerate(path.intermediate_steps):
                print(f"\n   Step {j}:")
                print(f"      Confidence: {path.confidence_scores[j]:.3f}")
                code_preview = step.code.strip().split('\n')[0]
                print(f"      Code: {code_preview}...")
            
            # Visualize transformation path
            visualizer.plot_transformation_path(
                path, save_path=f"demo_outputs/transformation_{i}.png"
            )
        else:
            print(f"❌ Transformation failed")

def demo_visualization_functionality(engine: SemanticSearchEngine, visualizer: SemanticSearchVisualizer):
    """Demonstrate visualization functionality."""
    print_subsection_header("Visualization")
    
    print("📊 Creating comprehensive visualization report...")
    
    # Create all visualizations
    visualizer.create_comprehensive_report(
        engine, save_dir="demo_outputs/visualization_report"
    )
    
    print("✅ Visualization report created!")
    print("📁 Check the 'demo_outputs/visualization_report' directory for:")
    print("   • embedding_space.png - 2D embedding space")
    print("   • similarity_matrix.png - Similarity heatmap")
    print("   • category_distribution.png - Algorithm distribution")
    print("   • statistics.json - Detailed statistics")

def demo_statistics_and_analysis(engine: SemanticSearchEngine):
    """Demonstrate statistics and analysis functionality."""
    print_subsection_header("Statistics & Analysis")
    
    stats = engine.get_statistics()
    
    print(f"📊 Search Corpus Statistics:")
    print(f"   Total algorithms: {stats['total_algorithms']}")
    print(f"   Vector dimension: {stats['vector_dimension']}")
    print(f"   Total code length: {stats['total_code_length']:,} characters")
    print(f"   Average code length: {stats['average_code_length']:.1f} characters")
    
    print(f"\n📂 Categories:")
    for category, count in stats['categories'].items():
        print(f"   • {category}: {count} algorithms")
    
    # Analyze similarity patterns
    print(f"\n🔍 Similarity Analysis:")
    
    # Find most similar pairs
    most_similar_pairs = []
    for i, snippet1 in enumerate(engine.search_index):
        for j, snippet2 in enumerate(engine.search_index[i+1:], i+1):
            if snippet1.vector is not None and snippet2.vector is not None:
                similarity = engine.similarity_evaluator._compute_cosine_similarity(
                    snippet1.vector, snippet2.vector
                )
                most_similar_pairs.append((similarity, snippet1.name, snippet2.name))
    
    most_similar_pairs.sort(reverse=True)
    
    print(f"   Top 5 most similar algorithm pairs:")
    for i, (similarity, name1, name2) in enumerate(most_similar_pairs[:5], 1):
        print(f"   {i}. {name1} ↔ {name2}: {similarity:.3f}")

def demo_benchmark_suite():
    """Demonstrate the benchmark suite."""
    print_subsection_header("Benchmark Suite")
    
    print("🧪 Running comprehensive benchmarks...")
    
    suite = BenchmarkSuite()
    suite.run_benchmarks()
    
    print("✅ Benchmarks completed!")

def main():
    """Main demo function."""
    print_section_header("HSC Semantic Search & Transformation Demo")
    
    # Create output directory
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    print("🎯 This demo showcases:")
    print("   • Semantic code search by meaning")
    print("   • Code transformation via interpolation")
    print("   • 2D/3D visualization of embedding space")
    print("   • Similarity analysis and statistics")
    print("   • Comprehensive benchmarking")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    engine = SemanticSearchEngine()
    visualizer = SemanticSearchVisualizer(output_dir="demo_outputs")
    
    # Build or load search corpus
    print("\n📚 Setting up search corpus...")
    if not Path("data/search_index.json").exists():
        print("   Building new search corpus...")
        engine.build_search_corpus()
    else:
        print("   Loading existing search corpus...")
        engine.load_index()
    
    # Run demos
    demo_search_functionality(engine, visualizer)
    demo_transformation_functionality(engine, visualizer)
    demo_visualization_functionality(engine, visualizer)
    demo_statistics_and_analysis(engine)
    demo_benchmark_suite()
    
    # Summary
    print_section_header("Demo Summary")
    
    print("🎉 Demo completed successfully!")
    print("\n📁 Generated outputs:")
    print("   • demo_outputs/ - All visualization files")
    print("   • data/search_index.json - Search corpus")
    print("   • Various PNG plots and JSON reports")
    
    print("\n🚀 Next steps:")
    print("   • Try the CLI: python app.py search 'your query'")
    print("   • Transform algorithms: python app.py transform start end")
    print("   • List available algorithms: python app.py list")
    print("   • Run benchmarks: python app.py benchmark")
    
    print("\n💡 Key insights:")
    print("   • HSC can find semantically similar code regardless of syntax")
    print("   • Code transformations preserve semantic meaning")
    print("   • The embedding space reveals algorithm relationships")
    print("   • Similarity scores help understand code relationships")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc() 