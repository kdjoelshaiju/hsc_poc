#!/usr/bin/env python3
"""
HSC Semantic Code Search & Transformation Engine

A command-line interface for semantic code search, comparison, and transformation.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import List, Optional

from evaluation.benchmarks import SemanticSearchEngine, BenchmarkSuite

def print_banner():
    """Print the application banner."""
    print("🚀 HSC Semantic Code Search & Transformation Engine")
    print("=" * 60)
    print("Search, compare, and transform code by meaning, not syntax")
    print("=" * 60)

def print_search_results(results: List, query: str):
    """Print search results in a formatted way."""
    print(f"\n🔍 Search Results for: '{query}'")
    print("-" * 50)
    
    if not results:
        print("❌ No results found")
        return
    
    for result in results:
        print(f"\n{result.rank}. {result.snippet.name} (Score: {result.similarity_score:.3f})")
        print(f"   Category: {result.snippet.category}")
        print(f"   Description: {result.snippet.description}")
        print("   Code Preview:")
        code_lines = result.snippet.code.strip().split('\n')[:5]
        for line in code_lines:
            print(f"     {line}")
        if len(result.snippet.code.split('\n')) > 5:
            remaining_lines = len(result.snippet.code.split('\n')) - 5
            print(f"     ... ({remaining_lines} more lines)")

def print_transformation_path(path, start_name: str, end_name: str):
    """Print transformation path results."""
    print(f"\n🔄 Transformation: {start_name} → {end_name}")
    print("-" * 50)
    
    if not path:
        print("❌ Transformation failed")
        return
    
    print(f"✅ Transformation successful!")
    print(f"📊 Average confidence: {sum(path.confidence_scores) / len(path.confidence_scores):.3f}")
    print(f"📊 Steps: {len(path.intermediate_steps)}")
    
    print(f"\n📝 Transformation Path:")
    for i, step in enumerate(path.intermediate_steps):
        print(f"\nStep {i}:")
        print(f"   Confidence: {path.confidence_scores[i]:.3f}")
        print("   Code:")
        code_lines = step.code.strip().split('\n')[:3]
        for line in code_lines:
            print(f"     {line}")
        if len(step.code.split('\n')) > 3:
            remaining_lines = len(step.code.split('\n')) - 3
            print(f"     ... ({remaining_lines} more lines)")

def search_command(args):
    """Handle search command."""
    print_banner()
    
    engine = SemanticSearchEngine()
    
    # Load existing index or build new one
    if args.rebuild:
        print("📚 Building search corpus...")
        engine.build_search_corpus()
    else:
        print("📚 Loading search index...")
        engine.load_index()
    
    # Perform search
    print(f"\n🔍 Searching for: '{args.query}'")
    results = engine.search(args.query, top_k=args.top_k)
    
    print_search_results(results, args.query)
    
    # Save results if requested
    if args.save:
        output_file = Path(args.save)
        output_data = {
            "query": args.query,
            "results": [
                {
                    "rank": r.rank,
                    "name": r.snippet.name,
                    "category": r.snippet.category,
                    "description": r.snippet.description,
                    "similarity_score": r.similarity_score,
                    "code": r.snippet.code
                }
                for r in results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n💾 Results saved to {output_file}")

def transform_command(args):
    """Handle transform command."""
    print_banner()
    
    engine = SemanticSearchEngine()
    engine.load_index()
    
    print(f"\n🔄 Transforming: {args.start} → {args.end}")
    path = engine.transform(args.start, args.end, steps=args.steps)
    
    print_transformation_path(path, args.start, args.end)
    
    # Save transformation if requested
    if args.save:
        output_file = Path(args.save)
        if path:
            output_data = {
                "start": args.start,
                "end": args.end,
                "steps": args.steps,
                "average_confidence": sum(path.confidence_scores) / len(path.confidence_scores),
                "intermediate_steps": [
                    {
                        "step": i,
                        "confidence": path.confidence_scores[i],
                        "code": step.code
                    }
                    for i, step in enumerate(path.intermediate_steps)
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"\n💾 Transformation saved to {output_file}")

def list_command(args):
    """Handle list command."""
    print_banner()
    
    engine = SemanticSearchEngine()
    engine.load_index()
    
    stats = engine.get_statistics()
    
    print(f"\n📊 Search Corpus Statistics:")
    print(f"   Total algorithms: {stats['total_algorithms']}")
    print(f"   Vector dimension: {stats['vector_dimension']}")
    print(f"   Average code length: {stats['average_code_length']:.1f} chars")
    
    print(f"\n📂 Categories:")
    for category, count in stats['categories'].items():
        print(f"   {category}: {count} algorithms")
    
    print(f"\n📝 Available Algorithms:")
    print("-" * 50)
    
    # Group by category
    by_category = {}
    for snippet in engine.search_index:
        cat = snippet.category
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(snippet)
    
    for category, snippets in by_category.items():
        print(f"\n{category.upper()}:")
        for snippet in snippets:
            print(f"   • {snippet.name}: {snippet.description}")

def benchmark_command(args):
    """Handle benchmark command."""
    print_banner()
    
    suite = BenchmarkSuite()
    suite.run_benchmarks()

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="HSC Semantic Code Search & Transformation Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s search "def binary_search" --top-k 5
  %(prog)s transform dfs_recursive dfs_iterative --steps 5
  %(prog)s list
  %(prog)s benchmark
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search for semantically similar code')
    search_parser.add_argument('query', help='Code query to search for')
    search_parser.add_argument('--top-k', type=int, default=5, help='Number of top results (default: 5)')
    search_parser.add_argument('--rebuild', action='store_true', help='Rebuild search corpus')
    search_parser.add_argument('--save', help='Save results to JSON file')
    search_parser.set_defaults(func=search_command)
    
    # Transform command
    transform_parser = subparsers.add_parser('transform', help='Transform one program into another')
    transform_parser.add_argument('start', help='Starting algorithm name')
    transform_parser.add_argument('end', help='Ending algorithm name')
    transform_parser.add_argument('--steps', type=int, default=5, help='Number of interpolation steps (default: 5)')
    transform_parser.add_argument('--save', help='Save transformation to JSON file')
    transform_parser.set_defaults(func=transform_command)
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available algorithms')
    list_parser.set_defaults(func=list_command)
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run comprehensive benchmarks')
    benchmark_parser.set_defaults(func=benchmark_command)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 