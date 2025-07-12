#!/usr/bin/env python3
"""
Holistic Semantic Continuum (HSC) - Main Entry Point

This module demonstrates the complete HSC pipeline:
1. DSL parsing and embedding
2. Manifold learning
3. Geodesic interpolation
4. Program synthesis and evaluation
5. Visualization

Usage:
    python main.py [--demo] [--interactive] [--config CONFIG_FILE]
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from embedding.engine import DSLToVectorEngine, EmbeddingConfig
from manifold.autoencoder import AutoencoderManifold
from interpolation.geodesic import GeodesicInterpolator
from evaluation.similarity import SimilarityEvaluator
from visualization.plotter import HSCVisualizer
from dsl.examples import examples


def setup_logging(level=logging.INFO):
    """Setup logging configuration."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('hsc.log')
        ]
    )


def run_demo():
    """Run a demonstration of the HSC system."""
    print("🚀 Holistic Semantic Continuum (HSC) Demo")
    print("=" * 50)
    
    # Initialize components
    print("\n1. Initializing DSL-to-Vector Engine...")
    config = EmbeddingConfig(vector_dim=64, normalize=True, use_semantic_weights=True)
    engine = DSLToVectorEngine(config)
    
    # Encode example programs
    print("\n2. Encoding example programs...")
    program_codes = list(examples.values())
    program_names = list(examples.keys())
    
    # Encode programs individually
    embeddings = []
    for code, name in zip(program_codes, program_names):
        result = engine.encode_program(code, name)
        embeddings.append(result.vector)
    
    embeddings = np.array(embeddings)
    print(f"   Encoded {len(embeddings)} programs into {embeddings.shape[1]}-dimensional vectors")
    
    # Compute similarities
    print("\n3. Computing semantic similarities...")
    
    # Find similar programs
    query_program = examples["square"]
    engine.encode_program(query_program, "query")
    
    # Get program IDs for similarity computation
    query_id = "query"
    
    similarities = []
    for i, name in enumerate(program_names):
        if name in engine.embeddings:
            similarity = engine.compute_similarity(query_id, name)
            similarities.append((i, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    similar_programs = similarities[:3]
    
    print("\n   Most similar to 'square':")
    for idx, similarity in similar_programs:
        print(f"   - {program_names[idx]}: {similarity:.3f}")
    
    # Initialize manifold learning
    print("\n4. Learning manifold structure...")
    manifold = AutoencoderManifold(input_dim=embeddings.shape[1], latent_dim=16)
    manifold.fit(embeddings)
    
    # Initialize interpolator
    print("\n5. Setting up geodesic interpolator...")
    interpolator = GeodesicInterpolator(manifold)
    
    # Demonstrate interpolation
    print("\n6. Demonstrating program interpolation...")
    start_program = examples["square"]
    end_program = examples["factorial"]
    
    interpolated_programs = interpolator.interpolate(
        start_program, end_program, steps=5, engine=engine
    )
    
    print("   Interpolated between 'square' and 'factorial' in 5 steps")
    
    # Evaluate results
    print("\n7. Evaluating interpolation quality...")
    evaluator = SimilarityEvaluator()
    quality_metrics = evaluator.evaluate_interpolation(
        start_program, end_program, interpolated_programs, engine
    )
    print(f"   Interpolation quality score: {quality_metrics['overall_quality']:.3f}")
    print(f"   Smoothness: {quality_metrics['smoothness']:.3f}")
    print(f"   Semantic coherence: {quality_metrics['semantic_coherence']:.3f}")
    
    # Visualization
    print("\n8. Creating visualizations...")
    visualizer = HSCVisualizer()
    visualizer.plot_embeddings(embeddings, program_names, save_path="embeddings_plot.png")
    visualizer.plot_interpolation_path(interpolated_programs, save_path="interpolation_path.png")
    
    print("\n✅ Demo completed successfully!")
    print("📊 Check generated plots: embeddings_plot.png, interpolation_path.png")


def run_interactive():
    """Run interactive mode for exploring the HSC system."""
    print("🔍 HSC Interactive Mode")
    print("=" * 30)
    print("Available commands:")
    print("  encode <program> - Encode a program")
    print("  similar <program> <k> - Find similar programs")
    print("  interpolate <start> <end> <steps> - Interpolate between programs")
    print("  visualize - Show current embeddings")
    print("  quit - Exit")
    
    engine = DSLToVectorEngine()
    
    while True:
        try:
            command = input("\nHSC> ").strip().split()
            if not command:
                continue
                
            cmd = command[0].lower()
            
            if cmd == "quit":
                break
            elif cmd == "encode":
                if len(command) < 2:
                    print("Usage: encode <program_name>")
                    continue
                program_name = command[1]
                if program_name in examples:
                    result = engine.encode_program(examples[program_name])
                    print(f"Encoded '{program_name}' into {len(result.vector)}-dimensional vector")
                else:
                    print(f"Unknown program: {program_name}")
            elif cmd == "similar":
                if len(command) < 3:
                    print("Usage: similar <program_name> <k>")
                    continue
                program_name = command[1]
                k = int(command[2])
                if program_name in examples:
                    # Encode the query program
                    engine.encode_program(examples[program_name], "query")
                    
                    # Find similar programs
                    similarities = []
                    for name in examples.keys():
                        if name in engine.embeddings and name != "query":
                            similarity = engine.compute_similarity("query", name)
                            similarities.append((name, similarity))
                    
                    # Sort and take top k
                    similarities.sort(key=lambda x: x[1], reverse=True)
                    similar = similarities[:k]
                    
                    print(f"Most similar to '{program_name}':")
                    for name, similarity in similar:
                        print(f"  {name}: {similarity:.3f}")
                else:
                    print(f"Unknown program: {program_name}")
            elif cmd == "interpolate":
                if len(command) < 4:
                    print("Usage: interpolate <start_program> <end_program> <steps>")
                    continue
                start_name = command[1]
                end_name = command[2]
                steps = int(command[3])
                if start_name in examples and end_name in examples:
                    print(f"Interpolating between '{start_name}' and '{end_name}' in {steps} steps...")
                    # This would require full interpolation setup
                    print("(Interpolation not yet implemented in interactive mode)")
                else:
                    print("Unknown program name(s)")
            elif cmd == "visualize":
                print("Creating visualization...")
                # This would create and show plots
                print("(Visualization not yet implemented in interactive mode)")
            else:
                print(f"Unknown command: {cmd}")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Holistic Semantic Continuum (HSC)")
    parser.add_argument("--demo", action="store_true", help="Run demonstration")
    parser.add_argument("--interactive", action="store_true", help="Run interactive mode")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting HSC system...")
    
    try:
        if args.demo:
            run_demo()
        elif args.interactive:
            run_interactive()
        else:
            # Default to demo
            run_demo()
            
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 