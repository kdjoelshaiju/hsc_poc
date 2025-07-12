"""
Demo: Hierarchical Embedding for Large Codebases

This script demonstrates how to use the HSC hierarchical embedding system
to process large codebases like TheAlgorithms/Python efficiently.
"""

import os
import logging
import time
from pathlib import Path
from embedding.hierarchical import HierarchicalEmbedder, ChunkedEmbeddingConfig
import numpy as np

def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def analyze_codebase_structure(directory: str) -> dict:
    """Analyze the structure of a codebase."""
    directory_path = Path(directory)
    if not directory_path.exists():
        return {"error": f"Directory {directory} not found"}
    
    stats = {
        "total_files": 0,
        "python_files": 0,
        "total_lines": 0,
        "total_functions": 0,
        "large_functions": 0,
        "file_types": {},
        "largest_files": []
    }
    
    file_sizes = []
    
    for file_path in directory_path.rglob("*"):
        if file_path.is_file():
            stats["total_files"] += 1
            
            # Count lines
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()
                    line_count = len(lines)
                    stats["total_lines"] += line_count
                    
                    if file_path.suffix == '.py':
                        stats["python_files"] += 1
                        file_sizes.append((file_path, line_count))
                        
                        # Rough function count (simple heuristic)
                        function_count = sum(1 for line in lines if line.strip().startswith('def '))
                        stats["total_functions"] += function_count
                        
                        if line_count > 500:
                            stats["large_functions"] += 1
                    
                    # File type stats
                    ext = file_path.suffix
                    stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1
                    
            except Exception as e:
                logging.warning(f"Could not read {file_path}: {e}")
    
    # Get largest files
    file_sizes.sort(key=lambda x: x[1], reverse=True)
    stats["largest_files"] = [(str(f), s) for f, s in file_sizes[:10]]
    
    return stats

def estimate_memory_requirements(stats: dict, config: ChunkedEmbeddingConfig) -> dict:
    """Estimate memory requirements for embedding the codebase."""
    python_files = stats["python_files"]
    total_functions = stats["total_functions"]
    vector_dim = config.vector_dim
    
    # Memory estimates (rough)
    function_embedding_size = vector_dim * 8  # float64 bytes
    file_embedding_size = vector_dim * 8
    
    # Peak memory during processing
    peak_memory_mb = (
        # Function embeddings in memory (batch processing)
        (config.batch_size * function_embedding_size) / (1024 * 1024) +
        # File embeddings
        (python_files * file_embedding_size) / (1024 * 1024) +
        # AST nodes and other overhead
        50  # MB overhead
    )
    
    # Total storage size
    total_storage_mb = (
        (total_functions * function_embedding_size + 
         python_files * file_embedding_size) / (1024 * 1024)
    )
    
    return {
        "peak_memory_mb": peak_memory_mb,
        "total_storage_mb": total_storage_mb,
        "function_embeddings_count": total_functions,
        "file_embeddings_count": python_files,
        "vector_dimension": vector_dim
    }

def demo_hierarchical_embedding(directory: str, output_path: str = "codebase_embeddings.h5"):
    """Demonstrate hierarchical embedding of a large codebase."""
    print("🚀 HSC Hierarchical Embedding Demo")
    print("="*60)
    
    # Analyze codebase
    print("📊 Analyzing codebase structure...")
    stats = analyze_codebase_structure(directory)
    
    if "error" in stats:
        print(f"❌ {stats['error']}")
        return
    
    print(f"✅ Codebase analysis complete!")
    print(f"   📁 Total files: {stats['total_files']}")
    print(f"   🐍 Python files: {stats['python_files']}")
    print(f"   📏 Total lines: {stats['total_lines']:,}")
    print(f"   🔧 Total functions: {stats['total_functions']:,}")
    print(f"   📈 Large functions (>500 lines): {stats['large_functions']}")
    
    # Show largest files
    print(f"\n📋 Largest Python files:")
    for i, (file_path, line_count) in enumerate(stats["largest_files"][:5]):
        print(f"   {i+1}. {Path(file_path).name}: {line_count:,} lines")
    
    # Configure embedding system
    config = ChunkedEmbeddingConfig(
        vector_dim=128,
        max_function_lines=500,
        max_ram_gb=2.0,
        use_parallel=True,
        max_workers=4,
        batch_size=10
    )
    
    # Estimate memory requirements
    memory_estimate = estimate_memory_requirements(stats, config)
    print(f"\n💾 Memory Requirements:")
    print(f"   📊 Peak memory: {memory_estimate['peak_memory_mb']:.1f}MB")
    print(f"   📊 Total storage: {memory_estimate['total_storage_mb']:.1f}MB")
    print(f"   📊 Function embeddings: {memory_estimate['function_embeddings_count']:,}")
    print(f"   📊 File embeddings: {memory_estimate['file_embeddings_count']:,}")
    
    if memory_estimate['peak_memory_mb'] > config.max_ram_gb * 1024:
        print(f"   ⚠️ Peak memory exceeds limit! Consider reducing batch_size or max_workers")
    else:
        print(f"   ✅ Memory requirements within limits")
    
    # Start embedding
    print(f"\n🔧 Starting hierarchical embedding...")
    embedder = HierarchicalEmbedder(config)
    
    try:
        start_time = time.time()
        
        # Process files
        file_infos = []
        processed_files = 0
        processed_functions = 0
        
        for file_info in embedder.embed_directory(directory, "*.py"):
            file_infos.append(file_info)
            processed_files += 1
            processed_functions += len(file_info.functions)
            
            # Progress update
            if processed_files % 10 == 0:
                elapsed = time.time() - start_time
                rate = processed_files / elapsed if elapsed > 0 else 0
                print(f"   📈 Processed {processed_files}/{stats['python_files']} files "
                      f"({processed_functions} functions) - {rate:.1f} files/sec")
        
        end_time = time.time()
        
        print(f"\n✅ Embedding completed!")
        print(f"   📊 Files processed: {processed_files}")
        print(f"   📊 Functions/chunks: {processed_functions}")
        print(f"   📊 Total time: {end_time - start_time:.1f}s")
        print(f"   📊 Processing rate: {processed_files/(end_time - start_time):.1f} files/sec")
        
        # Memory stats
        memory_stats = embedder.get_memory_stats()
        print(f"   💾 Final memory: {memory_stats['current_memory_mb']:.1f}MB")
        print(f"   💾 Peak memory: {memory_stats['peak_memory_mb']:.1f}MB")
        
        # Save embeddings
        print(f"\n💾 Saving embeddings to {output_path}...")
        embedder.save_embeddings(file_infos, output_path)
        
        # Show some statistics
        if file_infos:
            embeddings = [fi.file_embedding for fi in file_infos if fi.file_embedding is not None]
            if embeddings:
                embeddings_array = np.array(embeddings)
                print(f"\n📊 Embedding Statistics:")
                print(f"   📈 Mean embedding norm: {np.mean([np.linalg.norm(emb) for emb in embeddings]):.3f}")
                print(f"   📈 Embedding std: {np.std([np.linalg.norm(emb) for emb in embeddings]):.3f}")
                print(f"   📈 Embedding shape: {embeddings_array.shape}")
        
        print(f"\n🎉 Hierarchical embedding demo completed!")
        print(f"📁 Output file: {output_path}")
        
    except Exception as e:
        print(f"\n❌ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        embedder.close()

def demo_similarity_search(embeddings_file: str):
    """Demonstrate similarity search on embedded codebase."""
    print(f"\n🔍 Loading embeddings for similarity search...")
    
    try:
        import h5py
        
        with h5py.File(embeddings_file, 'r') as f:
            file_embeddings = f['file_embeddings'][:]
            file_paths = f['file_paths'][:]
            
        print(f"✅ Loaded {len(file_embeddings)} file embeddings")
        
        # Simple similarity search
        print(f"\n🔍 Finding similar files...")
        
        # Use first embedding as query
        query_embedding = file_embeddings[0]
        query_file = Path(file_paths[0]).name
        
        print(f"   🔍 Query file: {query_file}")
        
        # Compute similarities
        similarities = []
        for i, embedding in enumerate(file_embeddings):
            if i == 0:  # Skip self
                continue
            
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            similarities.append((Path(file_paths[i]).name, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   📋 Most similar files:")
        for i, (file_name, sim) in enumerate(similarities[:5]):
            print(f"      {i+1}. {file_name}: {sim:.3f}")
        
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")

def main():
    """Run the hierarchical embedding demo."""
    setup_logging()
    
    # Demo with current directory (or specify a path)
    demo_directory = "."  # Change this to point to TheAlgorithms/Python if available
    
    print("🚀 HSC Hierarchical Embedding Demo")
    print("="*60)
    print(f"📁 Target directory: {demo_directory}")
    print(f"💡 Tip: Change demo_directory to point to TheAlgorithms/Python for a real test")
    
    # Run demo
    output_file = "demo_embeddings.h5"
    demo_hierarchical_embedding(demo_directory, output_file)
    
    # Demo similarity search if embeddings were created
    if os.path.exists(output_file):
        demo_similarity_search(output_file)
    
    print(f"\n" + "="*60)
    print("🎉 Demo completed!")
    print("📊 The HSC system can now scale to large codebases!")
    print("="*60)

if __name__ == "__main__":
    main() 