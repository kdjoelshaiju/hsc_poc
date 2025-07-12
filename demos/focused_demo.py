"""
Focused Demo: Hierarchical Embedding with Memory Limits

This script demonstrates the HSC hierarchical embedding system
with strict memory limits (2GB) and focused processing.
"""

import os
import logging
import time
import psutil
from pathlib import Path
from embedding.hierarchical import HierarchicalEmbedder, ChunkedEmbeddingConfig
import numpy as np

def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def get_memory_usage():
    """Get current memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def create_small_test_set():
    """Create a small test set of Python files for demonstration."""
    test_dir = Path("demo_test_files")
    test_dir.mkdir(exist_ok=True)
    
    # Create a few focused test files
    files = {
        "sorting.py": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
        
        "searching.py": """
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def linear_search(arr, target):
    for i, item in enumerate(arr):
        if item == target:
            return i
    return -1
""",
        
        "graph.py": """
def dfs(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs(graph, neighbor, visited)
    return visited

def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        vertex = queue.pop(0)
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return visited
""",
        
        "math_utils.py": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a
"""
    }
    
    for filename, content in files.items():
        file_path = test_dir / filename
        file_path.write_text(content)
    
    print(f"✅ Created {len(files)} test files in {test_dir}")
    return test_dir

def demo_focused_embedding(test_dir: str, max_ram_mb: int = 2000):
    """Demonstrate focused hierarchical embedding with memory limits."""
    print("🚀 HSC Focused Hierarchical Embedding Demo")
    print("="*60)
    print(f"📁 Target directory: {test_dir}")
    print(f"💾 Memory limit: {max_ram_mb}MB")
    
    # Monitor initial memory
    initial_memory = get_memory_usage()
    print(f"💾 Initial memory usage: {initial_memory:.1f}MB")
    
    # Configure for memory efficiency
    config = ChunkedEmbeddingConfig(
        vector_dim=64,  # Smaller vectors for memory efficiency
        max_function_lines=200,  # Smaller chunks
        max_ram_gb=max_ram_mb / 1024,
        use_parallel=False,  # Sequential to avoid memory spikes
        max_workers=1,
        batch_size=5,  # Smaller batches
        window_size=256,  # Smaller windows
        window_stride=128
    )
    
    print(f"⚙️ Configuration:")
    print(f"   📊 Vector dimension: {config.vector_dim}")
    print(f"   📊 Max function lines: {config.max_function_lines}")
    print(f"   📊 Batch size: {config.batch_size}")
    print(f"   📊 Parallel processing: {config.use_parallel}")
    
    # Start embedding
    print(f"\n🔧 Starting focused embedding...")
    embedder = HierarchicalEmbedder(config)
    
    try:
        start_time = time.time()
        
        # Process files with memory monitoring
        file_infos = []
        processed_files = 0
        processed_functions = 0
        
        for file_info in embedder.embed_directory(test_dir, "*.py"):
            file_infos.append(file_info)
            processed_files += 1
            processed_functions += len(file_info.functions)
            
            # Memory monitoring
            current_memory = get_memory_usage()
            memory_increase = current_memory - initial_memory
            
            print(f"   📁 Processed: {Path(file_info.path).name}")
            print(f"      Functions: {len(file_info.functions)}")
            print(f"      Memory: {current_memory:.1f}MB (+{memory_increase:.1f}MB)")
            
            # Check memory limit
            if current_memory > max_ram_mb:
                print(f"   ⚠️ Memory limit exceeded! Stopping processing.")
                break
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.1)
        
        end_time = time.time()
        
        print(f"\n✅ Focused embedding completed!")
        print(f"   📊 Files processed: {processed_files}")
        print(f"   📊 Functions/chunks: {processed_functions}")
        print(f"   📊 Total time: {end_time - start_time:.1f}s")
        print(f"   📊 Processing rate: {processed_files/(end_time - start_time):.1f} files/sec")
        
        # Final memory stats
        final_memory = get_memory_usage()
        memory_used = final_memory - initial_memory
        print(f"   💾 Final memory: {final_memory:.1f}MB")
        print(f"   💾 Memory used: {memory_used:.1f}MB")
        print(f"   💾 Memory efficiency: {memory_used/processed_functions:.1f}MB/function")
        
        # Embedder memory stats
        embedder_stats = embedder.get_memory_stats()
        print(f"   💾 Embedder peak memory: {embedder_stats['peak_memory_mb']:.1f}MB")
        
        # Save embeddings if we have results
        if file_infos:
            output_path = "focused_embeddings.h5"
            embedder.save_embeddings(file_infos, output_path)
            print(f"   💾 Saved embeddings to: {output_path}")
            
            # Show embedding statistics
            embeddings = [fi.file_embedding for fi in file_infos if fi.file_embedding is not None]
            if embeddings:
                embeddings_array = np.array(embeddings)
                print(f"\n📊 Embedding Statistics:")
                print(f"   📈 Embedding shape: {embeddings_array.shape}")
                print(f"   📈 Mean embedding norm: {np.mean([np.linalg.norm(emb) for emb in embeddings]):.3f}")
                print(f"   📈 Embedding std: {np.std([np.linalg.norm(emb) for emb in embeddings]):.3f}")
        
        return file_infos
        
    except Exception as e:
        print(f"\n❌ Embedding failed: {e}")
        import traceback
        traceback.print_exc()
        return []
    
    finally:
        embedder.close()

def demo_similarity_search(embeddings_file: str):
    """Demonstrate similarity search on focused embeddings."""
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
        query_file = Path(str(file_paths[0])).name
        
        print(f"   🔍 Query file: {query_file}")
        
        # Compute similarities
        similarities = []
        for i, embedding in enumerate(file_embeddings):
            if i == 0:  # Skip self
                continue
            
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            file_path = str(file_paths[i])
            similarities.append((Path(file_path).name, similarity))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   📋 Most similar files:")
        for i, (file_name, sim) in enumerate(similarities[:3]):
            print(f"      {i+1}. {file_name}: {sim:.3f}")
        
    except Exception as e:
        print(f"❌ Similarity search failed: {e}")

def main():
    """Run the focused hierarchical embedding demo."""
    setup_logging()
    
    print("🚀 HSC Focused Hierarchical Embedding Demo")
    print("="*60)
    print("💡 This demo is optimized for 2GB RAM laptops")
    
    # Create small test set
    test_dir = create_small_test_set()
    
    # Run focused demo
    file_infos = demo_focused_embedding(str(test_dir), max_ram_mb=2000)
    
    # Demo similarity search if embeddings were created
    if file_infos and os.path.exists("focused_embeddings.h5"):
        demo_similarity_search("focused_embeddings.h5")
    
    print(f"\n" + "="*60)
    print("🎉 Focused demo completed!")
    print("📊 The HSC system works efficiently with memory constraints!")
    print("💡 Try changing the test files or configuration for different results")
    print("="*60)

if __name__ == "__main__":
    main() 