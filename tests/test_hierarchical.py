"""
Test script for Hierarchical & Chunked Embeddings

This script demonstrates the memory-efficient embedding of large codebases
using function-level chunking and streaming pipeline.
"""

import os
import logging
import time
from pathlib import Path
from embedding.hierarchical import HierarchicalEmbedder, ChunkedEmbeddingConfig
import numpy as np

def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def create_test_files():
    """Create some test Python files to demonstrate the system."""
    test_dir = Path("test_codebase")
    test_dir.mkdir(exist_ok=True)
    
    # Create a simple file
    simple_file = test_dir / "simple.py"
    simple_file.write_text("""
def add(a, b):
    return a + b

def multiply(x, y):
    return x * y

def greet(name):
    return f"Hello, {name}!"
""")
    
    # Create a file with larger functions
    complex_file = test_dir / "complex.py"
    complex_file.write_text("""
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

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

class Calculator:
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def subtract(self, a, b):
        result = a - b
        self.history.append(f"{a} - {b} = {result}")
        return result
    
    def get_history(self):
        return self.history
""")
    
    # Create a file with a very large function (to test chunking)
    large_file = test_dir / "large_function.py"
    large_content = """
def very_large_function():
    # This function is intentionally large to test chunking
    result = 0
"""
    # Add many lines to make it large
    for i in range(600):  # More than max_function_lines (500)
        large_content += f"    result += {i}\n"
    large_content += "    return result\n"
    
    large_file.write_text(large_content)
    
    print(f"✅ Created test files in {test_dir}")
    return test_dir

def test_single_file_embedding():
    """Test embedding a single file."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Single File Embedding")
    print("="*60)
    
    config = ChunkedEmbeddingConfig(
        vector_dim=128,
        max_function_lines=500,
        max_ram_gb=2.0,
        use_parallel=False  # Sequential for testing
    )
    
    embedder = HierarchicalEmbedder(config)
    
    try:
        # Test with a simple file
        test_file = "test_codebase/simple.py"
        if os.path.exists(test_file):
            print(f"📁 Embedding file: {test_file}")
            
            start_time = time.time()
            file_info = embedder.embed_file(test_file)
            end_time = time.time()
            
            print(f"✅ File embedded successfully!")
            print(f"   📊 Path: {file_info.path}")
            print(f"   📊 Functions found: {len(file_info.functions)}")
            if file_info.file_embedding is not None:
                print(f"   📊 File embedding shape: {file_info.file_embedding.shape}")
                print(f"   📊 File embedding norm: {np.linalg.norm(file_info.file_embedding):.3f}")
            else:
                print(f"   📊 File embedding: None")
            print(f"   📊 Time taken: {end_time - start_time:.2f}s")
            
            # Show function details
            for i, func in enumerate(file_info.functions):
                print(f"   📋 Function {i+1}: {func.name}")
                print(f"      Lines: {func.start_line}-{func.end_line}")
                if func.embedding is not None:
                    print(f"      Embedding norm: {np.linalg.norm(func.embedding):.3f}")
                else:
                    print(f"      Embedding: None")
            
            # Memory stats
            memory_stats = embedder.get_memory_stats()
            print(f"   💾 Memory usage: {memory_stats['current_memory_mb']:.1f}MB")
            print(f"   💾 Peak memory: {memory_stats['peak_memory_mb']:.1f}MB")
            
        else:
            print(f"❌ Test file not found: {test_file}")
    
    finally:
        embedder.close()

def test_directory_embedding():
    """Test embedding an entire directory."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Directory Embedding")
    print("="*60)
    
    config = ChunkedEmbeddingConfig(
        vector_dim=128,
        max_function_lines=500,
        max_ram_gb=2.0,
        use_parallel=True,
        max_workers=2
    )
    
    embedder = HierarchicalEmbedder(config)
    
    try:
        test_dir = "test_codebase"
        if os.path.exists(test_dir):
            print(f"📁 Embedding directory: {test_dir}")
            
            start_time = time.time()
            file_infos = list(embedder.embed_directory(test_dir))
            end_time = time.time()
            
            print(f"✅ Directory embedded successfully!")
            print(f"   📊 Files processed: {len(file_infos)}")
            print(f"   📊 Total time: {end_time - start_time:.2f}s")
            
            # Show file details
            total_functions = 0
            for file_info in file_infos:
                print(f"   📋 File: {Path(file_info.path).name}")
                print(f"      Functions: {len(file_info.functions)}")
                if file_info.file_embedding is not None:
                    print(f"      Embedding norm: {np.linalg.norm(file_info.file_embedding):.3f}")
                else:
                    print(f"      Embedding: None")
                total_functions += len(file_info.functions)
            
            print(f"   📊 Total functions: {total_functions}")
            
            # Memory stats
            memory_stats = embedder.get_memory_stats()
            print(f"   💾 Memory usage: {memory_stats['current_memory_mb']:.1f}MB")
            print(f"   💾 Peak memory: {memory_stats['peak_memory_mb']:.1f}MB")
            
        else:
            print(f"❌ Test directory not found: {test_dir}")
    
    finally:
        embedder.close()

def test_caching():
    """Test caching functionality."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Caching")
    print("="*60)
    
    config = ChunkedEmbeddingConfig(
        vector_dim=128,
        max_function_lines=500,
        max_ram_gb=2.0,
        cache_dir="embeddings_cache"
    )
    
    embedder = HierarchicalEmbedder(config)
    
    try:
        test_file = "test_codebase/simple.py"
        if os.path.exists(test_file):
            print(f"📁 First embedding (should compute): {test_file}")
            start_time = time.time()
            file_info1 = embedder.embed_file(test_file)
            time1 = time.time() - start_time
            
            print(f"📁 Second embedding (should use cache): {test_file}")
            start_time = time.time()
            file_info2 = embedder.embed_file(test_file)
            time2 = time.time() - start_time
            
            print(f"✅ Caching test completed!")
            print(f"   📊 First run time: {time1:.3f}s")
            print(f"   📊 Second run time: {time2:.3f}s")
            print(f"   📊 Speedup: {time1/time2:.1f}x")
            
            # Verify embeddings are the same
            if (file_info1.file_embedding is not None and 
                file_info2.file_embedding is not None):
                diff = np.linalg.norm(file_info1.file_embedding - file_info2.file_embedding)
                print(f"   📊 Embedding difference: {diff:.6f}")
                if diff < 1e-6:
                    print(f"   ✅ Cached embeddings match!")
                else:
                    print(f"   ⚠️ Cached embeddings differ!")
            
        else:
            print(f"❌ Test file not found: {test_file}")
    
    finally:
        embedder.close()

def test_memory_efficiency():
    """Test memory efficiency with large files."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Memory Efficiency")
    print("="*60)
    
    config = ChunkedEmbeddingConfig(
        vector_dim=128,
        max_function_lines=500,
        max_ram_gb=2.0,
        use_parallel=False
    )
    
    embedder = HierarchicalEmbedder(config)
    
    try:
        # Test with large function file
        test_file = "test_codebase/large_function.py"
        if os.path.exists(test_file):
            print(f"📁 Embedding large file: {test_file}")
            
            # Get initial memory
            initial_memory = embedder.get_memory_stats()['current_memory_mb']
            print(f"   💾 Initial memory: {initial_memory:.1f}MB")
            
            start_time = time.time()
            file_info = embedder.embed_file(test_file)
            end_time = time.time()
            
            # Get final memory
            final_memory = embedder.get_memory_stats()['current_memory_mb']
            memory_increase = final_memory - initial_memory
            
            print(f"✅ Large file embedded successfully!")
            print(f"   📊 Functions found: {len(file_info.functions)}")
            print(f"   📊 Time taken: {end_time - start_time:.2f}s")
            print(f"   💾 Final memory: {final_memory:.1f}MB")
            print(f"   💾 Memory increase: {memory_increase:.1f}MB")
            
            if memory_increase < 100:  # Less than 100MB increase
                print(f"   ✅ Memory efficient!")
            else:
                print(f"   ⚠️ High memory usage!")
            
        else:
            print(f"❌ Test file not found: {test_file}")
    
    finally:
        embedder.close()

def main():
    """Run all hierarchical embedding tests."""
    setup_logging()
    
    print("🚀 HSC Hierarchical & Chunked Embedding Test")
    print("="*60)
    
    try:
        # Create test files
        create_test_files()
        
        # Run tests
        test_single_file_embedding()
        test_directory_embedding()
        test_caching()
        test_memory_efficiency()
        
        print("\n" + "="*60)
        print("✅ All hierarchical embedding tests completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 