"""
Semantic Code Search & Transformation Engine

This module provides a comprehensive benchmarking and search system for
semantic code analysis, transformation, and visualization.
"""

import json
import os
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import requests
import tempfile
import zipfile
from urllib.parse import urlparse

from embedding.engine import DSLToVectorEngine, EmbeddingConfig
from embedding.hierarchical import HierarchicalEmbedder
from interpolation.geodesic import GeodesicInterpolator
from evaluation.similarity import SimilarityEvaluator
from evaluation.metrics import SemanticMetrics


@dataclass
class CodeSnippet:
    """Represents a code snippet with its metadata and embeddings."""
    name: str
    code: str
    category: str
    description: str
    ast: Optional[Dict] = None
    vector: Optional[np.ndarray] = None
    similarity_scores: Optional[Dict[str, float]] = None

@dataclass
class SearchResult:
    """Represents a search result with similarity information."""
    snippet: CodeSnippet
    similarity_score: float
    rank: int

@dataclass
class TransformationPath:
    """Represents a transformation path between two programs."""
    start: CodeSnippet
    end: CodeSnippet
    intermediate_steps: List[CodeSnippet]
    path_vectors: List[np.ndarray]
    confidence_scores: List[float]

class SemanticSearchEngine:
    """Semantic code search and transformation engine."""
    
    def __init__(self, config: EmbeddingConfig = None):
        """Initialize the search engine."""
        self.config = config or EmbeddingConfig(vector_dim=128)
        self.engine = DSLToVectorEngine(self.config)
        self.interpolator = GeodesicInterpolator(manifold=None)
        self.similarity_evaluator = SimilarityEvaluator()
        self.metrics = SemanticMetrics()
        
        self.search_index: List[CodeSnippet] = []
        self.index_path = Path("data/search_index.json")
        
        # Ensure data directory exists
        self.index_path.parent.mkdir(exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def build_search_corpus(self, algorithms_source: str = "thealgorithms") -> None:
        """Build the search corpus from various algorithm sources."""
        self.logger.info("Building search corpus...")
        
        if algorithms_source == "thealgorithms":
            self._build_thealgorithms_corpus()
        elif algorithms_source == "local":
            self._build_local_corpus()
        else:
            raise ValueError(f"Unknown algorithms source: {algorithms_source}")
        
        self._compute_embeddings()
        self._save_index()
        self.logger.info(f"Search corpus built with {len(self.search_index)} algorithms")
    
    def _build_thealgorithms_corpus(self) -> None:
        """Build corpus from TheAlgorithms/Python repository."""
        self.logger.info("Building corpus from TheAlgorithms/Python...")
        
        # Curated list of diverse algorithms for initial corpus
        algorithms = [
            # Graph Algorithms
            ("dfs_recursive", "graphs/dfs.py", "Depth-First Search (Recursive)", "graph"),
            ("dfs_iterative", "graphs/dfs_iterative.py", "Depth-First Search (Iterative)", "graph"),
            ("bfs", "graphs/bfs.py", "Breadth-First Search", "graph"),
            ("dijkstra", "graphs/dijkstra.py", "Dijkstra's Shortest Path", "graph"),
            ("bellman_ford", "graphs/bellman_ford.py", "Bellman-Ford Algorithm", "graph"),
            
            # Sorting Algorithms
            ("bubble_sort", "sorts/bubble_sort.py", "Bubble Sort", "sort"),
            ("quick_sort", "sorts/quick_sort.py", "Quick Sort", "sort"),
            ("merge_sort", "sorts/merge_sort.py", "Merge Sort", "sort"),
            ("heap_sort", "sorts/heap_sort.py", "Heap Sort", "sort"),
            
            # Search Algorithms
            ("binary_search", "searches/binary_search.py", "Binary Search", "search"),
            ("linear_search", "searches/linear_search.py", "Linear Search", "search"),
            
            # Mathematical Algorithms
            ("factorial", "maths/factorial.py", "Factorial", "math"),
            ("fibonacci", "maths/fibonacci.py", "Fibonacci Sequence", "math"),
            ("gcd", "maths/gcd.py", "Greatest Common Divisor", "math"),
            ("lcm", "maths/lcm.py", "Least Common Multiple", "math"),
            
            # Data Structures
            ("linked_list", "data_structures/linked_list.py", "Linked List", "data_structure"),
            ("binary_tree", "data_structures/binary_tree.py", "Binary Tree", "data_structure"),
            ("stack", "data_structures/stack.py", "Stack Implementation", "data_structure"),
            ("queue", "data_structures/queue.py", "Queue Implementation", "data_structure"),
        ]
        
        # For now, create placeholder implementations
        # In a real implementation, you'd fetch from the actual repository
        for name, path, description, category in algorithms:
            code = self._get_algorithm_implementation(name, path)
            snippet = CodeSnippet(
                name=name,
                code=code,
                category=category,
                description=description
            )
            self.search_index.append(snippet)
    
    def _get_algorithm_implementation(self, name: str, path: str) -> str:
        """Get algorithm implementation (placeholder for now)."""
        # Placeholder implementations - in reality, you'd fetch from GitHub
        implementations = {
            "dfs_recursive": """
def dfs_recursive(graph, start, visited=None):
    if visited is None:
        visited = set()
    visited.add(start)
    print(start, end=' ')
    for neighbor in graph[start]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
""",
            "dfs_iterative": """
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            print(vertex, end=' ')
            stack.extend(reversed(graph[vertex]))
""",
            "bfs": """
def bfs(graph, start):
    visited = set()
    queue = [start]
    visited.add(start)
    while queue:
        vertex = queue.pop(0)
        print(vertex, end=' ')
        for neighbor in graph[vertex]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
""",
            "bubble_sort": """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
""",
            "quick_sort": """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
""",
            "binary_search": """
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
""",
            "factorial": """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
""",
            "fibonacci": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
""",
            "linked_list": """
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None
    
    def append(self, data):
        new_node = Node(data)
        if not self.head:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
""",
            "stack": """
class Stack:
    def __init__(self):
        self.items = []
    
    def push(self, item):
        self.items.append(item)
    
    def pop(self):
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def is_empty(self):
        return len(self.items) == 0
"""
        }
        
        return implementations.get(name, f"# Placeholder for {name}\ndef {name}():\n    pass")
    
    def _compute_embeddings(self) -> None:
        """Compute embeddings for all code snippets."""
        self.logger.info("Computing embeddings...")
        
        for snippet in self.search_index:
            try:
                # Encode the program
                encode_result = self.engine.encode_program(snippet.code, snippet.name)
                snippet.vector = encode_result.vector
                snippet.ast = encode_result.ast
                
                self.logger.debug(f"Encoded {snippet.name}: {len(snippet.vector)} dimensions")
                
            except Exception as e:
                self.logger.warning(f"Failed to encode {snippet.name}: {e}")
                # Use zero vector as fallback
                snippet.vector = np.zeros(self.config.vector_dim)
    
    def _save_index(self) -> None:
        """Save the search index to disk."""
        self.logger.info(f"Saving search index to {self.index_path}")
        
        # Convert to serializable format
        index_data = []
        for snippet in self.search_index:
            data = {
                "name": snippet.name,
                "code": snippet.code,
                "category": snippet.category,
                "description": snippet.description,
                "vector": snippet.vector.tolist() if snippet.vector is not None else None,
                "ast": str(snippet.ast) if snippet.ast is not None else None  # Convert AST to string
            }
            index_data.append(data)
        
        with open(self.index_path, 'w') as f:
            json.dump(index_data, f, indent=2)
    
    def load_index(self) -> None:
        """Load the search index from disk."""
        if not self.index_path.exists():
            self.logger.warning("Search index not found. Run build_search_corpus() first.")
            return
        
        self.logger.info(f"Loading search index from {self.index_path}")
        
        with open(self.index_path, 'r') as f:
            index_data = json.load(f)
        
        self.search_index = []
        for data in index_data:
            snippet = CodeSnippet(
                name=data["name"],
                code=data["code"],
                category=data["category"],
                description=data["description"],
                vector=np.array(data["vector"]) if data["vector"] else None,
                ast=data["ast"]
            )
            self.search_index.append(snippet)
        
        self.logger.info(f"Loaded {len(self.search_index)} algorithms")
    
    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search for semantically similar code snippets."""
        if not self.search_index:
            self.load_index()
        
        # Encode the query
        try:
            encode_result = self.engine.encode_program(query, "query")
            query_vector = encode_result.vector
        except Exception as e:
            self.logger.error(f"Failed to encode query: {e}")
            return []
        
        # Compute similarities
        similarities = []
        for snippet in self.search_index:
            if snippet.vector is not None:
                similarity = self.similarity_evaluator._compute_cosine_similarity(
                    query_vector, snippet.vector
                )
                similarities.append((snippet, similarity))
        
        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (snippet, similarity) in enumerate(similarities[:top_k]):
            result = SearchResult(
                snippet=snippet,
                similarity_score=similarity,
                rank=i + 1
            )
            results.append(result)
        
        return results
    
    def transform(self, start_name: str, end_name: str, steps: int = 5) -> Optional[TransformationPath]:
        """Transform one program into another via interpolation."""
        if not self.search_index:
            self.load_index()
        
        # Find the programs
        start_snippet = next((s for s in self.search_index if s.name == start_name), None)
        end_snippet = next((s for s in self.search_index if s.name == end_name), None)
        
        if not start_snippet or not end_snippet:
            self.logger.error(f"Could not find programs: {start_name}, {end_name}")
            return None
        
        if start_snippet.vector is None or end_snippet.vector is None:
            self.logger.error("Programs not properly encoded")
            return None
        
        try:
            # Interpolate between vectors using linear interpolation
            alphas = np.linspace(0, 1, steps)
            path_vectors = []
            
            for alpha in alphas:
                interpolated_vector = (1 - alpha) * start_snippet.vector + alpha * end_snippet.vector
                path_vectors.append(interpolated_vector)
            
            # Decode intermediate steps
            intermediate_steps = []
            confidence_scores = []
            
            for i, vector in enumerate(path_vectors):
                try:
                    decode_result = self.engine.decode_program(vector)
                    if decode_result.success:
                        # Get the decoded program text
                        decoded_program = getattr(decode_result, 'program', f"# Interpolated step {i}")
                        intermediate_snippet = CodeSnippet(
                            name=f"step_{i}",
                            code=decoded_program,
                            category="intermediate",
                            description=f"Step {i} in transformation"
                        )
                        intermediate_steps.append(intermediate_snippet)
                        confidence_scores.append(decode_result.confidence)
                    else:
                        # Use fallback
                        intermediate_steps.append(CodeSnippet(
                            name=f"step_{i}",
                            code=f"# Failed to decode step {i}",
                            category="intermediate",
                            description=f"Step {i} (decode failed)"
                        ))
                        confidence_scores.append(0.0)
                        
                except Exception as e:
                    self.logger.warning(f"Failed to decode step {i}: {e}")
                    intermediate_steps.append(CodeSnippet(
                        name=f"step_{i}",
                        code=f"# Error in step {i}",
                        category="intermediate",
                        description=f"Step {i} (error)"
                    ))
                    confidence_scores.append(0.0)
            
            return TransformationPath(
                start=start_snippet,
                end=end_snippet,
                intermediate_steps=intermediate_steps,
                path_vectors=path_vectors,
                confidence_scores=confidence_scores
            )
            
        except Exception as e:
            self.logger.error(f"Transformation failed: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the search corpus."""
        if not self.search_index:
            self.load_index()
        
        categories = {}
        for snippet in self.search_index:
            cat = snippet.category
            categories[cat] = categories.get(cat, 0) + 1
        
        total_code_length = sum(len(snippet.code) for snippet in self.search_index)
        avg_code_length = total_code_length / len(self.search_index) if self.search_index else 0
        
        return {
            "total_algorithms": len(self.search_index),
            "categories": categories,
            "total_code_length": total_code_length,
            "average_code_length": avg_code_length,
            "vector_dimension": self.config.vector_dim
        }

class BenchmarkSuite:
    """Suite of benchmarks for HSC evaluation."""
    
    def __init__(self):
        self.search_engine = SemanticSearchEngine()
    
    def run_benchmarks(self):
        """Run all benchmarks."""
        print("🚀 Running HSC Semantic Search Benchmarks")
        print("=" * 60)
        
        # Build corpus
        print("\n📚 Building search corpus...")
        self.search_engine.build_search_corpus()
        
        # Get statistics
        stats = self.search_engine.get_statistics()
        print(f"\n📊 Corpus Statistics:")
        print(f"   Total algorithms: {stats['total_algorithms']}")
        print(f"   Categories: {stats['categories']}")
        print(f"   Average code length: {stats['average_code_length']:.1f} chars")
        print(f"   Vector dimension: {stats['vector_dimension']}")
        
        # Test search functionality
        print(f"\n🔍 Testing semantic search...")
        test_queries = [
            "def search(graph, start):",
            "def sort(array):",
            "def find_element(list, target):",
            "class DataStructure:"
        ]
        
        for query in test_queries:
            print(f"\n   Query: {query}")
            results = self.search_engine.search(query, top_k=3)
            for result in results:
                print(f"     {result.rank}. {result.snippet.name} ({result.similarity_score:.3f})")
        
        # Test transformation
        print(f"\n🔄 Testing code transformation...")
        transformations = [
            ("dfs_recursive", "dfs_iterative"),
            ("bubble_sort", "quick_sort"),
            ("linked_list", "stack")
        ]
        
        for start, end in transformations:
            print(f"\n   Transforming {start} → {end}")
            path = self.search_engine.transform(start, end, steps=3)
            if path:
                print(f"     ✅ Transformation successful")
                print(f"     📊 Average confidence: {np.mean(path.confidence_scores):.3f}")
            else:
                print(f"     ❌ Transformation failed")
        
        print(f"\n" + "=" * 60)
        print("✅ All benchmarks completed!") 