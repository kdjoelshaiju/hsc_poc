"""
DSL-to-Vector Embedding Engine

This module provides the main interface for converting DSL programs into
fixed-dimensional vector representations that preserve semantic relationships.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import json

from .encoder import encode_ast, VECTOR_DIM
from dsl.parser import parse_program


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding engine."""
    vector_dim: int = VECTOR_DIM
    normalize: bool = True
    use_semantic_weights: bool = True
    include_structure: bool = True
    seed: int = 42


class DSLToVectorEngine:
    """
    Main engine for converting DSL programs to vector embeddings.
    
    This class orchestrates the parsing, encoding, and normalization
    of DSL programs into semantic vector representations.
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.program_cache: Dict[str, np.ndarray] = {}
        self.semantic_weights = self._initialize_semantic_weights()
        
    def _initialize_semantic_weights(self) -> Dict[str, float]:
        """Initialize weights for different AST node types based on semantic importance."""
        return {
            'func_def': 2.0,      # Function definitions are semantically important
            'if_stmt': 1.5,       # Control flow is important
            'while_stmt': 1.5,    # Loops are important
            'assignment': 1.0,    # Basic assignments
            'return_stmt': 1.2,   # Return statements carry semantic meaning
            'func_call': 1.3,     # Function calls are important
            'expr_binop': 0.8,    # Binary operations
            'atom': 0.5,          # Basic atoms (variables, numbers)
        }
    
    def encode_program(self, program_code: str, program_id: Optional[str] = None) -> np.ndarray:
        """
        Encode a DSL program into a vector representation.
        
        Args:
            program_code: The DSL program as a string
            program_id: Optional identifier for caching
            
        Returns:
            Fixed-dimensional vector representation
        """
        if program_id and program_id in self.program_cache:
            return self.program_cache[program_id]
        
        # Parse the program
        try:
            ast = parse_program(program_code)
        except Exception as e:
            raise ValueError(f"Failed to parse program: {e}")
        
        # Encode the AST
        raw_vector = encode_ast(ast)
        
        # Apply semantic weighting if enabled
        if self.config.use_semantic_weights:
            weighted_vector = self._apply_semantic_weights(ast, raw_vector)
        else:
            weighted_vector = raw_vector
        
        # Normalize if requested
        if self.config.normalize:
            final_vector = self._normalize_vector(weighted_vector)
        else:
            final_vector = weighted_vector
        
        # Cache the result
        if program_id:
            self.program_cache[program_id] = final_vector
            
        return final_vector
    
    def _apply_semantic_weights(self, ast: Any, vector: np.ndarray) -> np.ndarray:
        """Apply semantic weights based on AST structure."""
        weight_factor = 1.0
        
        # Count different node types and apply weights
        node_counts = self._count_node_types(ast)
        for node_type, count in node_counts.items():
            if node_type in self.semantic_weights:
                weight_factor *= (self.semantic_weights[node_type] ** count)
        
        return vector * weight_factor
    
    def _count_node_types(self, ast: Any) -> Dict[str, int]:
        """Count occurrences of different node types in the AST."""
        counts = {}
        
        def traverse(node):
            if isinstance(node, tuple) and len(node) > 0:
                node_type = str(node[0])
                counts[node_type] = counts.get(node_type, 0) + 1
                for child in node[1:]:
                    traverse(child)
            elif isinstance(node, list):
                for item in node:
                    traverse(item)
        
        traverse(ast)
        return counts
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def encode_batch(self, programs: List[Tuple[str, Optional[str]]]) -> np.ndarray:
        """
        Encode multiple programs in batch.
        
        Args:
            programs: List of (program_code, program_id) tuples
            
        Returns:
            Matrix of embeddings, shape (n_programs, vector_dim)
        """
        embeddings = []
        for program_code, program_id in programs:
            embedding = self.encode_program(program_code, program_id)
            embeddings.append(embedding)
        
        return np.stack(embeddings)
    
    def get_similarity_matrix(self, programs: List[str]) -> np.ndarray:
        """
        Compute pairwise similarity matrix for a list of programs.
        
        Args:
            programs: List of program codes
            
        Returns:
            Similarity matrix, shape (n_programs, n_programs)
        """
        embeddings = self.encode_batch([(p, None) for p in programs])
        
        # Compute cosine similarity matrix
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarity_matrix = normalized_embeddings @ normalized_embeddings.T
        
        return similarity_matrix
    
    def find_most_similar(self, query_program: str, candidate_programs: List[str], 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the most similar programs to a query program.
        
        Args:
            query_program: The program to find similarities for
            candidate_programs: List of candidate programs
            top_k: Number of most similar programs to return
            
        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        query_embedding = self.encode_program(query_program)
        
        similarities = []
        for i, candidate in enumerate(candidate_programs):
            candidate_embedding = self.encode_program(candidate)
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, candidate_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(candidate_embedding)
            )
            similarities.append((i, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def save_embeddings(self, filepath: str):
        """Save cached embeddings to disk."""
        data = {
            'config': self.config.__dict__,
            'embeddings': {k: v.tolist() for k, v in self.program_cache.items()}
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_embeddings(self, filepath: str):
        """Load cached embeddings from disk."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = EmbeddingConfig(**data['config'])
        self.program_cache = {k: np.array(v) for k, v in data['embeddings'].items()}


# Convenience function for quick encoding
def encode_program(program_code: str, config: Optional[EmbeddingConfig] = None) -> np.ndarray:
    """Quick function to encode a single program."""
    engine = DSLToVectorEngine(config)
    return engine.encode_program(program_code) 