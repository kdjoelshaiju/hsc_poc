"""
AST-to-Vector Encoder

This module provides functionality to convert Abstract Syntax Trees (ASTs)
into fixed-dimensional vector representations that preserve semantic relationships.
"""

import numpy as np
import hashlib
from typing import Any, Dict, List, Tuple
import json

# Default vector dimension for embeddings
VECTOR_DIM = 128

def encode_ast(ast: Any, vector_dim: int = VECTOR_DIM) -> np.ndarray:
    """
    Encode an AST into a fixed-dimensional vector.
    
    Args:
        ast: Abstract Syntax Tree (tuple/list structure)
        vector_dim: Dimension of the output vector
        
    Returns:
        Fixed-dimensional vector representation
    """
    # Initialize vector with zeros
    vector = np.zeros(vector_dim)
    
    # Convert AST to string representation for hashing
    ast_str = _ast_to_string(ast)
    
    # Use hash-based encoding for deterministic results
    hash_value = hashlib.md5(ast_str.encode()).hexdigest()
    
    # Convert hash to vector using simple mapping
    for i, char in enumerate(hash_value):
        if i >= vector_dim:
            break
        # Map hex character to float value
        vector[i] = int(char, 16) / 15.0  # Normalize to [0, 1]
    
    # Add structural information
    structural_features = _extract_structural_features(ast)
    for i, feature in enumerate(structural_features):
        if i < vector_dim // 2:  # Use second half for structural features
            vector[vector_dim // 2 + i] = feature
    
    return vector

def _ast_to_string(ast: Any) -> str:
    """Convert AST to string representation for hashing."""
    if isinstance(ast, tuple):
        return f"({ast[0]}, " + ", ".join(_ast_to_string(child) for child in ast[1:]) + ")"
    elif isinstance(ast, list):
        return f"[{', '.join(_ast_to_string(item) for item in ast)}]"
    else:
        return str(ast)

def _extract_structural_features(ast: Any) -> List[float]:
    """Extract structural features from AST."""
    features = []
    
    # Count different node types
    node_counts = _count_node_types(ast)
    total_nodes = sum(node_counts.values())
    
    # Normalize counts to [0, 1]
    for node_type in ['func_def', 'if_stmt', 'while_stmt', 'assignment', 'return_stmt', 'func_call', 'expr_binop', 'atom']:
        count = node_counts.get(node_type, 0)
        features.append(count / max(total_nodes, 1))
    
    # Add depth information
    max_depth = _get_max_depth(ast)
    features.append(min(max_depth / 10.0, 1.0))  # Normalize depth
    
    # Add branching factor
    avg_branching = _get_avg_branching(ast)
    features.append(min(avg_branching / 5.0, 1.0))  # Normalize branching
    
    return features

def _count_node_types(ast: Any) -> Dict[str, int]:
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

def _get_max_depth(ast: Any) -> int:
    """Get maximum depth of the AST."""
    def traverse(node, depth=0):
        if isinstance(node, tuple) and len(node) > 1:
            return max(traverse(child, depth + 1) for child in node[1:])
        elif isinstance(node, list):
            return max(traverse(item, depth + 1) for item in node) if node else depth
        else:
            return depth
    
    return traverse(ast)

def _get_avg_branching(ast: Any) -> float:
    """Get average branching factor of the AST."""
    total_children = 0
    total_nodes = 0
    
    def traverse(node):
        nonlocal total_children, total_nodes
        
        if isinstance(node, tuple) and len(node) > 1:
            total_nodes += 1
            children = len(node) - 1
            total_children += children
            for child in node[1:]:
                traverse(child)
        elif isinstance(node, list):
            total_nodes += 1
            children = len(node)
            total_children += children
            for item in node:
                traverse(item)
        else:
            total_nodes += 1
    
    traverse(ast)
    return total_children / max(total_nodes, 1)

def encode_token(token: str, vector_dim: int = 32) -> np.ndarray:
    """
    Encode a single token into a vector.
    
    Args:
        token: String token to encode
        vector_dim: Dimension of the output vector
        
    Returns:
        Fixed-dimensional vector representation
    """
    # Simple hash-based encoding
    hash_value = hashlib.md5(token.encode()).hexdigest()
    vector = np.zeros(vector_dim)
    
    for i, char in enumerate(hash_value):
        if i >= vector_dim:
            break
        vector[i] = int(char, 16) / 15.0
    
    return vector

def encode_sequence(tokens: List[str], vector_dim: int = VECTOR_DIM) -> np.ndarray:
    """
    Encode a sequence of tokens into a single vector.
    
    Args:
        tokens: List of string tokens
        vector_dim: Dimension of the output vector
        
    Returns:
        Fixed-dimensional vector representation
    """
    if not tokens:
        return np.zeros(vector_dim)
    
    # Encode each token
    token_vectors = [encode_token(token, vector_dim // 2) for token in tokens]
    
    # Combine using simple averaging
    combined = np.mean(token_vectors, axis=0)
    
    # Pad or truncate to target dimension
    if len(combined) < vector_dim:
        combined = np.pad(combined, (0, vector_dim - len(combined)))
    else:
        combined = combined[:vector_dim]
    
    return combined 