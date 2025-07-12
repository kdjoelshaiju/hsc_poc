"""
AST-to-Vector Encoder with Neural Embeddings

This module provides functionality to convert Abstract Syntax Trees (ASTs)
into fixed-dimensional vector representations that preserve semantic relationships.
Supports hybrid embeddings with neural models and enhanced semantic analysis.
"""

import numpy as np
import hashlib
from typing import Any, Dict, List, Optional

# Default vector dimension for embeddings
VECTOR_DIM = 128

class HybridEmbeddingConfig:
    """Configuration for hybrid embedding approach."""
    def __init__(self, 
                 vector_dim: int = VECTOR_DIM,
                 use_neural_embeddings: bool = False,
                 neural_model_name: str = "codebert",
                 hash_weight: float = 0.3,
                 structural_weight: float = 0.4,
                 semantic_weight: float = 0.3,
                 context_window: int = 5):
        self.vector_dim = vector_dim
        self.use_neural_embeddings = use_neural_embeddings
        self.neural_model_name = neural_model_name
        self.hash_weight = hash_weight
        self.structural_weight = structural_weight
        self.semantic_weight = semantic_weight
        self.context_window = context_window

class NeuralEmbeddingProvider:
    """Provider for neural embeddings (placeholder for CodeBERT/GraphCodeBERT)."""
    
    def __init__(self, model_name: str = "codebert"):
        self.model_name = model_name
        self.is_available = self._check_availability()
        
    def _check_availability(self) -> bool:
        """Check if neural model is available."""
        try:
            # Placeholder for actual model loading
            # In production, this would load transformers, torch, etc.
            return False  # Default to False for now
        except ImportError:
            return False
    
    def encode(self, text: str, context: str = "") -> Optional[np.ndarray]:
        """Encode text using neural model."""
        if not self.is_available:
            return None
        
        # Placeholder for actual neural encoding
        # This would use transformers library to encode code
        return None
    
    def encode_batch(self, texts: List[str], contexts: Optional[List[str]] = None) -> Optional[np.ndarray]:
        """Encode batch of texts using neural model."""
        if not self.is_available:
            return None
        
        if contexts is None:
            contexts = [""] * len(texts)
        
        # Placeholder for batch encoding
        return None

def encode_ast(ast: Any, vector_dim: int = VECTOR_DIM, 
               config: Optional[HybridEmbeddingConfig] = None) -> np.ndarray:
    """
    Encode an AST into a fixed-dimensional vector using hybrid approach.
    
    Args:
        ast: Abstract Syntax Tree (tuple/list structure or ASTNode list)
        vector_dim: Dimension of the output vector
        config: Configuration for hybrid embedding
        
    Returns:
        Fixed-dimensional vector representation
    """
    if config is None:
        config = HybridEmbeddingConfig(vector_dim=vector_dim)
    
    # Initialize vector components
    hash_vector = np.zeros(vector_dim)
    structural_vector = np.zeros(vector_dim)
    semantic_vector = np.zeros(vector_dim)
    neural_vector = np.zeros(vector_dim)
    
    # 1. Hash-based encoding
    if config.hash_weight > 0:
        hash_vector = _hash_encode_ast(ast, vector_dim)
        # Validate hash vector
        if not np.all(np.isfinite(hash_vector)):
            hash_vector = np.zeros(vector_dim)
    
    # 2. Structural features
    if config.structural_weight > 0:
        structural_vector = _extract_structural_features(ast, vector_dim)
        # Validate structural vector
        if not np.all(np.isfinite(structural_vector)):
            structural_vector = np.zeros(vector_dim)
    
    # 3. Semantic features
    if config.semantic_weight > 0:
        semantic_vector = _extract_semantic_features(ast, vector_dim)
        # Validate semantic vector
        if not np.all(np.isfinite(semantic_vector)):
            semantic_vector = np.zeros(vector_dim)
    
    # 4. Neural embeddings (if available)
    if config.use_neural_embeddings and config.semantic_weight > 0:
        neural_provider = NeuralEmbeddingProvider(config.neural_model_name)
        ast_text = _ast_to_string(ast)
        neural_emb = neural_provider.encode(ast_text)
        if neural_emb is not None:
            # Resize neural embedding to match target dimension
            if len(neural_emb) != vector_dim:
                neural_emb = _resize_vector(neural_emb, vector_dim)
            # Validate neural vector
            if np.all(np.isfinite(neural_emb)):
                neural_vector = neural_emb
    
    # Combine vectors with weights
    final_vector = (
        config.hash_weight * hash_vector +
        config.structural_weight * structural_vector +
        config.semantic_weight * (semantic_vector + neural_vector)
    )
    
    # Normalize with proper handling of zero vectors
    norm = np.linalg.norm(final_vector)
    if norm > 1e-10:  # Use small threshold instead of 0
        final_vector = final_vector / norm
    else:
        # If vector is essentially zero, create a small random vector
        # This prevents NaN values while maintaining some diversity
        np.random.seed(hash(str(ast)) % 2**32)  # Deterministic seed based on AST
        final_vector = np.random.uniform(0.01, 0.1, vector_dim)
        final_vector = final_vector / np.linalg.norm(final_vector)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(final_vector)):
        # Fallback: create a simple hash-based vector
        ast_str = _ast_to_string(ast)
        hash_value = hashlib.md5(ast_str.encode()).hexdigest()
        final_vector = np.zeros(vector_dim)
        for i, char in enumerate(hash_value):
            if i >= vector_dim:
                break
            final_vector[i] = int(char, 16) / 15.0
        # Normalize the fallback vector
        norm = np.linalg.norm(final_vector)
        if norm > 0:
            final_vector = final_vector / norm
    
    return final_vector

def _hash_encode_ast(ast: Any, vector_dim: int) -> np.ndarray:
    """Hash-based encoding of AST."""
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
    
    return vector

def _extract_structural_features(ast: Any, vector_dim: int) -> np.ndarray:
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
    
    # Add complexity metrics
    complexity = _calculate_complexity(ast)
    features.append(min(complexity / 10.0, 1.0))  # Normalize complexity
    
    # Pad or truncate to target dimension
    if len(features) < vector_dim:
        features.extend([0.0] * (vector_dim - len(features)))
    else:
        features = features[:vector_dim]
    
    return np.array(features)

def _extract_semantic_features(ast: Any, vector_dim: int) -> np.ndarray:
    """Extract semantic features from AST."""
    features = []
    
    # Type distribution
    type_distribution = _get_type_distribution(ast)
    for type_cat in ['numeric', 'boolean', 'string', 'function', 'unknown']:
        features.append(type_distribution.get(type_cat, 0.0))
    
    # Semantic patterns
    semantic_patterns = _detect_semantic_patterns(ast)
    features.extend(semantic_patterns)
    
    # Control flow complexity
    control_flow = _analyze_control_flow(ast)
    features.extend(control_flow)
    
    # Data flow analysis
    data_flow = _analyze_data_flow(ast)
    features.extend(data_flow)
    
    # Pad or truncate to target dimension
    if len(features) < vector_dim:
        features.extend([0.0] * (vector_dim - len(features)))
    else:
        features = features[:vector_dim]
    
    return np.array(features)

def _get_type_distribution(ast: Any) -> Dict[str, float]:
    """Get distribution of types in AST."""
    type_counts = {'numeric': 0, 'boolean': 0, 'string': 0, 'function': 0, 'unknown': 0}
    total_nodes = 0
    
    def traverse(node):
        nonlocal total_nodes
        if hasattr(node, 'type_info'):
            # Enhanced AST node
            type_cat = node.type_info.category.value
            type_counts[type_cat] = type_counts.get(type_cat, 0) + 1
            total_nodes += 1
        elif isinstance(node, tuple) and len(node) > 0:
            # Simple AST node
            total_nodes += 1
            for child in node[1:]:
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(ast)
    
    # Normalize
    if total_nodes > 0:
        return {k: float(v) / total_nodes for k, v in type_counts.items()}
    return {k: 0.0 for k in type_counts.keys()}

def _detect_semantic_patterns(ast: Any) -> List[float]:
    """Detect semantic patterns in AST."""
    patterns = {
        'recursive': 0.0,
        'iterative': 0.0,
        'conditional': 0.0,
        'mathematical': 0.0,
        'string_ops': 0.0,
        'side_effects': 0.0,
        'pure_function': 0.0
    }
    
    def traverse(node):
        if hasattr(node, 'content'):
            content = node.content.lower()
            if 'fact(' in content or 'fib(' in content:
                patterns['recursive'] += 1.0
            if 'while' in content or 'for' in content:
                patterns['iterative'] += 1.0
            if 'if' in content:
                patterns['conditional'] += 1.0
            if any(op in content for op in ['+', '-', '*', '/', '%']):
                patterns['mathematical'] += 1.0
            if any(op in content for op in ['+', 'join', 'split', 'replace']):
                patterns['string_ops'] += 1.0
            if any(op in content for op in ['print', 'append', 'extend']):
                patterns['side_effects'] += 1.0
            if 'return' in content and patterns['side_effects'] == 0:
                patterns['pure_function'] += 1.0
        elif isinstance(node, tuple) and len(node) > 0:
            for child in node[1:]:
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(ast)
    
    # Normalize patterns with proper handling of zero total
    total = sum(patterns.values())
    if total > 0:
        return [v / total for v in patterns.values()]
    else:
        # Return uniform distribution when no patterns found
        return [1.0 / len(patterns)] * len(patterns)

def _analyze_control_flow(ast: Any) -> List[float]:
    """Analyze control flow characteristics."""
    metrics = {
        'nesting_depth': 0,
        'branching_factor': 0,
        'cyclomatic_complexity': 1,  # Start with 1
        'linear_sequences': 0
    }
    
    def traverse(node, depth=0):
        metrics['nesting_depth'] = max(metrics['nesting_depth'], depth)
        
        if hasattr(node, 'content'):
            content = node.content.lower()
            if 'if' in content or 'while' in content or 'for' in content:
                metrics['cyclomatic_complexity'] += 1
                metrics['branching_factor'] += 1
            elif 'and' in content or 'or' in content:
                metrics['cyclomatic_complexity'] += 1
        elif isinstance(node, tuple) and len(node) > 0:
            children = len(node) - 1
            if children > 0:
                metrics['branching_factor'] += children
            for child in node[1:]:
                traverse(child, depth + 1)
        elif isinstance(node, list):
            for item in node:
                traverse(item, depth + 1)
    
    traverse(ast)
    
    # Normalize metrics
    return [
        min(metrics['nesting_depth'] / 10.0, 1.0),
        min(metrics['branching_factor'] / 10.0, 1.0),
        min(metrics['cyclomatic_complexity'] / 10.0, 1.0),
        min(metrics['linear_sequences'] / 10.0, 1.0)
    ]

def _analyze_data_flow(ast: Any) -> List[float]:
    """Analyze data flow characteristics."""
    metrics = {
        'variable_count': 0,
        'assignment_count': 0,
        'function_calls': 0,
        'data_dependencies': 0
    }
    
    def traverse(node):
        if hasattr(node, 'content'):
            content = node.content.lower()
            if '=' in content:
                metrics['assignment_count'] += 1
            if '(' in content and ')' in content and 'def' not in content:
                metrics['function_calls'] += 1
        elif isinstance(node, tuple) and len(node) > 0:
            for child in node[1:]:
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(ast)
    
    # Normalize metrics with proper handling of zero total
    total = sum(metrics.values())
    if total > 0:
        return [v / total for v in metrics.values()]
    else:
        # Return uniform distribution when no metrics found
        return [1.0 / len(metrics)] * len(metrics)

def _calculate_complexity(ast: Any) -> float:
    """Calculate overall complexity of AST."""
    complexity = 0.0
    
    def traverse(node):
        nonlocal complexity
        if hasattr(node, 'semantic_annotations'):
            complexity += node.semantic_annotations.get('complexity', 1)
        elif isinstance(node, tuple) and len(node) > 0:
            complexity += 1
            for child in node[1:]:
                traverse(child)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(ast)
    return complexity

def _resize_vector(vector: np.ndarray, target_dim: int) -> np.ndarray:
    """Resize vector to target dimension."""
    if len(vector) == target_dim:
        return vector
    elif len(vector) > target_dim:
        return vector[:target_dim]
    else:
        # Pad with zeros
        result = np.zeros(target_dim)
        result[:len(vector)] = vector
        return result

def _ast_to_string(ast: Any) -> str:
    """Convert AST to string representation for hashing."""
    if hasattr(ast, 'content'):
        # Enhanced AST node
        return ast.content
    elif isinstance(ast, tuple):
        return f"({ast[0]}, " + ", ".join(_ast_to_string(child) for child in ast[1:]) + ")"
    elif isinstance(ast, list):
        return f"[{', '.join(_ast_to_string(item) for item in ast)}]"
    else:
        return str(ast)

def _count_node_types(ast: Any) -> Dict[str, int]:
    """Count occurrences of different node types in the AST."""
    counts = {}
    
    def traverse(node):
        if hasattr(node, 'node_type'):
            # Enhanced AST node
            node_type = node.node_type
            counts[node_type] = counts.get(node_type, 0) + 1
        elif isinstance(node, tuple) and len(node) > 0:
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
        if hasattr(node, 'children'):
            # Enhanced AST node
            if node.children:
                return max(traverse(child, depth + 1) for child in node.children)
            return depth
        elif isinstance(node, tuple) and len(node) > 1:
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
        
        if hasattr(node, 'children'):
            # Enhanced AST node
            total_nodes += 1
            children = len(node.children)
            total_children += children
            for child in node.children:
                traverse(child)
        elif isinstance(node, tuple) and len(node) > 1:
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