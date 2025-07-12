"""
Embedding Module for HSC

This module provides tools for converting DSL programs into vector representations
that preserve semantic relationships.
"""

from .engine import DSLToVectorEngine, EmbeddingConfig
from .encoder import encode_ast, VECTOR_DIM
from .decoder import decode_vec_to_ast

__all__ = [
    'DSLToVectorEngine',
    'EmbeddingConfig', 
    'encode_ast',
    'VECTOR_DIM',
    'decode_vec_to_ast'
] 