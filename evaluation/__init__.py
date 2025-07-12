"""
Evaluation Module for HSC

This module provides tools for evaluating semantic similarity and
interpolation quality in the HSC system.
"""

from .similarity import SimilarityEvaluator
from .metrics import SemanticMetrics
from .benchmarks import BenchmarkSuite

__all__ = ['SimilarityEvaluator', 'SemanticMetrics', 'BenchmarkSuite'] 