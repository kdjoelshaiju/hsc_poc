"""
DSL Module for HSC

This module provides the domain-specific language parser and examples
for the Holistic Semantic Continuum system.
"""

from .examples import examples
from .parser import parse_program

__all__ = ['examples', 'parse_program'] 