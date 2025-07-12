"""
Manifold Learning Module for HSC

This module provides tools for learning the underlying manifold structure
of program embeddings to enable meaningful interpolation.
"""

from .autoencoder import AutoencoderManifold
from .diffusion import DiffusionMaps
from .reduction import DimensionalityReducer

__all__ = ['AutoencoderManifold', 'DiffusionMaps', 'DimensionalityReducer'] 