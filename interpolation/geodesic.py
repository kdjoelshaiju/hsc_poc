"""
Geodesic Interpolator

This module implements geodesic interpolation between programs
in the learned semantic manifold.
"""

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
import json

from embedding.engine import DSLToVectorEngine


@dataclass
class InterpolationConfig:
    """Configuration for geodesic interpolation."""
    method: str = "linear"  # "linear", "spline", "geodesic"
    steps: int = 10
    validate_semantics: bool = True
    smoothness_weight: float = 0.1
    semantic_weight: float = 1.0


class GeodesicInterpolator:
    """
    Geodesic interpolator for program synthesis.
    
    This class computes optimal paths between programs in the learned
    manifold and generates intermediate programs along these paths.
    """
    
    def __init__(self, manifold, config: Optional[InterpolationConfig] = None):
        self.manifold = manifold
        self.config = config or InterpolationConfig()
        
    def interpolate(self, start_program: str, end_program: str, 
                   steps: Optional[int] = None, 
                   engine: Optional[DSLToVectorEngine] = None) -> List[str]:
        """
        Interpolate between two programs.
        
        Args:
            start_program: Starting program code
            end_program: Ending program code
            steps: Number of interpolation steps (overrides config)
            engine: Embedding engine for encoding/decoding
            
        Returns:
            List of interpolated program codes
        """
        if steps is None:
            steps = self.config.steps
        
        print(f"Interpolating between programs in {steps} steps...")
        
        # Encode programs
        if engine is None:
            engine = DSLToVectorEngine()
        
        start_embedding = engine.encode_program(start_program)
        end_embedding = engine.encode_program(end_program)
        
        # Interpolate in embedding space
        interpolated_embeddings = self._interpolate_embeddings(
            start_embedding, end_embedding, steps
        )
        
        # Decode back to programs
        interpolated_programs = []
        for embedding in interpolated_embeddings:
            program = self._decode_embedding(embedding, engine)
            interpolated_programs.append(program)
        
        return interpolated_programs
    
    def _interpolate_embeddings(self, start_embedding: np.ndarray, 
                               end_embedding: np.ndarray, steps: int) -> List[np.ndarray]:
        """Interpolate between two embeddings."""
        if self.config.method == "linear":
            return self._linear_interpolation(start_embedding, end_embedding, steps)
        elif self.config.method == "spline":
            return self._spline_interpolation(start_embedding, end_embedding, steps)
        elif self.config.method == "geodesic":
            return self._geodesic_interpolation(start_embedding, end_embedding, steps)
        else:
            raise ValueError(f"Unknown interpolation method: {self.config.method}")
    
    def _linear_interpolation(self, start_embedding: np.ndarray, 
                             end_embedding: np.ndarray, steps: int) -> List[np.ndarray]:
        """Linear interpolation between embeddings."""
        alphas = np.linspace(0, 1, steps)
        interpolated = []
        
        for alpha in alphas:
            point = (1 - alpha) * start_embedding + alpha * end_embedding
            interpolated.append(point)
        
        return interpolated
    
    def _spline_interpolation(self, start_embedding: np.ndarray, 
                             end_embedding: np.ndarray, steps: int) -> List[np.ndarray]:
        """Spline interpolation between embeddings (simplified)."""
        # For now, use linear interpolation with some smoothing
        # In a full implementation, this would use cubic splines
        
        linear_points = self._linear_interpolation(start_embedding, end_embedding, steps)
        
        # Apply simple smoothing
        smoothed = []
        for i, point in enumerate(linear_points):
            if i == 0 or i == len(linear_points) - 1:
                smoothed.append(point)
            else:
                # Simple moving average smoothing
                prev_point = linear_points[i - 1]
                next_point = linear_points[i + 1]
                smoothed_point = 0.25 * prev_point + 0.5 * point + 0.25 * next_point
                smoothed.append(smoothed_point)
        
        return smoothed
    
    def _geodesic_interpolation(self, start_embedding: np.ndarray, 
                               end_embedding: np.ndarray, steps: int) -> List[np.ndarray]:
        """Geodesic interpolation using the learned manifold."""
        if not hasattr(self.manifold, 'encode') or not hasattr(self.manifold, 'decode'):
            # Fallback to linear interpolation if manifold doesn't support encoding/decoding
            return self._linear_interpolation(start_embedding, end_embedding, steps)
        
        # Encode to latent space
        start_latent = self.manifold.encode(start_embedding.reshape(1, -1))[0]
        end_latent = self.manifold.encode(end_embedding.reshape(1, -1))[0]
        
        # Interpolate in latent space
        alphas = np.linspace(0, 1, steps)
        interpolated = []
        
        for alpha in alphas:
            latent_point = (1 - alpha) * start_latent + alpha * end_latent
            # Decode back to embedding space
            embedding_point = self.manifold.decode(latent_point.reshape(1, -1))[0]
            interpolated.append(embedding_point)
        
        return interpolated
    
    def _decode_embedding(self, embedding: np.ndarray, engine: DSLToVectorEngine) -> str:
        """Decode an embedding back to program code."""
        # For now, return a placeholder program
        # In a full implementation, this would use the decoder to generate actual code
        
        # Simple heuristic: if embedding is close to start, return start program
        # This is a very simplified approach
        return f"# Interpolated program (embedding norm: {np.linalg.norm(embedding):.3f})\ndef interpolated_function():\n    return 'interpolated_result'"
    
    def find_optimal_path(self, start_program: str, end_program: str, 
                         engine: Optional[DSLToVectorEngine] = None) -> Dict[str, Any]:
        """
        Find the optimal path between two programs.
        
        Args:
            start_program: Starting program
            end_program: Ending program
            engine: Embedding engine
            
        Returns:
            Dictionary containing path information
        """
        if engine is None:
            engine = DSLToVectorEngine()
        
        start_embedding = engine.encode_program(start_program)
        end_embedding = engine.encode_program(end_program)
        
        # Compute path metrics
        euclidean_distance = np.linalg.norm(end_embedding - start_embedding)
        
        # If manifold supports geodesic distance, compute it
        if hasattr(self.manifold, 'get_manifold_curvature'):
            curvature = self.manifold.get_manifold_curvature(
                np.vstack([start_embedding, end_embedding])
            )
        else:
            curvature = 0.0
        
        return {
            'start_program': start_program,
            'end_program': end_program,
            'euclidean_distance': euclidean_distance,
            'manifold_curvature': curvature,
            'interpolation_method': self.config.method,
            'steps': self.config.steps
        }
    
    def validate_interpolation(self, interpolated_programs: List[str], 
                              start_program: str, end_program: str) -> Dict[str, float]:
        """
        Validate the quality of interpolation.
        
        Args:
            interpolated_programs: List of interpolated programs
            start_program: Starting program
            end_program: Ending program
            
        Returns:
            Dictionary of validation metrics
        """
        if not interpolated_programs:
            return {'smoothness': 0.0, 'semantic_coherence': 0.0, 'overall_quality': 0.0}
        
        # Compute smoothness (how smoothly programs change)
        smoothness = self._compute_smoothness(interpolated_programs)
        
        # Compute semantic coherence (how well programs maintain meaning)
        semantic_coherence = self._compute_semantic_coherence(
            interpolated_programs, start_program, end_program
        )
        
        # Overall quality score
        overall_quality = (
            self.config.smoothness_weight * smoothness +
            self.config.semantic_weight * semantic_coherence
        ) / (self.config.smoothness_weight + self.config.semantic_weight)
        
        return {
            'smoothness': smoothness,
            'semantic_coherence': semantic_coherence,
            'overall_quality': overall_quality
        }
    
    def _compute_smoothness(self, programs: List[str]) -> float:
        """Compute smoothness of interpolation."""
        if len(programs) < 2:
            return 1.0
        
        # Simple heuristic: count how many programs are different from neighbors
        changes = 0
        for i in range(1, len(programs)):
            if programs[i] != programs[i-1]:
                changes += 1
        
        # Normalize by number of transitions
        smoothness = 1.0 - (changes / (len(programs) - 1))
        return smoothness
    
    def _compute_semantic_coherence(self, programs: List[str], 
                                   start_program: str, end_program: str) -> float:
        """Compute semantic coherence of interpolated programs."""
        # Simple heuristic: check if programs contain key elements from start/end
        start_keywords = self._extract_keywords(start_program)
        end_keywords = self._extract_keywords(end_program)
        
        coherence_scores = []
        for program in programs:
            program_keywords = self._extract_keywords(program)
            
            # Check overlap with start and end keywords
            start_overlap = len(set(program_keywords) & set(start_keywords)) / max(len(start_keywords), 1)
            end_overlap = len(set(program_keywords) & set(end_keywords)) / max(len(end_keywords), 1)
            
            # Average overlap
            coherence = (start_overlap + end_overlap) / 2
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores)
    
    def _extract_keywords(self, program: str) -> List[str]:
        """Extract keywords from a program."""
        # Simple keyword extraction
        keywords = []
        lines = program.split('\n')
        
        for line in lines:
            line = line.strip()
            if line.startswith('def '):
                keywords.append('function')
            elif 'if ' in line:
                keywords.append('conditional')
            elif 'while ' in line:
                keywords.append('loop')
            elif 'return ' in line:
                keywords.append('return')
            elif '=' in line:
                keywords.append('assignment')
        
        return keywords


if __name__ == "__main__":
    # Test geodesic interpolator
    from manifold.autoencoder import AutoencoderManifold
    
    # Create a simple manifold
    manifold = AutoencoderManifold(input_dim=64, latent_dim=16)
    
    # Create interpolator
    interpolator = GeodesicInterpolator(manifold)
    
    # Test programs
    start_program = "def square(x): return x * x"
    end_program = "def cube(x): return x * x * x"
    
    # Test interpolation
    interpolated = interpolator.interpolate(start_program, end_program, steps=5)
    
    print(f"Interpolated {len(interpolated)} programs:")
    for i, program in enumerate(interpolated):
        print(f"  Step {i}: {program[:50]}...")
    
    # Test validation
    metrics = interpolator.validate_interpolation(interpolated, start_program, end_program)
    print(f"\nValidation metrics: {metrics}") 