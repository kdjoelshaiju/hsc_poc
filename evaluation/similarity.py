"""
Similarity Evaluator

This module provides tools for evaluating semantic similarity between
programs and interpolation quality.
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass

from embedding.engine import DSLToVectorEngine


@dataclass
class EvaluationConfig:
    """Configuration for similarity evaluation."""
    use_cosine_similarity: bool = True
    use_euclidean_distance: bool = True
    use_semantic_overlap: bool = True
    normalize_scores: bool = True


class SimilarityEvaluator:
    """
    Evaluator for semantic similarity and interpolation quality.
    
    This class provides comprehensive evaluation metrics for comparing
    programs and assessing interpolation results.
    """
    
    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()
        
    def evaluate_similarity(self, program1: str, program2: str, 
                           engine: Optional[DSLToVectorEngine] = None) -> Dict[str, float]:
        """
        Evaluate similarity between two programs.
        
        Args:
            program1: First program
            program2: Second program
            engine: Embedding engine
            
        Returns:
            Dictionary of similarity metrics
        """
        if engine is None:
            engine = DSLToVectorEngine()
        
        # Encode programs
        result1 = engine.encode_program(program1)
        result2 = engine.encode_program(program2)
        
        # Extract vectors from results
        embedding1 = result1.vector
        embedding2 = result2.vector
        
        metrics = {}
        
        # Cosine similarity
        if self.config.use_cosine_similarity:
            cosine_sim = self._compute_cosine_similarity(embedding1, embedding2)
            metrics['cosine_similarity'] = cosine_sim
        
        # Euclidean distance
        if self.config.use_euclidean_distance:
            euclidean_dist = self._compute_euclidean_distance(embedding1, embedding2)
            metrics['euclidean_distance'] = euclidean_dist
        
        # Semantic overlap
        if self.config.use_semantic_overlap:
            semantic_overlap = self._compute_semantic_overlap(program1, program2)
            metrics['semantic_overlap'] = semantic_overlap
        
        # Overall similarity score
        overall_score = self._compute_overall_similarity(metrics)
        metrics['overall_similarity'] = overall_score
        
        return metrics
    
    def evaluate_interpolation(self, start_program: str, end_program: str,
                              interpolated_programs: List[str],
                              engine: Optional[DSLToVectorEngine] = None) -> Dict[str, float]:
        """
        Evaluate the quality of interpolation between two programs.
        
        Args:
            start_program: Starting program
            end_program: Ending program
            interpolated_programs: List of interpolated programs
            engine: Embedding engine
            
        Returns:
            Dictionary of evaluation metrics
        """
        if engine is None:
            engine = DSLToVectorEngine()
        
        if not interpolated_programs:
            return {'smoothness': 0.0, 'semantic_coherence': 0.0, 'overall_quality': 0.0}
        
        # Encode all programs
        start_result = engine.encode_program(start_program)
        end_result = engine.encode_program(end_program)
        interpolated_results = [engine.encode_program(p) for p in interpolated_programs]
        
        # Extract vectors from results
        start_embedding = start_result.vector
        end_embedding = end_result.vector
        interpolated_embeddings = [result.vector for result in interpolated_results]
        
        metrics = {}
        
        # Smoothness: how smoothly embeddings change along the path
        smoothness = self._compute_interpolation_smoothness(interpolated_embeddings)
        metrics['smoothness'] = smoothness
        
        # Semantic coherence: how well programs maintain meaning
        semantic_coherence = self._compute_semantic_coherence(
            interpolated_programs, start_program, end_program
        )
        metrics['semantic_coherence'] = semantic_coherence
        
        # Path consistency: how well the path follows the manifold
        path_consistency = self._compute_path_consistency(
            start_embedding, end_embedding, interpolated_embeddings
        )
        metrics['path_consistency'] = path_consistency
        
        # Overall quality score
        overall_quality = self._compute_overall_quality(metrics)
        metrics['overall_quality'] = overall_quality
        
        return metrics
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(cosine_sim)
    
    def _compute_euclidean_distance(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute Euclidean distance between embeddings."""
        distance = np.linalg.norm(embedding1 - embedding2)
        return float(distance)
    
    def _compute_semantic_overlap(self, program1: str, program2: str) -> float:
        """Compute semantic overlap between programs."""
        # Extract keywords from both programs
        keywords1 = self._extract_keywords(program1)
        keywords2 = self._extract_keywords(program2)
        
        if not keywords1 and not keywords2:
            return 1.0  # Both empty programs are considered similar
        
        if not keywords1 or not keywords2:
            return 0.0  # One empty, one non-empty
        
        # Compute Jaccard similarity
        intersection = len(set(keywords1) & set(keywords2))
        union = len(set(keywords1) | set(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_keywords(self, program: str) -> List[str]:
        """Extract semantic keywords from a program."""
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
            elif 'for ' in line:
                keywords.append('loop')
            elif 'return ' in line:
                keywords.append('return')
            elif '=' in line and '==' not in line:
                keywords.append('assignment')
            elif '+' in line or '-' in line or '*' in line or '/' in line:
                keywords.append('arithmetic')
            elif '>' in line or '<' in line:
                keywords.append('comparison')
        
        return keywords
    
    def _compute_overall_similarity(self, metrics: Dict[str, float]) -> float:
        """Compute overall similarity score from individual metrics."""
        if not metrics:
            return 0.0
        
        # Normalize and combine metrics
        scores = []
        
        if 'cosine_similarity' in metrics:
            scores.append(metrics['cosine_similarity'])
        
        if 'euclidean_distance' in metrics:
            # Convert distance to similarity (inverse relationship)
            # Assuming typical distance range of 0-10
            distance_sim = max(0, 1 - metrics['euclidean_distance'] / 10)
            scores.append(distance_sim)
        
        if 'semantic_overlap' in metrics:
            scores.append(metrics['semantic_overlap'])
        
        return float(np.mean(scores)) if scores else 0.0
    
    def _compute_interpolation_smoothness(self, embeddings: List[np.ndarray]) -> float:
        """Compute smoothness of interpolation path."""
        if len(embeddings) < 2:
            return 1.0
        
        # Compute distances between consecutive embeddings
        distances = []
        for i in range(1, len(embeddings)):
            dist = np.linalg.norm(embeddings[i] - embeddings[i-1])
            distances.append(dist)
        
        # Smoothness is inversely proportional to variance of distances
        if not distances:
            return 1.0
        
        variance = float(np.var(distances))
        smoothness = 1.0 / (1.0 + variance)
        return float(smoothness)
    
    def _compute_semantic_coherence(self, programs: List[str], 
                                   start_program: str, end_program: str) -> float:
        """Compute semantic coherence of interpolated programs."""
        if not programs:
            return 0.0
        
        # Extract keywords from start and end programs
        start_keywords = set(self._extract_keywords(start_program))
        end_keywords = set(self._extract_keywords(end_program))
        
        coherence_scores = []
        for program in programs:
            program_keywords = set(self._extract_keywords(program))
            
            # Check overlap with start and end keywords
            start_overlap = len(program_keywords & start_keywords) / max(len(start_keywords), 1)
            end_overlap = len(program_keywords & end_keywords) / max(len(end_keywords), 1)
            
            # Average overlap
            coherence = (start_overlap + end_overlap) / 2
            coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores))
    
    def _compute_path_consistency(self, start_embedding: np.ndarray, 
                                 end_embedding: np.ndarray,
                                 interpolated_embeddings: List[np.ndarray]) -> float:
        """Compute how well the interpolation path follows the manifold."""
        if not interpolated_embeddings:
            return 1.0
        
        # Compute total path length
        total_path_length = 0.0
        for i in range(1, len(interpolated_embeddings)):
            dist = np.linalg.norm(interpolated_embeddings[i] - interpolated_embeddings[i-1])
            total_path_length += dist
        
        # Compute direct distance
        direct_distance = np.linalg.norm(end_embedding - start_embedding)
        
        # Path consistency is inversely proportional to path length ratio
        if direct_distance == 0:
            return 1.0
        
        path_ratio = float(total_path_length / direct_distance)
        consistency = 1.0 / max(1.0, path_ratio)
        return float(consistency)
    
    def _compute_overall_quality(self, metrics: Dict[str, float]) -> float:
        """Compute overall quality score from interpolation metrics."""
        if not metrics:
            return 0.0
        
        # Weighted average of metrics
        weights = {
            'smoothness': 0.3,
            'semantic_coherence': 0.4,
            'path_consistency': 0.3
        }
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_sum += weight * metrics[metric]
                total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def compare_interpolation_methods(self, start_program: str, end_program: str,
                                     methods: List[str],
                                     engine: Optional[DSLToVectorEngine] = None) -> Dict[str, Dict[str, float]]:
        """
        Compare different interpolation methods.
        
        Args:
            start_program: Starting program
            end_program: Ending program
            methods: List of interpolation methods to compare
            engine: Embedding engine
            
        Returns:
            Dictionary mapping method names to evaluation metrics
        """
        if engine is None:
            engine = DSLToVectorEngine()
        
        results = {}
        
        for method in methods:
            # This would require the actual interpolation implementation
            # For now, return placeholder results
            results[method] = {
                'smoothness': 0.8,
                'semantic_coherence': 0.7,
                'path_consistency': 0.9,
                'overall_quality': 0.8
            }
        
        return results


if __name__ == "__main__":
    # Test similarity evaluator
    evaluator = SimilarityEvaluator()
    
    # Test programs
    program1 = "def square(x): return x * x"
    program2 = "def cube(x): return x * x * x"
    program3 = "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
    
    # Test similarity evaluation
    similarity = evaluator.evaluate_similarity(program1, program2)
    print(f"Similarity between square and cube: {similarity}")
    
    # Test interpolation evaluation
    interpolated = [
        "def square(x): return x * x",
        "def power2_5(x): return x * x * x**0.5",
        "def cube(x): return x * x * x"
    ]
    
    quality = evaluator.evaluate_interpolation(program1, program2, interpolated)
    print(f"Interpolation quality: {quality}") 