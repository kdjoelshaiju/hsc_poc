"""
Diffusion Maps for Manifold Learning

This module implements diffusion maps for discovering the underlying
manifold structure of program embeddings.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass
import json


@dataclass
class DiffusionConfig:
    """Configuration for diffusion maps."""
    epsilon: float = 1.0
    n_eigenvectors: int = 10
    alpha: float = 1.0
    normalize: bool = True


class DiffusionMaps:
    """
    Diffusion maps for manifold learning of program embeddings.
    
    This class implements the diffusion maps algorithm to discover
    the intrinsic geometry of the program embedding space.
    """
    
    def __init__(self, config: Optional[DiffusionConfig] = None):
        self.config = config or DiffusionConfig()
        self.is_fitted = False
        
    def fit(self, embeddings: np.ndarray):
        """
        Fit diffusion maps to the embedding data.
        
        Args:
            embeddings: Input embeddings, shape (n_samples, n_features)
        """
        print(f"Fitting diffusion maps on {len(embeddings)} samples")
        
        # Compute pairwise distances
        distances = self._compute_distances(embeddings)
        
        # Compute diffusion kernel
        kernel = self._compute_kernel(distances)
        
        # Compute diffusion operator
        self._compute_diffusion_operator(kernel)
        
        self.is_fitted = True
        print("Diffusion maps fitting completed")
    
    def _compute_distances(self, embeddings: np.ndarray) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        n_samples = len(embeddings)
        distances = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(embeddings[i] - embeddings[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def _compute_kernel(self, distances: np.ndarray) -> np.ndarray:
        """Compute the diffusion kernel matrix."""
        # Gaussian kernel
        kernel = np.exp(-distances**2 / (2 * self.config.epsilon**2))
        
        # Apply alpha normalization
        if self.config.alpha > 0:
            q = np.sum(kernel, axis=1)
            kernel = kernel / (q[:, np.newaxis] * q[np.newaxis, :])**self.config.alpha
        
        return kernel
    
    def _compute_diffusion_operator(self, kernel: np.ndarray):
        """Compute the diffusion operator and its eigendecomposition."""
        # Compute degree matrix
        degree = np.sum(kernel, axis=1)
        
        # Normalize kernel
        if self.config.normalize:
            degree_inv_sqrt = 1.0 / np.sqrt(degree)
            kernel_normalized = kernel * degree_inv_sqrt[:, np.newaxis] * degree_inv_sqrt[np.newaxis, :]
        else:
            kernel_normalized = kernel
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(kernel_normalized)
        
        # Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[sorted_indices]
        self.eigenvectors_ = eigenvectors[:, sorted_indices]
        
        # Store degree for later use
        self.degree_ = degree
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to diffusion coordinates.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Diffusion coordinates
        """
        if not self.is_fitted:
            raise ValueError("Diffusion maps must be fitted before transformation")
        
        # For simplicity, return the top eigenvectors
        # In a full implementation, this would compute the Nystrom extension
        return self.eigenvectors_[:, :self.config.n_eigenvectors]
    
    def get_diffusion_distance(self, i: int, j: int, t: float = 1.0) -> float:
        """
        Compute diffusion distance between two points.
        
        Args:
            i, j: Indices of the two points
            t: Diffusion time parameter
            
        Returns:
            Diffusion distance
        """
        if not self.is_fitted:
            raise ValueError("Diffusion maps must be fitted before computing distances")
        
        # Compute diffusion distance using eigenvalues and eigenvectors
        diff = self.eigenvectors_[i, 1:] - self.eigenvectors_[j, 1:]  # Skip first eigenvector
        weights = self.eigenvalues_[1:] ** (2 * t)
        
        distance = np.sqrt(np.sum(weights * diff**2))
        return distance
    
    def get_manifold_dimension(self) -> int:
        """
        Estimate the intrinsic dimension of the manifold.
        
        Returns:
            Estimated manifold dimension
        """
        if not self.is_fitted:
            raise ValueError("Diffusion maps must be fitted before estimating dimension")
        
        # Simple heuristic: count eigenvalues above a threshold
        threshold = 0.1 * np.max(self.eigenvalues_)
        dimension = np.sum(self.eigenvalues_ > threshold)
        
        return dimension
    
    def save_model(self, filepath: str):
        """Save the fitted diffusion maps model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        data = {
            'config': self.config.__dict__,
            'eigenvalues': self.eigenvalues_.tolist(),
            'eigenvectors': self.eigenvectors_.tolist(),
            'degree': self.degree_.tolist(),
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_model(self, filepath: str):
        """Load a fitted diffusion maps model."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = DiffusionConfig(**data['config'])
        self.eigenvalues_ = np.array(data['eigenvalues'])
        self.eigenvectors_ = np.array(data['eigenvectors'])
        self.degree_ = np.array(data['degree'])
        self.is_fitted = data['is_fitted']


if __name__ == "__main__":
    # Test diffusion maps
    np.random.seed(42)
    
    # Generate synthetic embeddings on a curved manifold
    n_samples = 100
    n_features = 64
    
    # Create a 2D Swiss roll-like structure in high dimensions
    t = np.linspace(0, 4*np.pi, n_samples)
    x = t * np.cos(t)
    y = t * np.sin(t)
    
    # Add noise and embed in high dimensions
    embeddings = np.random.randn(n_samples, n_features)
    embeddings[:, 0] = x + 0.1 * np.random.randn(n_samples)
    embeddings[:, 1] = y + 0.1 * np.random.randn(n_samples)
    
    # Fit diffusion maps
    diffusion = DiffusionMaps()
    diffusion.fit(embeddings)
    
    # Transform to diffusion coordinates
    coords = diffusion.transform(embeddings)
    
    print(f"Original shape: {embeddings.shape}")
    print(f"Diffusion coordinates shape: {coords.shape}")
    print(f"Estimated manifold dimension: {diffusion.get_manifold_dimension()}")
    
    # Test diffusion distance
    dist = diffusion.get_diffusion_distance(0, 10)
    print(f"Diffusion distance between points 0 and 10: {dist:.4f}") 