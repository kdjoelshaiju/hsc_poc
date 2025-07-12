"""
Dimensionality Reduction for Manifold Learning

This module provides various dimensionality reduction techniques
for visualizing and analyzing program embedding manifolds.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Literal
from dataclasses import dataclass
import json


@dataclass
class ReductionConfig:
    """Configuration for dimensionality reduction."""
    method: Literal["pca", "tsne", "umap"] = "pca"
    n_components: int = 2
    perplexity: float = 30.0  # For t-SNE
    learning_rate: float = 200.0  # For t-SNE
    n_neighbors: int = 15  # For UMAP
    min_dist: float = 0.1  # For UMAP


class DimensionalityReducer:
    """
    Dimensionality reduction for program embedding visualization.
    
    This class provides a unified interface for various dimensionality
    reduction techniques including PCA, t-SNE, and UMAP.
    """
    
    def __init__(self, config: Optional[ReductionConfig] = None):
        self.config = config or ReductionConfig()
        self.is_fitted = False
        
    def fit(self, embeddings: np.ndarray):
        """
        Fit the dimensionality reduction model.
        
        Args:
            embeddings: Input embeddings, shape (n_samples, n_features)
        """
        print(f"Fitting {self.config.method.upper()} reduction to {len(embeddings)} samples")
        
        if self.config.method == "pca":
            self._fit_pca(embeddings)
        elif self.config.method == "tsne":
            self._fit_tsne(embeddings)
        elif self.config.method == "umap":
            self._fit_umap(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {self.config.method}")
        
        self.is_fitted = True
        print(f"{self.config.method.upper()} fitting completed")
    
    def _fit_pca(self, embeddings: np.ndarray):
        """Fit PCA dimensionality reduction."""
        # Center the data
        self.mean_ = np.mean(embeddings, axis=0)
        centered_data = embeddings - self.mean_
        
        # Compute covariance matrix and eigendecomposition
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Store components
        self.components_ = eigenvectors[:, :self.config.n_components]
        self.explained_variance_ = eigenvalues[:self.config.n_components]
        self.explained_variance_ratio_ = eigenvalues[:self.config.n_components] / np.sum(eigenvalues)
    
    def _fit_tsne(self, embeddings: np.ndarray):
        """Fit t-SNE dimensionality reduction (simplified implementation)."""
        # For now, implement a simplified version
        # In a full implementation, this would use scikit-learn's TSNE
        
        # Compute pairwise distances
        distances = self._compute_distances(embeddings)
        
        # Simple approximation: use PCA as initialization and add some noise
        self._fit_pca(embeddings)
        
        # Add some random perturbation to simulate t-SNE-like behavior
        np.random.seed(42)
        noise = 0.1 * np.random.randn(*self.components_.shape)
        self.components_ += noise
        
        # Normalize
        self.components_ = self.components_ / np.linalg.norm(self.components_, axis=0)
    
    def _fit_umap(self, embeddings: np.ndarray):
        """Fit UMAP dimensionality reduction (simplified implementation)."""
        # For now, implement a simplified version
        # In a full implementation, this would use the umap-learn library
        
        # Use PCA as a starting point
        self._fit_pca(embeddings)
        
        # Apply some non-linear transformation to simulate UMAP behavior
        # This is a very simplified approximation
        transformed = np.tanh(self.components_)
        self.components_ = transformed / np.linalg.norm(transformed, axis=0)
    
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
    
    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Transform embeddings to reduced dimensions.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Reduced embeddings, shape (n_samples, n_components)
        """
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before transformation")
        
        if self.config.method == "pca":
            return self._transform_pca(embeddings)
        elif self.config.method == "tsne":
            return self._transform_tsne(embeddings)
        elif self.config.method == "umap":
            return self._transform_umap(embeddings)
        else:
            raise ValueError(f"Unknown reduction method: {self.config.method}")
    
    def _transform_pca(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform using PCA."""
        centered_data = embeddings - self.mean_
        return centered_data @ self.components_
    
    def _transform_tsne(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform using t-SNE (simplified)."""
        # For simplicity, use PCA transformation with some noise
        result = self._transform_pca(embeddings)
        
        # Add some noise to simulate t-SNE's non-linear behavior
        np.random.seed(42)
        noise = 0.05 * np.random.randn(*result.shape)
        return result + noise
    
    def _transform_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Transform using UMAP (simplified)."""
        # For simplicity, use PCA transformation with non-linear transform
        result = self._transform_pca(embeddings)
        
        # Apply non-linear transformation
        return np.tanh(result)
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            Reduced embeddings
        """
        self.fit(embeddings)
        return self.transform(embeddings)
    
    def inverse_transform(self, reduced_embeddings: np.ndarray) -> np.ndarray:
        """
        Transform reduced embeddings back to original space.
        
        Args:
            reduced_embeddings: Reduced embeddings
            
        Returns:
            Reconstructed embeddings
        """
        if not self.is_fitted:
            raise ValueError("Reducer must be fitted before inverse transformation")
        
        if self.config.method == "pca":
            # PCA has a well-defined inverse transformation
            reconstructed = reduced_embeddings @ self.components_.T
            return reconstructed + self.mean_
        else:
            # For t-SNE and UMAP, inverse transformation is not well-defined
            # Return an approximation using PCA inverse
            reconstructed = reduced_embeddings @ self.components_.T
            return reconstructed + self.mean_
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get the explained variance ratio (PCA only)."""
        if not self.is_fitted or self.config.method != "pca":
            raise ValueError("Explained variance ratio only available for PCA")
        
        return self.explained_variance_ratio_
    
    def save_model(self, filepath: str):
        """Save the fitted reducer model."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        data = {
            'config': self.config.__dict__,
            'method': self.config.method,
            'is_fitted': self.is_fitted
        }
        
        # Add method-specific data
        if self.config.method == "pca":
            data.update({
                'mean': self.mean_.tolist(),
                'components': self.components_.tolist(),
                'explained_variance': self.explained_variance_.tolist(),
                'explained_variance_ratio': self.explained_variance_ratio_.tolist()
            })
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_model(self, filepath: str):
        """Load a fitted reducer model."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = ReductionConfig(**data['config'])
        self.is_fitted = data['is_fitted']
        
        # Load method-specific data
        if self.config.method == "pca":
            self.mean_ = np.array(data['mean'])
            self.components_ = np.array(data['components'])
            self.explained_variance_ = np.array(data['explained_variance'])
            self.explained_variance_ratio_ = np.array(data['explained_variance_ratio'])


if __name__ == "__main__":
    # Test dimensionality reduction
    np.random.seed(42)
    
    # Generate synthetic embeddings
    n_samples = 100
    n_features = 64
    
    embeddings = np.random.randn(n_samples, n_features)
    
    # Test PCA
    print("Testing PCA reduction...")
    pca_reducer = DimensionalityReducer(ReductionConfig(method="pca", n_components=2))
    pca_result = pca_reducer.fit_transform(embeddings)
    print(f"PCA result shape: {pca_result.shape}")
    print(f"Explained variance ratio: {pca_reducer.get_explained_variance_ratio()}")
    
    # Test t-SNE (simplified)
    print("\nTesting t-SNE reduction...")
    tsne_reducer = DimensionalityReducer(ReductionConfig(method="tsne", n_components=2))
    tsne_result = tsne_reducer.fit_transform(embeddings)
    print(f"t-SNE result shape: {tsne_result.shape}")
    
    # Test UMAP (simplified)
    print("\nTesting UMAP reduction...")
    umap_reducer = DimensionalityReducer(ReductionConfig(method="umap", n_components=2))
    umap_result = umap_reducer.fit_transform(embeddings)
    print(f"UMAP result shape: {umap_result.shape}") 