"""
Autoencoder Manifold Learning with Quality Metrics

This module implements neural autoencoders for learning the underlying
manifold structure of program embeddings with quality assessment metrics.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances

@dataclass
class ManifoldQualityMetrics:
    """Quality metrics for manifold learning."""
    trustworthiness: float
    continuity: float
    neighborhood_preservation: float
    stress: float
    reconstruction_error: float
    curvature_estimate: float
    local_density_ratio: float
    global_structure_preservation: float

@dataclass
class AutoencoderConfig:
    """Configuration for the autoencoder manifold."""
    input_dim: int
    latent_dim: int
    hidden_dims: Tuple[int, ...] = (128, 64)
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    dropout_rate: float = 0.2
    use_batch_norm: bool = True
    quality_metrics_k: int = 15  # k for k-nearest neighbors in quality metrics

class AutoencoderManifold:
    """
    Neural autoencoder for learning program embedding manifolds with quality assessment.
    
    This class implements a simple autoencoder architecture that can be
    extended with more sophisticated neural network implementations.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, config: Optional[AutoencoderConfig] = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.config = config or AutoencoderConfig(input_dim=input_dim, latent_dim=latent_dim)
        
        # Placeholder for neural network components
        self.encoder = None
        self.decoder = None
        self.is_fitted = False
        
        # Quality metrics storage
        self.quality_metrics: Optional[ManifoldQualityMetrics] = None
        self.embedding_history: List[np.ndarray] = []
        self.latent_history: List[np.ndarray] = []
        
    def fit(self, embeddings: np.ndarray, validation_data: Optional[np.ndarray] = None):
        """
        Fit the autoencoder to the embedding data.
        
        Args:
            embeddings: Training embeddings, shape (n_samples, input_dim)
            validation_data: Optional validation embeddings
        """
        print(f"Fitting autoencoder: {self.input_dim} -> {self.latent_dim}")
        print(f"Training on {len(embeddings)} samples")
        
        # Store original embeddings for quality assessment
        self.original_embeddings = embeddings.copy()
        
        # For now, implement a simple PCA-based approximation
        # In a full implementation, this would use PyTorch/TensorFlow
        self._fit_pca_approximation(embeddings)
        
        # Set fitted flag before computing quality metrics
        self.is_fitted = True
        
        # Compute quality metrics
        self._compute_quality_metrics(embeddings)
        
        print("Autoencoder fitting completed")
        if self.quality_metrics:
            print(f"Quality metrics - Trustworthiness: {self.quality_metrics.trustworthiness:.3f}, "
                  f"Continuity: {self.quality_metrics.continuity:.3f}")
    
    def _fit_pca_approximation(self, embeddings: np.ndarray):
        """Fit a PCA-based approximation of the autoencoder."""
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
        
        # Store the top components for encoding/decoding
        self.components_ = eigenvectors[:, :self.latent_dim]
        self.explained_variance_ = eigenvalues[:self.latent_dim]
        
        # Compute reconstruction error directly
        encoded = centered_data @ self.components_
        decoded = encoded @ self.components_.T + self.mean_
        self.reconstruction_error_ = np.mean(np.square(embeddings - decoded))
        
        print(f"PCA approximation fitted with reconstruction error: {self.reconstruction_error_:.6f}")
    
    def _compute_quality_metrics(self, embeddings: np.ndarray):
        """Compute manifold quality metrics."""
        # Encode to latent space
        latent_embeddings = self.encode(embeddings)
        
        # Adjust k for quality metrics based on sample size
        n_samples = len(embeddings)
        k = min(self.config.quality_metrics_k, n_samples - 1)
        if k < 2:
            # Not enough samples for quality metrics
            self.quality_metrics = ManifoldQualityMetrics(
                trustworthiness=1.0,
                continuity=1.0,
                neighborhood_preservation=1.0,
                stress=0.0,
                reconstruction_error=float(self.reconstruction_error_),
                curvature_estimate=0.0,
                local_density_ratio=1.0,
                global_structure_preservation=1.0
            )
            return
        
        # Compute pairwise distances in original and latent space
        original_distances = pairwise_distances(embeddings, metric='euclidean')
        latent_distances = pairwise_distances(latent_embeddings, metric='euclidean')
        
        # Trustworthiness: how well local structure is preserved
        trustworthiness = self._compute_trustworthiness(original_distances, latent_distances)
        
        # Continuity: how well global structure is preserved
        continuity = self._compute_continuity(original_distances, latent_distances)
        
        # Neighborhood preservation
        neighborhood_preservation = self._compute_neighborhood_preservation(
            original_distances, latent_distances
        )
        
        # Stress (Sammon's stress)
        stress = self._compute_stress(original_distances, latent_distances)
        
        # Curvature estimate
        curvature_estimate = self._estimate_curvature(embeddings, latent_embeddings)
        
        # Local density ratio
        local_density_ratio = self._compute_local_density_ratio(embeddings, latent_embeddings)
        
        # Global structure preservation
        global_structure_preservation = self._compute_global_structure_preservation(
            original_distances, latent_distances
        )
        
        self.quality_metrics = ManifoldQualityMetrics(
            trustworthiness=float(trustworthiness),
            continuity=float(continuity),
            neighborhood_preservation=float(neighborhood_preservation),
            stress=float(stress),
            reconstruction_error=float(self.reconstruction_error_),
            curvature_estimate=float(curvature_estimate),
            local_density_ratio=float(local_density_ratio),
            global_structure_preservation=float(global_structure_preservation)
        )
    
    def _compute_trustworthiness(self, original_distances: np.ndarray, 
                                latent_distances: np.ndarray) -> float:
        """Compute trustworthiness metric (Venna & Kaski, 2001)."""
        n_samples = original_distances.shape[0]
        k = min(self.config.quality_metrics_k, n_samples - 2)  # Ensure k+1 <= n_samples
        
        if k < 1:
            return 1.0
        
        # Find k-nearest neighbors in latent space
        nn_latent = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
        nn_latent.fit(latent_distances)
        _, indices_latent = nn_latent.kneighbors()
        
        # Remove self from neighbors
        indices_latent = indices_latent[:, 1:]
        
        trustworthiness = 0.0
        for i in range(n_samples):
            # Find points that are neighbors in latent space but not in original
            latent_neighbors = set(indices_latent[i])
            
            # Find k-nearest neighbors in original space
            nn_original = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
            nn_original.fit(original_distances)
            _, indices_original = nn_original.kneighbors()
            original_neighbors = set(indices_original[i, 1:])
            
            # Count intruders (points that are neighbors in latent but not original)
            intruders = latent_neighbors - original_neighbors
            
            # Compute penalty
            penalty = 0.0
            for intruder in intruders:
                # Find rank of intruder in original space
                original_ranks = np.argsort(original_distances[i])
                intruder_rank = np.where(original_ranks == intruder)[0][0]
                if n_samples - k > 0:  # Avoid division by zero
                    penalty += (intruder_rank - k) / (n_samples - k)
            
            denominator = k * (2 * n_samples - 3 * k - 1)
            if denominator > 0:  # Avoid division by zero
                trustworthiness += 1.0 - (2.0 / denominator) * penalty
            else:
                trustworthiness += 1.0
        
        return float(trustworthiness / n_samples)
    
    def _compute_continuity(self, original_distances: np.ndarray, 
                           latent_distances: np.ndarray) -> float:
        """Compute continuity metric (Venna & Kaski, 2001)."""
        n_samples = original_distances.shape[0]
        k = min(self.config.quality_metrics_k, n_samples - 2)  # Ensure k+1 <= n_samples
        
        if k < 1:
            return 1.0
        
        # Find k-nearest neighbors in original space
        nn_original = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
        nn_original.fit(original_distances)
        _, indices_original = nn_original.kneighbors()
        
        # Remove self from neighbors
        indices_original = indices_original[:, 1:]
        
        continuity = 0.0
        for i in range(n_samples):
            # Find points that are neighbors in original space but not in latent
            original_neighbors = set(indices_original[i])
            
            # Find k-nearest neighbors in latent space
            nn_latent = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
            nn_latent.fit(latent_distances)
            _, indices_latent = nn_latent.kneighbors()
            latent_neighbors = set(indices_latent[i, 1:])
            
            # Count intruders (points that are neighbors in original but not latent)
            intruders = original_neighbors - latent_neighbors
            
            # Compute penalty
            penalty = 0.0
            for intruder in intruders:
                # Find rank of intruder in latent space
                latent_ranks = np.argsort(latent_distances[i])
                intruder_rank = np.where(latent_ranks == intruder)[0][0]
                if n_samples - k > 0:  # Avoid division by zero
                    penalty += (intruder_rank - k) / (n_samples - k)
            
            denominator = k * (2 * n_samples - 3 * k - 1)
            if denominator > 0:  # Avoid division by zero
                continuity += 1.0 - (2.0 / denominator) * penalty
            else:
                continuity += 1.0
        
        return float(continuity / n_samples)
    
    def _compute_neighborhood_preservation(self, original_distances: np.ndarray,
                                          latent_distances: np.ndarray) -> float:
        """Compute neighborhood preservation metric."""
        n_samples = original_distances.shape[0]
        k = min(self.config.quality_metrics_k, n_samples - 2)  # Ensure k+1 <= n_samples
        
        if k < 1:
            return 1.0
        
        # Find k-nearest neighbors in both spaces
        nn_original = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
        nn_original.fit(original_distances)
        _, indices_original = nn_original.kneighbors()
        
        nn_latent = NearestNeighbors(n_neighbors=k+1, metric='precomputed')
        nn_latent.fit(latent_distances)
        _, indices_latent = nn_latent.kneighbors()
        
        # Remove self from neighbors
        indices_original = indices_original[:, 1:]
        indices_latent = indices_latent[:, 1:]
        
        preservation_ratios = []
        for i in range(n_samples):
            original_neighbors = set(indices_original[i])
            latent_neighbors = set(indices_latent[i])
            
            # Compute intersection
            intersection = original_neighbors & latent_neighbors
            preservation_ratio = len(intersection) / k
            preservation_ratios.append(preservation_ratio)
        
        return float(np.mean(preservation_ratios))
    
    def _compute_stress(self, original_distances: np.ndarray, 
                       latent_distances: np.ndarray) -> float:
        """Compute Sammon's stress."""
        # Normalize distances
        original_norm = np.sum(original_distances ** 2)
        
        # Compute stress
        stress = np.sum((original_distances - latent_distances) ** 2) / original_norm
        return float(stress)
    
    def _estimate_curvature(self, original_embeddings: np.ndarray, 
                           latent_embeddings: np.ndarray) -> float:
        """Estimate manifold curvature based on reconstruction error."""
        # Simple curvature estimation based on local reconstruction error
        # In practice, this would use more sophisticated geometric methods
        
        n_samples = len(original_embeddings)
        if n_samples < 3:
            return 0.0
        
        # Compute local reconstruction errors
        local_errors = []
        for i in range(n_samples):
            # Find local neighborhood
            distances = np.linalg.norm(original_embeddings - original_embeddings[i], axis=1)
            local_indices = np.argsort(distances)[:min(10, n_samples)]  # Use fewer neighbors for small datasets
            
            # Compute local reconstruction error
            local_original = original_embeddings[local_indices]
            local_latent = latent_embeddings[local_indices]
            local_reconstructed = self.decode(local_latent)
            local_error = np.mean(np.square(local_original - local_reconstructed))
            local_errors.append(local_error)
        
        # Curvature estimate is the variance of local errors
        curvature = np.var(local_errors)
        return float(curvature)
    
    def _compute_local_density_ratio(self, original_embeddings: np.ndarray,
                                   latent_embeddings: np.ndarray) -> float:
        """Compute ratio of local densities between original and latent space."""
        # Compute local densities using k-nearest neighbors
        n_samples = len(original_embeddings)
        k = min(10, n_samples - 2)  # Ensure k+1 <= n_samples
        
        if k < 1:
            return 1.0
        
        # Original space density
        nn_original = NearestNeighbors(n_neighbors=k+1)
        nn_original.fit(original_embeddings)
        distances_original, _ = nn_original.kneighbors()
        density_original = 1.0 / np.mean(distances_original[:, 1:], axis=1)
        
        # Latent space density
        nn_latent = NearestNeighbors(n_neighbors=k+1)
        nn_latent.fit(latent_embeddings)
        distances_latent, _ = nn_latent.kneighbors()
        density_latent = 1.0 / np.mean(distances_latent[:, 1:], axis=1)
        
        # Compute ratio
        density_ratio = np.mean(density_latent / density_original)
        return float(density_ratio)
    
    def _compute_global_structure_preservation(self, original_distances: np.ndarray,
                                             latent_distances: np.ndarray) -> float:
        """Compute global structure preservation using correlation."""
        # Flatten distance matrices
        original_flat = original_distances[np.triu_indices_from(original_distances, k=1)]
        latent_flat = latent_distances[np.triu_indices_from(latent_distances, k=1)]
        
        # Compute correlation
        correlation = np.corrcoef(original_flat, latent_flat)[0, 1]
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def encode(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Encode embeddings to latent space.
        
        Args:
            embeddings: Input embeddings, shape (n_samples, input_dim)
            
        Returns:
            Latent representations, shape (n_samples, latent_dim)
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before encoding")
        
        # Center the data
        centered_data = embeddings - self.mean_
        
        # Project to latent space
        latent = centered_data @ self.components_
        
        # Store for history
        self.latent_history.append(latent.copy())
        
        return latent
    
    def decode(self, latent: np.ndarray) -> np.ndarray:
        """
        Decode latent representations back to embedding space.
        
        Args:
            latent: Latent representations, shape (n_samples, latent_dim)
            
        Returns:
            Reconstructed embeddings, shape (n_samples, input_dim)
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before decoding")
        
        # Project back to embedding space
        reconstructed = latent @ self.components_.T
        
        # Add back the mean
        reconstructed += self.mean_
        
        # Store for history
        self.embedding_history.append(reconstructed.copy())
        
        return reconstructed
    
    def interpolate_in_latent(self, start_latent: np.ndarray, end_latent: np.ndarray, 
                             steps: int = 10) -> np.ndarray:
        """
        Interpolate between two points in latent space.
        
        Args:
            start_latent: Starting point in latent space
            end_latent: Ending point in latent space
            steps: Number of interpolation steps
            
        Returns:
            Interpolated latent points, shape (steps, latent_dim)
        """
        # Linear interpolation in latent space
        alphas = np.linspace(0, 1, steps)
        interpolated = []
        
        for alpha in alphas:
            point = (1 - alpha) * start_latent + alpha * end_latent
            interpolated.append(point)
        
        return np.array(interpolated)
    
    def get_manifold_curvature(self, embeddings: np.ndarray) -> float:
        """
        Estimate the curvature of the learned manifold.
        
        Args:
            embeddings: Sample embeddings to estimate curvature
            
        Returns:
            Estimated curvature value
        """
        if not self.is_fitted:
            raise ValueError("Autoencoder must be fitted before computing curvature")
        
        if self.quality_metrics is not None:
            return self.quality_metrics.curvature_estimate
        
        # Fallback to simple curvature estimation
        encoded = self.encode(embeddings)
        decoded = self.decode(encoded)
        
        # Compute local curvature as reconstruction error
        curvature = np.mean(np.square(embeddings - decoded))
        
        return float(curvature)
    
    def get_quality_report(self) -> Dict[str, Any]:
        """Get comprehensive quality report for the manifold."""
        if self.quality_metrics is None:
            return {"error": "Quality metrics not computed"}
        
        return {
            "trustworthiness": self.quality_metrics.trustworthiness,
            "continuity": self.quality_metrics.continuity,
            "neighborhood_preservation": self.quality_metrics.neighborhood_preservation,
            "stress": self.quality_metrics.stress,
            "reconstruction_error": self.quality_metrics.reconstruction_error,
            "curvature_estimate": self.quality_metrics.curvature_estimate,
            "local_density_ratio": self.quality_metrics.local_density_ratio,
            "global_structure_preservation": self.quality_metrics.global_structure_preservation,
            "overall_quality": np.mean([
                self.quality_metrics.trustworthiness,
                self.quality_metrics.continuity,
                self.quality_metrics.neighborhood_preservation,
                1.0 - self.quality_metrics.stress,  # Invert stress
                1.0 - self.quality_metrics.reconstruction_error,  # Invert error
                self.quality_metrics.global_structure_preservation
            ])
        }
    
    def save_model(self, filepath: str):
        """Save the trained autoencoder model with quality metrics."""
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        data = {
            'config': self.config.__dict__,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'mean': self.mean_.tolist(),
            'components': self.components_.tolist(),
            'explained_variance': self.explained_variance_.tolist(),
            'reconstruction_error': self.reconstruction_error_,
            'is_fitted': self.is_fitted,
            'quality_metrics': self.quality_metrics.__dict__ if self.quality_metrics else None
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
    
    def load_model(self, filepath: str):
        """Load a trained autoencoder model with quality metrics."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.config = AutoencoderConfig(**data['config'])
        self.input_dim = data['input_dim']
        self.latent_dim = data['latent_dim']
        self.mean_ = np.array(data['mean'])
        self.components_ = np.array(data['components'])
        self.explained_variance_ = np.array(data['explained_variance'])
        self.reconstruction_error_ = data['reconstruction_error']
        self.is_fitted = data['is_fitted']
        
        if data['quality_metrics']:
            self.quality_metrics = ManifoldQualityMetrics(**data['quality_metrics'])


if __name__ == "__main__":
    # Test the autoencoder
    np.random.seed(42)
    
    # Generate synthetic embeddings
    n_samples = 100
    input_dim = 64
    latent_dim = 16
    
    embeddings = np.random.randn(n_samples, input_dim)
    
    # Fit autoencoder
    autoencoder = AutoencoderManifold(input_dim, latent_dim)
    autoencoder.fit(embeddings)
    
    # Test encoding/decoding
    encoded = autoencoder.encode(embeddings)
    decoded = autoencoder.decode(encoded)
    
    print(f"Original shape: {embeddings.shape}")
    print(f"Encoded shape: {encoded.shape}")
    print(f"Decoded shape: {decoded.shape}")
    print(f"Reconstruction error: {np.mean(np.square(embeddings - decoded)):.6f}") 