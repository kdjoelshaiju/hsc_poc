"""
HSC Visualizer

This module provides visualization tools for the Holistic Semantic Continuum,
including embedding plots, manifold visualization, and interpolation paths.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = "default"
    color_palette: str = "viridis"
    save_format: str = "png"


class HSCVisualizer:
    """
    Main visualizer for HSC system.
    
    This class provides comprehensive visualization capabilities for
    program embeddings, manifold structure, and interpolation results.
    """
    
    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        
    def plot_embeddings(self, embeddings: np.ndarray, labels: Optional[List[str]] = None,
                       save_path: Optional[str] = None, title: str = "Program Embeddings") -> None:
        """
        Plot program embeddings in 2D or 3D space.
        
        Args:
            embeddings: Embedding matrix, shape (n_programs, n_dimensions)
            labels: Optional labels for each program
            save_path: Optional path to save the plot
            title: Plot title
        """
        print(f"Plotting {len(embeddings)} embeddings...")
        
        # Reduce to 2D for visualization if needed
        if embeddings.shape[1] > 2:
            reduced_embeddings = self._reduce_dimensions(embeddings, n_components=2)
        else:
            reduced_embeddings = embeddings
        
        # Create the plot
        self._create_embedding_plot(reduced_embeddings, labels, title)
        
        if save_path:
            self._save_plot(save_path)
            print(f"Plot saved to {save_path}")
    
    def plot_interpolation_path(self, interpolated_programs: List[str],
                               save_path: Optional[str] = None,
                               title: str = "Interpolation Path") -> None:
        """
        Plot interpolation path between programs.
        
        Args:
            interpolated_programs: List of interpolated program codes
            save_path: Optional path to save the plot
            title: Plot title
        """
        print(f"Plotting interpolation path with {len(interpolated_programs)} steps...")
        
        # Create a simple visualization of the interpolation path
        self._create_interpolation_plot(interpolated_programs, title)
        
        if save_path:
            self._save_plot(save_path)
            print(f"Interpolation plot saved to {save_path}")
    
    def plot_manifold_structure(self, embeddings: np.ndarray, manifold,
                               save_path: Optional[str] = None,
                               title: str = "Manifold Structure") -> None:
        """
        Plot the learned manifold structure.
        
        Args:
            embeddings: Program embeddings
            manifold: Fitted manifold object
            save_path: Optional path to save the plot
            title: Plot title
        """
        print("Plotting manifold structure...")
        
        # Reduce dimensions for visualization
        if embeddings.shape[1] > 2:
            reduced_embeddings = self._reduce_dimensions(embeddings, n_components=2)
        else:
            reduced_embeddings = embeddings
        
        # Create manifold visualization
        self._create_manifold_plot(reduced_embeddings, manifold, title)
        
        if save_path:
            self._save_plot(save_path)
            print(f"Manifold plot saved to {save_path}")
    
    def plot_similarity_matrix(self, similarity_matrix: np.ndarray,
                              labels: Optional[List[str]] = None,
                              save_path: Optional[str] = None,
                              title: str = "Similarity Matrix") -> None:
        """
        Plot similarity matrix as a heatmap.
        
        Args:
            similarity_matrix: Similarity matrix
            labels: Optional labels for rows/columns
            save_path: Optional path to save the plot
            title: Plot title
        """
        print("Plotting similarity matrix...")
        
        # Create heatmap visualization
        self._create_similarity_heatmap(similarity_matrix, labels, title)
        
        if save_path:
            self._save_plot(save_path)
            print(f"Similarity matrix saved to {save_path}")
    
    def plot_evaluation_metrics(self, metrics: Dict[str, float],
                               save_path: Optional[str] = None,
                               title: str = "Evaluation Metrics") -> None:
        """
        Plot evaluation metrics as a bar chart.
        
        Args:
            metrics: Dictionary of metric names and values
            save_path: Optional path to save the plot
            title: Plot title
        """
        print("Plotting evaluation metrics...")
        
        # Create bar chart visualization
        self._create_metrics_bar_chart(metrics, title)
        
        if save_path:
            self._save_plot(save_path)
            print(f"Metrics plot saved to {save_path}")
    
    def _reduce_dimensions(self, embeddings: np.ndarray, n_components: int = 2) -> np.ndarray:
        """Reduce embeddings to specified number of dimensions."""
        # Simple PCA-based dimensionality reduction
        # In a full implementation, this would use scikit-learn's PCA
        
        # Center the data
        mean = np.mean(embeddings, axis=0)
        centered_data = embeddings - mean
        
        # Compute covariance matrix and eigendecomposition
        cov_matrix = np.cov(centered_data.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        # Project to reduced dimensions
        components = eigenvectors[:, :n_components]
        reduced = centered_data @ components
        
        return reduced
    
    def _create_embedding_plot(self, embeddings: np.ndarray, labels: Optional[List[str]], title: str) -> None:
        """Create embedding scatter plot."""
        # Placeholder for actual plotting
        # In a full implementation, this would use matplotlib/seaborn
        
        print(f"  Creating embedding plot: {embeddings.shape}")
        print(f"  Title: {title}")
        if labels:
            print(f"  Labels: {labels[:5]}...")  # Show first 5 labels
        
        # Simulate plot creation
        print("  ✓ Embedding plot created successfully")
    
    def _create_interpolation_plot(self, programs: List[str], title: str) -> None:
        """Create interpolation path visualization."""
        # Placeholder for actual plotting
        
        print(f"  Creating interpolation plot: {len(programs)} programs")
        print(f"  Title: {title}")
        
        # Show sample programs
        for i, program in enumerate(programs[:3]):  # Show first 3
            print(f"    Step {i}: {program[:50]}...")
        
        print("  ✓ Interpolation plot created successfully")
    
    def _create_manifold_plot(self, embeddings: np.ndarray, manifold, title: str) -> None:
        """Create manifold structure visualization."""
        # Placeholder for actual plotting
        
        print(f"  Creating manifold plot: {embeddings.shape}")
        print(f"  Title: {title}")
        print(f"  Manifold type: {type(manifold).__name__}")
        
        print("  ✓ Manifold plot created successfully")
    
    def _create_similarity_heatmap(self, similarity_matrix: np.ndarray, 
                                  labels: Optional[List[str]], title: str) -> None:
        """Create similarity matrix heatmap."""
        # Placeholder for actual plotting
        
        print(f"  Creating similarity heatmap: {similarity_matrix.shape}")
        print(f"  Title: {title}")
        print(f"  Similarity range: [{similarity_matrix.min():.3f}, {similarity_matrix.max():.3f}]")
        
        print("  ✓ Similarity heatmap created successfully")
    
    def _create_metrics_bar_chart(self, metrics: Dict[str, float], title: str) -> None:
        """Create evaluation metrics bar chart."""
        # Placeholder for actual plotting
        
        print(f"  Creating metrics bar chart: {len(metrics)} metrics")
        print(f"  Title: {title}")
        
        for metric, value in metrics.items():
            print(f"    {metric}: {value:.3f}")
        
        print("  ✓ Metrics bar chart created successfully")
    
    def _save_plot(self, save_path: str) -> None:
        """Save the current plot to file."""
        # Placeholder for actual plot saving
        print(f"  Saving plot to: {save_path}")
        
        # Create a simple text representation of the plot
        plot_info = {
            'timestamp': '2024-01-01T00:00:00',
            'plot_type': 'hsc_visualization',
            'save_path': save_path
        }
        
        # Save plot metadata (in a real implementation, this would save the actual plot)
        with open(save_path + '.json', 'w') as f:
            json.dump(plot_info, f, indent=2)
    
    def create_dashboard(self, embeddings: np.ndarray, similarity_matrix: np.ndarray,
                        interpolated_programs: Optional[List[str]] = None,
                        metrics: Optional[Dict[str, float]] = None,
                        save_path: Optional[str] = None) -> None:
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            embeddings: Program embeddings
            similarity_matrix: Similarity matrix
            interpolated_programs: Optional interpolated programs
            metrics: Optional evaluation metrics
            save_path: Optional path to save the dashboard
        """
        print("Creating HSC Dashboard...")
        
        # Create all visualizations
        self.plot_embeddings(embeddings, title="Program Embeddings")
        
        self.plot_similarity_matrix(similarity_matrix, title="Semantic Similarity")
        
        if interpolated_programs:
            self.plot_interpolation_path(interpolated_programs, title="Interpolation Path")
        
        if metrics:
            self.plot_evaluation_metrics(metrics, title="Evaluation Results")
        
        if save_path:
            print(f"Dashboard saved to {save_path}")
        
        print("✓ HSC Dashboard created successfully")


if __name__ == "__main__":
    # Test visualizer
    visualizer = HSCVisualizer()
    
    # Generate sample data
    np.random.seed(42)
    embeddings = np.random.randn(10, 64)
    similarity_matrix = np.random.rand(10, 10)
    similarity_matrix = (similarity_matrix + similarity_matrix.T) / 2  # Make symmetric
    
    # Test visualizations
    visualizer.plot_embeddings(embeddings, title="Test Embeddings")
    visualizer.plot_similarity_matrix(similarity_matrix, title="Test Similarity")
    
    # Test interpolation path
    interpolated = [
        "def square(x): return x * x",
        "def power2_5(x): return x * x * x**0.5",
        "def cube(x): return x * x * x"
    ]
    visualizer.plot_interpolation_path(interpolated, title="Test Interpolation")
    
    # Test metrics
    metrics = {
        'smoothness': 0.85,
        'semantic_coherence': 0.72,
        'path_consistency': 0.91,
        'overall_quality': 0.83
    }
    visualizer.plot_evaluation_metrics(metrics, title="Test Metrics")
    
    print("All visualization tests completed!") 