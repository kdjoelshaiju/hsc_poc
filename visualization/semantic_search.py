"""
Semantic Search Visualization

This module provides visualization tools for the semantic search engine,
including 2D/3D embedding plots and transformation path visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union
from pathlib import Path
import json

from evaluation.benchmarks import SemanticSearchEngine, CodeSnippet, TransformationPath

class SemanticSearchVisualizer:
    """Visualization tools for semantic search and transformation."""
    
    def __init__(self, output_dir: str = "visualization_outputs"):
        """Initialize the visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting style
        try:
            plt.style.use('seaborn-v0_8')
        except:
            plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_embedding_space(self, engine: SemanticSearchEngine, 
                           method: str = "pca", 
                           save_path: Optional[str] = None) -> None:
        """Plot the 2D embedding space of all algorithms."""
        if not engine.search_index:
            engine.load_index()
        
        # Extract vectors and metadata
        vectors = []
        names = []
        categories = []
        
        for snippet in engine.search_index:
            if snippet.vector is not None:
                vectors.append(snippet.vector)
                names.append(snippet.name)
                categories.append(snippet.category)
        
        if not vectors:
            print("❌ No valid vectors found for visualization")
            return
        
        vectors = np.array(vectors)
        
        # Dimensionality reduction
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(vectors)
        
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(vectors)
        
        else:
            print(f"⚠️ Method {method} not available, falling back to PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(vectors)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Color by category
        unique_categories = list(set(categories))
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(unique_categories)))
        
        for i, category in enumerate(unique_categories):
            mask = [c == category for c in categories]
            category_coords = coords_2d[mask]
            category_names = [n for n, c in zip(names, categories) if c == category]
            
            ax.scatter(category_coords[:, 0], category_coords[:, 1], 
                      c=[colors[i]], label=category, s=100, alpha=0.7)
            
            # Add labels for some points
            for j, name in enumerate(category_names):
                if j % 2 == 0:  # Label every other point to avoid clutter
                    ax.annotate(name, (category_coords[j, 0], category_coords[j, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Semantic Embedding Space ({method.upper()})')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"💾 Embedding space plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_transformation_path(self, path: TransformationPath,
                               method: str = "pca",
                               save_path: Optional[str] = None) -> None:
        """Plot the transformation path between two programs."""
        if not path.path_vectors:
            print("❌ No transformation path vectors found")
            return
        
        # Dimensionality reduction for path vectors
        vectors = np.array(path.path_vectors)
        
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(vectors)
        
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            coords_2d = reducer.fit_transform(vectors)
        
        else:
            print(f"⚠️ Method {method} not available, falling back to PCA")
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            coords_2d = reducer.fit_transform(vectors)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot the path
        ax.plot(coords_2d[:, 0], coords_2d[:, 1], 'b-', alpha=0.5, linewidth=2)
        
        # Plot points with confidence-based coloring
        scatter = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], 
                           c=path.confidence_scores, cmap='RdYlGn', 
                           s=100, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Confidence Score')
        
        # Label start and end points
        ax.annotate(f'Start: {path.start.name}', 
                   (coords_2d[0, 0], coords_2d[0, 1]),
                   xytext=(10, 10), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                   fontsize=10)
        
        ax.annotate(f'End: {path.end.name}', 
                   (coords_2d[-1, 0], coords_2d[-1, 1]),
                   xytext=(10, -20), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                   fontsize=10)
        
        # Add step numbers
        for i, (x, y) in enumerate(coords_2d[1:-1], 1):
            ax.annotate(f'Step {i}', (x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=8)
        
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Transformation Path: {path.start.name} → {path.end.name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"💾 Transformation path plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_similarity_matrix(self, engine: SemanticSearchEngine,
                             save_path: Optional[str] = None) -> None:
        """Plot similarity matrix between all algorithms."""
        if not engine.search_index:
            engine.load_index()
        
        # Compute similarity matrix
        vectors = []
        names = []
        
        for snippet in engine.search_index:
            if snippet.vector is not None:
                vectors.append(snippet.vector)
                names.append(snippet.name)
        
        if not vectors:
            print("❌ No valid vectors found for similarity matrix")
            return
        
        vectors = np.array(vectors)
        
        # Compute cosine similarity matrix
        similarity_matrix = np.zeros((len(vectors), len(vectors)))
        for i in range(len(vectors)):
            for j in range(len(vectors)):
                similarity_matrix[i, j] = engine.similarity_evaluator._compute_cosine_similarity(
                    vectors[i], vectors[j]
                )
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='viridis', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Cosine Similarity')
        
        # Set ticks and labels
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticklabels(names)
        
        ax.set_title('Semantic Similarity Matrix')
        ax.set_xlabel('Algorithms')
        ax.set_ylabel('Algorithms')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"💾 Similarity matrix plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_category_distribution(self, engine: SemanticSearchEngine,
                                 save_path: Optional[str] = None) -> None:
        """Plot distribution of algorithms by category."""
        if not engine.search_index:
            engine.load_index()
        
        # Count algorithms by category
        category_counts = {}
        for snippet in engine.search_index:
            cat = snippet.category
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(categories)))
        bars = ax.bar(categories, counts, color=colors)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{count}', ha='center', va='bottom')
        
        ax.set_xlabel('Algorithm Categories')
        ax.set_ylabel('Number of Algorithms')
        ax.set_title('Distribution of Algorithms by Category')
        ax.set_xticklabels(categories, rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"💾 Category distribution plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def create_comprehensive_report(self, engine: SemanticSearchEngine,
                                  save_dir: Optional[Union[str, Path]] = None) -> None:
        """Create a comprehensive visualization report."""
        if save_dir is None:
            save_dir = self.output_dir
        elif isinstance(save_dir, str):
            save_dir = Path(save_dir)
        
        save_dir.mkdir(exist_ok=True)
        
        print("📊 Creating comprehensive visualization report...")
        
        # Generate all plots
        self.plot_embedding_space(engine, save_path=str(save_dir / "embedding_space.png"))
        self.plot_similarity_matrix(engine, save_path=str(save_dir / "similarity_matrix.png"))
        self.plot_category_distribution(engine, save_path=str(save_dir / "category_distribution.png"))
        
        # Save statistics
        stats = engine.get_statistics()
        with open(save_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Comprehensive report saved to {save_dir}")
    
    def visualize_search_results(self, results: List, query: str,
                               save_path: Optional[str] = None) -> None:
        """Visualize search results as a bar chart."""
        if not results:
            print("❌ No search results to visualize")
            return
        
        # Extract data
        names = [r.snippet.name for r in results]
        scores = [r.similarity_score for r in results]
        categories = [r.snippet.category for r in results]
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        cmap = plt.get_cmap("tab10")
        colors = cmap(np.linspace(0, 1, len(names)))
        bars = ax.barh(names, scores, color=colors)
        
        # Add value labels
        for bar, score in zip(bars, scores):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{score:.3f}', ha='left', va='center')
        
        ax.set_xlabel('Similarity Score')
        ax.set_ylabel('Algorithms')
        ax.set_title(f'Search Results for: "{query}"')
        ax.set_xlim(0, 1.1)
        
        # Add category information
        for i, (name, category) in enumerate(zip(names, categories)):
            ax.text(-0.05, i, f'({category})', ha='right', va='center', 
                   fontsize=8, alpha=0.7)
        
        plt.tight_layout()
        
        # Save or show
        if save_path:
            plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
            print(f"💾 Search results plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close() 