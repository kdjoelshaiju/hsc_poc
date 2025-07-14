"""
HSC Semantic Search - Streamlit Web UI

A comprehensive web interface for the Holistic Semantic Continuum system,
featuring interactive search, visualization, and analytics.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time
from typing import List, Dict, Optional, Tuple
import json

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from evaluation.benchmarks import SemanticSearchEngine, BenchmarkSuite
from visualization.semantic_search import SemanticSearchVisualizer

# Page configuration
st.set_page_config(
    page_title="HSC Semantic Search",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .search-box {
        background: #f0f2f6;
        padding: 2rem;
        border-radius: 15px;
        border: 2px solid #e0e0e0;
    }
    
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    """Load the semantic search engine with caching."""
    engine = SemanticSearchEngine()
    if not Path("data/search_index.json").exists():
        with st.spinner("Building search corpus for the first time..."):
            engine.build_search_corpus()
    else:
        engine.load_index()
    return engine

@st.cache_data
def get_engine_statistics(_engine):
    """Get engine statistics with caching."""
    return _engine.get_statistics()

def create_embedding_space_plot(_engine, method="pca"):
    """Create interactive embedding space plot."""
    if not _engine.search_index:
        return None
    
    # Extract data
    vectors = []
    names = []
    categories = []
    descriptions = []
    
    for snippet in _engine.search_index:
        if snippet.vector is not None:
            vectors.append(snippet.vector)
            names.append(snippet.name)
            categories.append(snippet.category)
            descriptions.append(snippet.description)
    
    if not vectors:
        return None
    
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
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        coords_2d = reducer.fit_transform(vectors)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'x': coords_2d[:, 0],
        'y': coords_2d[:, 1],
        'name': names,
        'category': categories,
        'description': descriptions
    })
    
    # Create interactive plot
    fig = px.scatter(
        df, x='x', y='y', 
        color='category',
        hover_data=['name', 'description'],
        title=f'Semantic Embedding Space ({method.upper()})',
        labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'}
    )
    
    fig.update_layout(
        width=800,
        height=600,
        showlegend=True,
        hovermode='closest'
    )
    
    return fig

def create_similarity_matrix_plot(_engine):
    """Create interactive similarity matrix heatmap."""
    if not _engine.search_index:
        return None
    
    # Extract vectors and names
    vectors = []
    names = []
    
    for snippet in _engine.search_index:
        if snippet.vector is not None:
            vectors.append(snippet.vector)
            names.append(snippet.name)
    
    if not vectors:
        return None
    
    vectors = np.array(vectors)
    
    # Compute similarity matrix
    similarity_matrix = np.zeros((len(vectors), len(vectors)))
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            similarity_matrix[i, j] = _engine.similarity_evaluator._compute_cosine_similarity(
                vectors[i], vectors[j]
            )
    
    # Create heatmap
    fig = px.imshow(
        similarity_matrix,
        x=names,
        y=names,
        color_continuous_scale='viridis',
        title='Semantic Similarity Matrix',
        labels={'x': 'Algorithms', 'y': 'Algorithms', 'color': 'Cosine Similarity'}
    )
    
    fig.update_layout(
        width=800,
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig

def create_category_distribution_plot(stats):
    """Create category distribution bar chart."""
    categories = list(stats['categories'].keys())
    counts = list(stats['categories'].values())
    
    fig = px.bar(
        x=categories,
        y=counts,
        title='Distribution of Algorithms by Category',
        labels={'x': 'Categories', 'y': 'Number of Algorithms'},
        color=counts,
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        width=800,
        height=400,
        xaxis_tickangle=-45
    )
    
    return fig

def create_search_results_plot(results, query):
    """Create search results visualization."""
    if not results:
        return None
    
    names = [r.snippet.name for r in results]
    scores = [r.similarity_score for r in results]
    categories = [r.snippet.category for r in results]
    
    fig = px.bar(
        x=scores,
        y=names,
        orientation='h',
        title=f'Search Results for: "{query}"',
        labels={'x': 'Similarity Score', 'y': 'Algorithms'},
        color=categories,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    
    fig.update_layout(
        width=800,
        height=400,
        xaxis_range=[0, 1.1]
    )
    
    return fig

def create_transformation_path_plot(path):
    """Create transformation path visualization."""
    if not path or not path.path_vectors:
        return None
    
    vectors = np.array(path.path_vectors)
    
    # Dimensionality reduction
    from sklearn.decomposition import PCA
    reducer = PCA(n_components=2)
    coords_2d = reducer.fit_transform(vectors)
    
    # Create path plot
    fig = go.Figure()
    
    # Add path line
    fig.add_trace(go.Scatter(
        x=coords_2d[:, 0],
        y=coords_2d[:, 1],
        mode='lines+markers',
        line=dict(color='blue', width=3),
        marker=dict(
            size=10,
            color=path.confidence_scores,
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Confidence Score")
        ),
        text=[f"Step {i}: {score:.3f}" for i, score in enumerate(path.confidence_scores)],
        hovertemplate='%{text}<extra></extra>'
    ))
    
    # Add start and end labels
    fig.add_annotation(
        x=coords_2d[0, 0], y=coords_2d[0, 1],
        text=f"Start: {path.start.name}",
        showarrow=True,
        arrowhead=2,
        bgcolor="lightblue"
    )
    
    fig.add_annotation(
        x=coords_2d[-1, 0], y=coords_2d[-1, 1],
        text=f"End: {path.end.name}",
        showarrow=True,
        arrowhead=2,
        bgcolor="lightgreen"
    )
    
    fig.update_layout(
        title=f'Transformation Path: {path.start.name} → {path.end.name}',
        width=800,
        height=600,
        xaxis_title='PCA Component 1',
        yaxis_title='PCA Component 2'
    )
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 HSC Semantic Search</h1>', unsafe_allow_html=True)
    st.markdown("**Holistic Semantic Continuum - Interactive Code Search & Transformation**")
    
    # Load engine
    with st.spinner("Loading semantic search engine..."):
        engine = load_engine()
    
    # Sidebar
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🔍 Search", "🔄 Transform", "📊 Analytics", "🎨 Visualizations", "⚡ Benchmarks"]
    )
    
    # Dashboard Page
    if page == "🏠 Dashboard":
        st.header("📊 System Overview")
        
        # Get statistics
        stats = get_engine_statistics(engine)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📚 Total Algorithms</h3>
                <h2>{stats['total_algorithms']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔢 Vector Dimension</h3>
                <h2>{stats['vector_dimension']}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📝 Total Code Length</h3>
                <h2>{stats['total_code_length']:,}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📂 Categories</h3>
                <h2>{len(stats['categories'])}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick search
        st.subheader("🔍 Quick Search")
        query = st.text_input("Enter a code query:", placeholder="e.g., def search(graph, start):")
        
        if query:
            with st.spinner("Searching..."):
                results = engine.search(query, top_k=5)
            
            if results:
                st.success(f"Found {len(results)} results!")
                for i, result in enumerate(results, 1):
                    with st.expander(f"{i}. {result.snippet.name} (Score: {result.similarity_score:.3f})"):
                        st.write(f"**Category:** {result.snippet.category}")
                        st.write(f"**Description:** {result.snippet.description}")
                        st.code(result.snippet.code, language="python")
            else:
                st.warning("No results found.")
        
        # Recent activity
        st.subheader("📈 Recent Activity")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Popular Categories:**")
            for category, count in sorted(stats['categories'].items(), key=lambda x: x[1], reverse=True)[:5]:
                st.write(f"• {category}: {count} algorithms")
        
        with col2:
            st.write("**System Status:**")
            st.write("✅ Search engine loaded")
            st.write("✅ Index ready")
            st.write("✅ Transformations available")
    
    # Search Page
    elif page == "🔍 Search":
        st.header("🔍 Semantic Code Search")
        
        # Search interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            query = st.text_input("Enter your search query:", placeholder="e.g., def sort(array):")
        
        with col2:
            top_k = st.selectbox("Results:", [3, 5, 10, 15, 20])
        
        if st.button("🔍 Search", type="primary"):
            if query:
                with st.spinner("Searching..."):
                    start_time = time.time()
                    results = engine.search(query, top_k=top_k)
                    search_time = time.time() - start_time
                
                if results:
                    st.success(f"Found {len(results)} results in {search_time:.3f}s!")
                    
                    # Results visualization
                    fig = create_search_results_plot(results, query)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed results
                    st.subheader("📋 Detailed Results")
                    for i, result in enumerate(results, 1):
                        with st.expander(f"{i}. {result.snippet.name} (Score: {result.similarity_score:.3f})"):
                            col1, col2 = st.columns([1, 2])
                            
                            with col1:
                                st.write(f"**Category:** {result.snippet.category}")
                                st.write(f"**Similarity:** {result.similarity_score:.3f}")
                                st.write(f"**Code Length:** {len(result.snippet.code)} chars")
                            
                            with col2:
                                st.write(f"**Description:** {result.snippet.description}")
                                st.code(result.snippet.code, language="python")
                else:
                    st.warning("No results found.")
            else:
                st.error("Please enter a search query.")
    
    # Transform Page
    elif page == "🔄 Transform":
        st.header("🔄 Code Transformation")
        
        # Get available algorithms
        algorithms = [snippet.name for snippet in engine.search_index]
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_alg = st.selectbox("Start Algorithm:", algorithms, index=0)
        
        with col2:
            end_alg = st.selectbox("End Algorithm:", algorithms, index=1)
        
        steps = st.slider("Number of steps:", 3, 10, 5)
        
        if st.button("🔄 Transform", type="primary"):
            with st.spinner("Computing transformation path..."):
                start_time = time.time()
                path = engine.transform(start_alg, end_alg, steps=steps)
                transform_time = time.time() - start_time
            
            if path:
                st.success(f"Transformation completed in {transform_time:.3f}s!")
                
                # Transformation path visualization
                fig = create_transformation_path_plot(path)
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
                
                # Transformation details
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Average Confidence", f"{sum(path.confidence_scores) / len(path.confidence_scores):.3f}")
                
                with col2:
                    st.metric("Steps", len(path.intermediate_steps))
                
                with col3:
                    st.metric("Min Confidence", f"{min(path.confidence_scores):.3f}")
                
                # Intermediate steps
                st.subheader("📋 Transformation Steps")
                for i, (step, confidence) in enumerate(zip(path.intermediate_steps, path.confidence_scores)):
                    with st.expander(f"Step {i+1} (Confidence: {confidence:.3f})"):
                        st.code(step.code, language="python")
            else:
                st.error("Transformation failed. Please try different algorithms.")
    
    # Analytics Page
    elif page == "📊 Analytics":
        st.header("📊 Analytics Dashboard")
        
        stats = get_engine_statistics(engine)
        
        # Overview metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Algorithms", stats['total_algorithms'])
        
        with col2:
            st.metric("Categories", len(stats['categories']))
        
        with col3:
            st.metric("Avg Code Length", f"{stats['average_code_length']:.1f}")
        
        with col4:
            st.metric("Vector Dimension", stats['vector_dimension'])
        
        # Category distribution
        st.subheader("📂 Category Distribution")
        fig = create_category_distribution_plot(stats)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Similarity analysis
        st.subheader("🔍 Similarity Analysis")
        
        # Find most similar pairs
        most_similar_pairs = []
        for i, snippet1 in enumerate(engine.search_index):
            for j, snippet2 in enumerate(engine.search_index[i+1:], i+1):
                if snippet1.vector is not None and snippet2.vector is not None:
                    similarity = engine.similarity_evaluator._compute_cosine_similarity(
                        snippet1.vector, snippet2.vector
                    )
                    most_similar_pairs.append((similarity, snippet1.name, snippet2.name))
        
        most_similar_pairs.sort(reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Similar Pairs:**")
            for i, (similarity, name1, name2) in enumerate(most_similar_pairs[:10], 1):
                st.write(f"{i}. {name1} ↔ {name2}: {similarity:.3f}")
        
        with col2:
            st.write("**Similarity Statistics:**")
            similarities = [pair[0] for pair in most_similar_pairs]
            if similarities:
                st.write(f"• Average: {np.mean(similarities):.3f}")
                st.write(f"• Median: {np.median(similarities):.3f}")
                st.write(f"• Max: {np.max(similarities):.3f}")
                st.write(f"• Min: {np.min(similarities):.3f}")
        
        # Code length analysis
        st.subheader("📏 Code Length Analysis")
        code_lengths = [len(snippet.code) for snippet in engine.search_index]
        
        fig = px.histogram(
            x=code_lengths,
            title="Distribution of Code Lengths",
            labels={'x': 'Code Length (characters)', 'y': 'Count'},
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Visualizations Page
    elif page == "🎨 Visualizations":
        st.header("🎨 Interactive Visualizations")
        
        # Embedding space
        st.subheader("🌌 Semantic Embedding Space")
        
        method = st.selectbox("Dimensionality Reduction Method:", ["pca", "tsne"])
        
        fig = create_embedding_space_plot(engine, method)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No embedding data available.")
        
        # Similarity matrix
        st.subheader("🔥 Similarity Matrix")
        
        fig = create_similarity_matrix_plot(engine)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.error("No similarity data available.")
        
        # 3D embedding (if available)
        st.subheader("🌍 3D Embedding Space")
        
        if st.button("Generate 3D Plot"):
            with st.spinner("Computing 3D embedding..."):
                # Extract vectors
                vectors = []
                names = []
                categories = []
                
                for snippet in engine.search_index:
                    if snippet.vector is not None:
                        vectors.append(snippet.vector)
                        names.append(snippet.name)
                        categories.append(snippet.category)
                
                if vectors:
                    vectors = np.array(vectors)
                    
                    # 3D PCA
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=3)
                    coords_3d = reducer.fit_transform(vectors)
                    
                    # Create 3D scatter plot
                    fig_3d = px.scatter_3d(
                        x=coords_3d[:, 0],
                        y=coords_3d[:, 1],
                        z=coords_3d[:, 2],
                        color=categories,
                        hover_name=names,
                        title="3D Semantic Embedding Space"
                    )
                    
                    st.plotly_chart(fig_3d, use_container_width=True)
                else:
                    st.error("No embedding data available.")
    
    # Benchmarks Page
    elif page == "⚡ Benchmarks":
        st.header("⚡ Performance Benchmarks")
        
        if st.button("🚀 Run Benchmarks", type="primary"):
            with st.spinner("Running comprehensive benchmarks..."):
                suite = BenchmarkSuite()
                results = suite.run_benchmarks()
            
            st.success("Benchmarks completed!")
            
            # Display results
            st.subheader("📊 Benchmark Results")
            
            # Create results visualization
            if hasattr(results, 'metrics'):
                metrics_df = pd.DataFrame(results.metrics)
                st.dataframe(metrics_df)
                
                # Performance charts
                if 'search_time' in metrics_df.columns:
                    fig = px.line(
                        metrics_df,
                        x='query_length',
                        y='search_time',
                        title="Search Performance vs Query Length"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            st.info("Check the console for detailed benchmark output.")

if __name__ == "__main__":
    main() 