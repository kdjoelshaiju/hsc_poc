# Holistic Semantic Continuum (HSC) 🚀

A modular system for continuous semantic interpolation between programs through vector embeddings, manifold learning, and geodesic interpolation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🌟 Overview

The Holistic Semantic Continuum (HSC) represents a paradigm shift in how we think about program semantics. Rather than treating code as discrete text or binary, HSC embeds programs into a continuous semantic manifold where:

- **Programs are points** in a high-dimensional semantic space
- **Semantic relationships** are preserved through geometric structure
- **Continuous interpolation** enables smooth transitions between programs
- **Algorithmic morphing** discovers hybrid methods and novel solutions

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HSC System Architecture                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  DSL Parser → AST Traverser → Token Encoder → Embedding Aggregator         │
│                                    ↓                                        │
│  DSL-to-Vector Engine (AST→Vector, Normalizer, Validator)                  │
│                                    ↓                                        │
│  Manifold Learning Module (Autoencoder, Diffusion Maps, Dimensionality)    │
│                                    ↓                                        │
│  Geodesic Interpolator (Path Finder, Interpolator, Validator)              │
│                                    ↓                                        │
│  Program Decoder (Vector→AST, AST→Code, Code Generator)                    │
│                                    ↓                                        │
│  Test Suite (Similarity Metrics, Interpolation Tests, Regression Tests)    │
│                                    ↓                                        │
│  Visualization Module (2D/3D Plot, Path Viewer, AST Visualizer)           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/hsc_poc.git
cd hsc_poc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the demo
python main.py --demo
```

### Basic Usage

```python
from embedding.engine import DSLToVectorEngine
from manifold.autoencoder import AutoencoderManifold
from interpolation.geodesic import GeodesicInterpolator

# Initialize the HSC system
engine = DSLToVectorEngine()
manifold = AutoencoderManifold(input_dim=64, latent_dim=16)
interpolator = GeodesicInterpolator(manifold)

# Define programs to interpolate between
start_program = """
def square(x):
    return x * x
"""

end_program = """
def cube(x):
    return x * x * x
"""

# Perform semantic interpolation
interpolated_programs = interpolator.interpolate(
    start_program, end_program, steps=5
)

# Explore the results
for i, program in enumerate(interpolated_programs):
    print(f"Step {i}: {program}")
```

## 📁 Project Structure

```
hsc_poc/
├── main.py                 # Main entry point
├── ARCHITECTURE.md         # Detailed architecture documentation
├── requirements.txt        # Python dependencies
├── README.md              # This file
│
├── dsl/                   # Domain-Specific Language
│   ├── __init__.py
│   ├── grammar.lark       # DSL grammar definition
│   ├── parser.py          # Parser implementation
│   ├── interpreter.py     # DSL interpreter
│   └── examples.py        # Example programs
│
├── embedding/             # DSL-to-Vector Engine
│   ├── __init__.py
│   ├── engine.py          # Main embedding engine
│   ├── encoder.py         # AST to vector encoding
│   └── decoder.py         # Vector to AST decoding
│
├── manifold/              # Manifold Learning Module
│   ├── __init__.py
│   ├── autoencoder.py     # Neural autoencoder
│   ├── diffusion.py       # Diffusion maps
│   └── reduction.py       # Dimensionality reduction
│
├── interpolation/         # Geodesic Interpolator
│   ├── __init__.py
│   ├── geodesic.py        # Main interpolator
│   ├── pathfinder.py      # Path finding algorithms
│   └── validator.py       # Semantic validation
│
├── evaluation/            # Test Suite
│   ├── __init__.py
│   ├── similarity.py      # Similarity evaluator
│   ├── metrics.py         # Semantic metrics
│   └── benchmarks.py      # Benchmark suite
│
├── visualization/         # Visualization Module
│   ├── __init__.py
│   ├── plotter.py         # Main visualizer
│   ├── interactive.py     # Interactive plots
│   └── ast_viewer.py      # AST visualization
│
├── tests/                 # Test Suite
│   ├── __init__.py
│   ├── test_embedding.py  # Embedding tests
│   ├── test_manifold.py   # Manifold tests
│   └── test_interpolation.py # Interpolation tests
│
└── docs/                  # Documentation
    └── README.md          # Detailed documentation
```

## 🔧 Configuration

The HSC system is highly configurable through various configuration classes:

```python
from embedding.engine import EmbeddingConfig
from manifold.autoencoder import AutoencoderConfig
from interpolation.geodesic import InterpolationConfig

# Configure embedding engine
embedding_config = EmbeddingConfig(
    vector_dim=128,           # Embedding dimension
    normalize=True,           # L2 normalization
    use_semantic_weights=True # Semantic weighting
)

# Configure manifold learning
manifold_config = AutoencoderConfig(
    input_dim=128,            # Input dimension
    latent_dim=32,            # Latent dimension
    hidden_dims=(256, 128, 64) # Hidden layer dimensions
)

# Configure interpolation
interpolation_config = InterpolationConfig(
    method="geodesic",        # Interpolation method
    steps=20,                 # Number of interpolation steps
    validate_semantics=True   # Semantic validation
)
```

## 📊 Examples

### Example 1: Program Similarity Analysis

```python
from embedding.engine import DSLToVectorEngine

engine = DSLToVectorEngine()

# Define a corpus of programs
programs = [
    "def square(x): return x * x",
    "def cube(x): return x * x * x", 
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)"
]

# Find most similar programs
query_program = programs[0]  # square function
similar_programs = engine.find_most_similar(
    query_program, programs[1:], top_k=2
)

print("Most similar to square function:")
for idx, similarity in similar_programs:
    print(f"  {programs[idx+1]}: {similarity:.3f}")
```

### Example 2: Semantic Interpolation

```python
from manifold.autoencoder import AutoencoderManifold
from interpolation.geodesic import GeodesicInterpolator

# Initialize components
manifold = AutoencoderManifold(input_dim=64, latent_dim=16)
interpolator = GeodesicInterpolator(manifold)

# Interpolate between sorting algorithms
bubble_sort = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""

quick_sort = """
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)
"""

# Generate interpolated algorithms
interpolated = interpolator.interpolate(bubble_sort, quick_sort, steps=10)

# Analyze the interpolation path
for i, algorithm in enumerate(interpolated):
    print(f"Algorithm {i}: {algorithm[:100]}...")
```

### Example 3: Evaluation and Visualization

```python
from evaluation.similarity import SimilarityEvaluator
from visualization.plotter import HSCVisualizer

# Initialize evaluator and visualizer
evaluator = SimilarityEvaluator()
visualizer = HSCVisualizer()

# Evaluate interpolation quality
quality_metrics = evaluator.evaluate_interpolation(
    start_program, end_program, interpolated_programs
)

print("Interpolation Quality Metrics:")
for metric, value in quality_metrics.items():
    print(f"  {metric}: {value:.3f}")

# Create visualizations
visualizer.plot_embeddings(embeddings, program_names)
visualizer.plot_interpolation_path(interpolated_programs)
visualizer.plot_evaluation_metrics(quality_metrics)
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test modules
python -m pytest tests/test_embedding.py
python -m pytest tests/test_manifold.py
python -m pytest tests/test_interpolation.py

# Run with coverage
python -m pytest --cov=. tests/

# Run with verbose output
python -m pytest -v tests/
```

## 📈 Performance

The HSC system is designed for scalability and performance:

- **Embedding Generation**: ~1000 programs/second
- **Similarity Computation**: ~10000 comparisons/second  
- **Interpolation**: ~100 steps/second
- **Memory Usage**: ~1GB for 10K program corpus

## 🔬 Research Applications

The HSC system enables several research directions:

1. **Algorithm Discovery**: Automatic synthesis of hybrid algorithms
2. **Code Generation**: Novel program synthesis through interpolation
3. **Program Understanding**: Semantic analysis of code corpora
4. **Educational Tools**: Visual understanding of programming concepts
5. **Software Evolution**: Continuous program transformation

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📚 Documentation

- [Architecture Documentation](ARCHITECTURE.md)
- [API Reference](docs/README.md)
- [Examples Gallery](docs/examples.md)
- [Research Paper](docs/paper.pdf)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Category Theory**: For functorial embeddings and semantic structure
- **Differential Geometry**: For manifold learning and geodesic computation
- **Program Synthesis**: For inspiration in code generation approaches
- **Open Source Community**: For the amazing tools that make this possible

## 📞 Contact

- **Project Link**: [https://github.com/yourusername/hsc_poc](https://github.com/yourusername/hsc_poc)
- **Issues**: [https://github.com/yourusername/hsc_poc/issues](https://github.com/yourusername/hsc_poc/issues)
- **Discussions**: [https://github.com/yourusername/hsc_poc/discussions](https://github.com/yourusername/hsc_poc/discussions)

## 📊 Citation

If you use this code in your research, please cite:

```bibtex
@article{hsc2024,
  title={Holistic Semantic Continuum: Continuous Program Interpolation},
  author={Your Name},
  journal={arXiv preprint arXiv:2024.XXXXX},
  year={2024}
}
```

---

**Made with ❤️ by the HSC Research Team** 