# Holistic Semantic Continuum (HSC) - Documentation

## Overview

The Holistic Semantic Continuum (HSC) is a modular system that enables continuous semantic interpolation between programs through vector embeddings, manifold learning, and geodesic interpolation. This system provides a foundation for understanding program semantics as a continuous space rather than discrete entities.

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd hsc_poc

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

# Initialize components
engine = DSLToVectorEngine()
manifold = AutoencoderManifold(input_dim=64, latent_dim=16)
interpolator = GeodesicInterpolator(manifold)

# Encode programs
program1 = "def square(x): return x * x"
program2 = "def cube(x): return x * x * x"

# Interpolate between programs
interpolated = interpolator.interpolate(program1, program2, steps=5)
```

## Architecture

The HSC system consists of the following core modules:

### 1. Embedding Module (`embedding/`)
- **Purpose**: Converts DSL programs into vector representations
- **Key Components**:
  - `engine.py`: Main embedding engine
  - `encoder.py`: AST to vector encoding
  - `decoder.py`: Vector to AST decoding

### 2. Manifold Learning Module (`manifold/`)
- **Purpose**: Learns the underlying manifold structure
- **Key Components**:
  - `autoencoder.py`: Neural autoencoder for manifold learning
  - `diffusion.py`: Diffusion maps for manifold discovery
  - `reduction.py`: Dimensionality reduction techniques

### 3. Interpolation Module (`interpolation/`)
- **Purpose**: Computes geodesic paths between programs
- **Key Components**:
  - `geodesic.py`: Main interpolator
  - `pathfinder.py`: Path finding algorithms
  - `validator.py`: Semantic validation

### 4. Evaluation Module (`evaluation/`)
- **Purpose**: Evaluates similarity and interpolation quality
- **Key Components**:
  - `similarity.py`: Similarity evaluator
  - `metrics.py`: Semantic metrics
  - `benchmarks.py`: Benchmark suite

### 5. Visualization Module (`visualization/`)
- **Purpose**: Visualizes embeddings and interpolation paths
- **Key Components**:
  - `plotter.py`: Main visualizer
  - `interactive.py`: Interactive visualizations
  - `ast_viewer.py`: AST visualization

## Configuration

The system can be configured through various configuration classes:

```python
from embedding.engine import EmbeddingConfig
from manifold.autoencoder import AutoencoderConfig
from interpolation.geodesic import InterpolationConfig

# Configure embedding
embedding_config = EmbeddingConfig(
    vector_dim=128,
    normalize=True,
    use_semantic_weights=True
)

# Configure manifold
manifold_config = AutoencoderConfig(
    input_dim=128,
    latent_dim=32,
    hidden_dims=(256, 128, 64)
)

# Configure interpolation
interpolation_config = InterpolationConfig(
    method="geodesic",
    steps=20,
    validate_semantics=True
)
```

## Examples

### Example 1: Basic Program Similarity

```python
from embedding.engine import DSLToVectorEngine

engine = DSLToVectorEngine()

programs = [
    "def square(x): return x * x",
    "def cube(x): return x * x * x",
    "def factorial(n): return 1 if n <= 1 else n * factorial(n-1)"
]

# Find most similar programs
similar = engine.find_most_similar(programs[0], programs[1:], top_k=2)
print(f"Most similar to square: {similar}")
```

### Example 2: Program Interpolation

```python
from manifold.autoencoder import AutoencoderManifold
from interpolation.geodesic import GeodesicInterpolator

# Initialize manifold and interpolator
manifold = AutoencoderManifold(input_dim=64, latent_dim=16)
interpolator = GeodesicInterpolator(manifold)

# Interpolate between programs
start_program = "def square(x): return x * x"
end_program = "def cube(x): return x * x * x"

interpolated = interpolator.interpolate(start_program, end_program, steps=10)
for i, program in enumerate(interpolated):
    print(f"Step {i}: {program}")
```

### Example 3: Evaluation

```python
from evaluation.similarity import SimilarityEvaluator

evaluator = SimilarityEvaluator()

# Evaluate similarity
similarity = evaluator.evaluate_similarity(program1, program2)
print(f"Similarity metrics: {similarity}")

# Evaluate interpolation quality
quality = evaluator.evaluate_interpolation(
    start_program, end_program, interpolated_programs
)
print(f"Interpolation quality: {quality}")
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test module
python -m pytest tests/test_embedding.py

# Run with coverage
python -m pytest --cov=. tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{hsc2024,
  title={Holistic Semantic Continuum: Continuous Program Interpolation},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
``` 