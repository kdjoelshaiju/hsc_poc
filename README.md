# Holistic Semantic Continuum (HSC) 🚀

A modular system for continuous semantic interpolation between programs through vector embeddings, manifold learning, and geodesic interpolation.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 🌟 Overview

The Holistic Semantic Continuum (HSC) represents a paradigm shift in how we think about program semantics. Rather than treating code as discrete text or binary, HSC embeds programs into a continuous semantic manifold where:

- **Programs are points** in a high-dimensional semantic space
- **Semantic relationships** are preserved through geometric structure
- **Continuous interpolation** enables smooth transitions between programs
- **Algorithmic morphing** discovers hybrid methods and novel solutions

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
$ git clone https://github.com/yourusername/hsc_poc.git
$ cd hsc_poc

# Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
$ pip install -r requirements.txt
```

### Run Demos

```bash
# Run all demos (recommended)
$ python run_demos.py all

# Run a specific demo
$ python run_demos.py basic
$ python run_demos.py focused
$ python run_demos.py error-recovery
$ python run_demos.py hierarchical
$ python run_demos.py real-programs

# Run with verbose output
$ python run_demos.py focused --verbose
```

### Run Tests

```bash
# Run all tests (requires pytest)
$ pytest tests/

# Run a specific test module
$ python -m tests.test_error_recovery
$ python -m tests.test_hierarchical
$ python -m tests.test_real_programs
```

---

## 📁 Project Structure

```
hsc_poc/
├── run_demos.py                 # Demo runner script
├── main.py                      # Main application entry point
├── requirements.txt             # Python dependencies
├── README.md                    # This file
│
├── demos/                       # Demonstration scripts
│   ├── __init__.py
│   ├── basic_demo.py
│   ├── focused_demo.py
│   └── large_codebase_demo.py
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_embedding.py
│   ├── test_interpolation.py
│   ├── test_manifold.py
│   ├── test_error_recovery.py
│   ├── test_hierarchical.py
│   └── test_real_programs.py
│
├── error_handling/              # Error handling system
│   ├── __init__.py
│   ├── recovery.py
│   └── analysis.py
│
├── embedding/                   # Core embedding system
│   ├── __init__.py
│   ├── encoder.py
│   ├── decoder.py
│   ├── engine.py
│   ├── hierarchical.py
│   └── utils.py
│
├── dsl/                         # Domain-specific language
│   ├── __init__.py
│   ├── parser.py
│   ├── grammar.lark
│   ├── interpreter.py
│   └── examples.py
│
├── manifold/                    # Manifold learning
│   ├── __init__.py
│   └── autoencoder.py
│
├── interpolation/               # Interpolation system
│   ├── __init__.py
│   └── geodesic.py
│
├── models/                      # Model management
│   ├── __init__.py
│   └── versioning.py
│
├── evaluation/                  # Evaluation and metrics
│   ├── __init__.py
│   ├── metrics.py
│   ├── similarity.py
│   └── benchmarks.py
│
├── visualization/               # Visualization tools
│   ├── __init__.py
│   └── error_analysis.py
│
├── docs/                        # Documentation
│   ├── README.md
│   ├── ARCHITECTURE.md
│   └── TECHNICAL_SUMMARY.md
```

---

## 🧩 Key Features

- **Semantic Manifold Embedding**: Programs, data, and specifications are mapped into a smooth manifold via a functorial embedding process, preserving compositional structure.
- **Hierarchical & Chunked Embeddings**: Efficiently process large codebases with function-level and windowed chunking, optimized for low memory usage.
- **Continuous Algorithmic Morphing**: Interpolate between programs to discover hybrid or novel algorithms.
- **Topology-Aware Optimization**: Analyze geodesics and curvature on the semantic manifold for optimal algorithm discovery.
- **Comprehensive Error Recovery**: Advanced error handling, feedback, and visualization for robust decoding and debugging.
- **Versioning & Snapshots**: Persistent snapshots for embeddings, manifold states, and interpolation paths.
- **Visualization**: 2D/3D embedding plots, error analysis, and interactive tools.

---

## 🛠️ Error Handling & Recovery

- **Error Recovery System**: All decoding and embedding errors are handled by a dedicated error handling package (`error_handling/`).
- **Detailed Feedback**: Error types, severity, suggestions, and recovery strategies are provided for every failure.
- **Visualization**: Error patterns, vector analysis, and recovery recommendations are visualized and saved automatically.
- **Tested**: See `tests/test_error_recovery.py` for comprehensive error handling tests and visualizations.

---

## 🧪 Testing

- All features are covered by a comprehensive test suite in `tests/`.
- Run all tests with `pytest tests/` or run individual modules as shown above.
- Tests include: embedding, manifold learning, interpolation, error recovery, hierarchical embedding, and real program validation.

---

## 📊 Demos

- All major features are demonstrated in the `demos/` package.
- Use `run_demos.py` to run all or individual demos.
- Demos include: basic system, memory-focused, large codebase, error recovery, hierarchical embedding, and real program validation.

---

## 📚 Documentation

- [Architecture Documentation](ARCHITECTURE.md)
- [Organization Guide](ORGANIZATION.md)
- [Technical Summary](TECHNICAL_SUMMARY.md)
- [API Reference](docs/README.md)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ by the HSC Research Team** 