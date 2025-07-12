"""
DSL-to-Vector Embedding Engine with Enhanced Features

This module provides the main engine for converting DSL programs into vector
representations with support for abstract type inference, neural embeddings,
error recovery, quality metrics, and versioning.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from .encoder import encode_ast, HybridEmbeddingConfig, NeuralEmbeddingProvider
from .decoder import decode_vec_to_ast, DecodeResult, TransitionAnnotation, ErrorRecoveryFeedback
from dsl.parser import parse_program, get_type_summary
from manifold.autoencoder import AutoencoderManifold, AutoencoderConfig
from models.versioning import VersioningSystem, EmbeddingSnapshot, ManifoldSnapshot

@dataclass
class EmbeddingConfig:
    """Configuration for the embedding engine."""
    vector_dim: int = 128
    normalize: bool = True
    use_semantic_weights: bool = True
    use_neural_embeddings: bool = False
    neural_model_name: str = "codebert"
    hash_weight: float = 0.3
    structural_weight: float = 0.4
    semantic_weight: float = 0.3
    context_window: int = 5
    enable_versioning: bool = True
    snapshot_auto_save: bool = True

@dataclass
class EmbeddingResult:
    """Result of embedding operation with metadata."""
    vector: np.ndarray
    ast: Any
    type_summary: Dict[str, Any]
    confidence: float
    errors: List[str]
    metadata: Dict[str, Any]

class DSLToVectorEngine:
    """
    Enhanced DSL-to-Vector embedding engine with comprehensive features.
    
    This engine provides:
    - Abstract type inference and type-aware parsing
    - Hybrid embeddings with neural model support
    - Error recovery and feedback
    - Manifold learning with quality metrics
    - Explainable transitions during interpolation
    - Versioning and snapshot capabilities
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        
        # Initialize hybrid embedding configuration
        self.hybrid_config = HybridEmbeddingConfig(
            vector_dim=self.config.vector_dim,
            use_neural_embeddings=self.config.use_neural_embeddings,
            neural_model_name=self.config.neural_model_name,
            hash_weight=self.config.hash_weight,
            structural_weight=self.config.structural_weight,
            semantic_weight=self.config.semantic_weight,
            context_window=self.config.context_window
        )
        
        # Initialize neural embedding provider
        self.neural_provider = NeuralEmbeddingProvider(self.config.neural_model_name)
        
        # Initialize error recovery system
        self.error_feedback = ErrorRecoveryFeedback()
        
        # Initialize manifold learning
        self.manifold = AutoencoderManifold(
            input_dim=self.config.vector_dim,
            latent_dim=max(2, self.config.vector_dim // 4),
            config=AutoencoderConfig(
                input_dim=self.config.vector_dim,
                latent_dim=max(2, self.config.vector_dim // 4),
                quality_metrics_k=15
            )
        )
        
        # Initialize versioning system
        self.versioning = VersioningSystem() if self.config.enable_versioning else None
        
        # Storage for embeddings and programs
        self.embeddings: Dict[str, np.ndarray] = {}
        self.programs: Dict[str, str] = {}
        self.asts: Dict[str, Any] = {}
        self.type_summaries: Dict[str, Dict[str, Any]] = {}
        
        # Interpolation history
        self.interpolation_history: List[Dict[str, Any]] = []
        
        print(f"Initialized DSL-to-Vector Engine with {self.config.vector_dim}D vectors")
        if self.config.use_neural_embeddings:
            print(f"Neural embeddings enabled: {self.config.neural_model_name}")
        if self.config.enable_versioning:
            print("Versioning system enabled")
    
    def encode_program(self, program_code: str, program_id: Optional[str] = None) -> EmbeddingResult:
        """
        Encode a DSL program into a vector representation with enhanced features.
        
        Args:
            program_code: The program code as a string
            program_id: Optional unique identifier for the program
            
        Returns:
            EmbeddingResult with vector, AST, and metadata
        """
        if program_id is None:
            program_id = self._generate_program_id(program_code)
        
        errors = []
        confidence = 1.0
        
        try:
            # 1. Parse program with type inference
            print(f"Parsing program {program_id}...")
            ast_nodes = parse_program(program_code)
            
            # 2. Generate type summary
            type_summary = get_type_summary(ast_nodes)
            
            # 3. Encode AST to vector
            print(f"Encoding AST to {self.config.vector_dim}D vector...")
            vector = encode_ast(ast_nodes, self.config.vector_dim, self.hybrid_config)
            
            # 4. Normalize if requested
            if self.config.normalize:
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
            
            # 5. Store results
            self.embeddings[program_id] = vector
            self.programs[program_id] = program_code
            self.asts[program_id] = ast_nodes
            self.type_summaries[program_id] = type_summary
            
            # 6. Create metadata
            metadata = {
                'program_id': program_id,
                'vector_norm': float(np.linalg.norm(vector)),
                'type_coverage': type_summary.get('type_coverage', 0.0),
                'type_confidence': type_summary.get('type_confidence', 0.0),
                'ast_node_count': len(ast_nodes),
                'encoding_config': self.hybrid_config.__dict__
            }
            
            # 7. Auto-save snapshot if enabled
            if self.config.enable_versioning and self.config.snapshot_auto_save:
                self._create_auto_snapshot(program_id, "Auto-save after encoding")
            
            print(f"Successfully encoded program {program_id}")
            
        except Exception as e:
            errors.append(f"Encoding failed: {str(e)}")
            confidence = 0.0
            vector = np.zeros(self.config.vector_dim)
            ast_nodes = []
            type_summary = {}
            metadata = {'error': str(e)}
        
        return EmbeddingResult(
            vector=vector,
            ast=ast_nodes,
            type_summary=type_summary,
            confidence=confidence,
            errors=errors,
            metadata=metadata
        )
    
    def decode_program(self, vector: np.ndarray, target_structure: Optional[Any] = None) -> DecodeResult:
        """
        Decode a vector back to a program with comprehensive error recovery and analysis.
        
        Args:
            vector: Vector representation to decode
            target_structure: Optional target structure for guided decoding
            
        Returns:
            DecodeResult with AST and detailed error information
        """
        print(f"Decoding {len(vector)}D vector to program...")
        
        # Validate vector
        if len(vector) != self.config.vector_dim:
            raise ValueError(f"Vector dimension {len(vector)} doesn't match config {self.config.vector_dim}")
        
        # Decode with comprehensive error recovery
        decode_result = decode_vec_to_ast(vector, target_structure, {
            'vector_dim': self.config.vector_dim,
            'normalize': self.config.normalize
        })
        
        # Add suggestions for errors
        for error in decode_result.errors:
            suggestions = self.error_feedback.analyze_error(error, vector)
            if error.suggestions is None:
                error.suggestions = []
            error.suggestions.extend(suggestions)
        
        # Print error summary if there are errors
        if decode_result.errors:
            print(f"⚠️ Decoding completed with {len(decode_result.errors)} errors")
            self._print_error_summary(decode_result)
        else:
            print("✅ Decoding completed successfully")
        
        return decode_result
    
    def _print_error_summary(self, decode_result: DecodeResult):
        """Print a concise error summary."""
        error_report = decode_result.metadata.get('error_report', {})
        
        print(f"   Overall confidence: {error_report.get('overall_confidence', 0):.3f}")
        print(f"   Error types: {list(error_report.get('error_types', {}).keys())}")
        print(f"   Severity levels: {list(error_report.get('severity_distribution', {}).keys())}")
        
        # Show top recovery recommendations
        recommendations = error_report.get('recovery_recommendations', [])
        if recommendations:
            print(f"   Top recommendation: {recommendations[0]}")
    
    def analyze_decoding_errors(self, vector: np.ndarray, save_analysis: bool = False) -> Dict[str, Any]:
        """
        Perform comprehensive error analysis on a vector.
        
        Args:
            vector: Vector to analyze
            save_analysis: Whether to save analysis plots
            
        Returns:
            Comprehensive error analysis report
        """
        print(f"🔍 Performing comprehensive error analysis on {len(vector)}D vector...")
        
        # Decode with full error tracking
        decode_result = self.decode_program(vector)
        
        # Create error analysis
        analysis = {
            'decode_result': decode_result,
            'error_report': decode_result.metadata.get('error_report', {}),
            'error_patterns': decode_result.metadata.get('error_patterns', {}),
            'vector_analysis': decode_result.metadata.get('vector_analysis', {}),
            'recovery_suggestions': decode_result.metadata.get('recovery_suggestions', [])
        }
        
        # Generate visualizations if requested
        if save_analysis and decode_result.errors:
            try:
                from visualization.error_analysis import ErrorAnalyzer
                analyzer = ErrorAnalyzer()
                saved_files = analyzer.create_error_report(decode_result, "error_analysis")
                analysis['visualization_files'] = saved_files
                print("📊 Error analysis plots saved to error_analysis/ directory")
            except ImportError:
                print("⚠️ Visualization module not available - skipping plots")
        
        return analysis
    
    def compute_similarity(self, program_id1: str, program_id2: str) -> float:
        """
        Compute semantic similarity between two programs.
        
        Args:
            program_id1: ID of first program
            program_id2: ID of second program
            
        Returns:
            Similarity score between 0 and 1
        """
        if program_id1 not in self.embeddings or program_id2 not in self.embeddings:
            raise ValueError("One or both programs not found")
        
        vec1 = self.embeddings[program_id1]
        vec2 = self.embeddings[program_id2]
        
        # Cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(similarity)
    
    def find_similar_programs(self, program_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find programs similar to a given program.
        
        Args:
            program_id: ID of reference program
            top_k: Number of similar programs to return
            
        Returns:
            List of (program_id, similarity) tuples
        """
        if program_id not in self.embeddings:
            raise ValueError(f"Program {program_id} not found")
        
        similarities = []
        
        for pid, vector in self.embeddings.items():
            if pid != program_id:
                similarity = self.compute_similarity(program_id, pid)
                similarities.append((pid, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def interpolate_programs(self, program_id1: str, program_id2: str, 
                           steps: int = 10) -> List[Tuple[np.ndarray, DecodeResult, List[TransitionAnnotation]]]:
        """
        Interpolate between two programs with explainable transitions.
        
        Args:
            program_id1: ID of first program
            program_id2: ID of second program
            steps: Number of interpolation steps
            
        Returns:
            List of (vector, decode_result, transition_annotations) tuples
        """
        if program_id1 not in self.embeddings or program_id2 not in self.embeddings:
            raise ValueError("One or both programs not found")
        
        vec1 = self.embeddings[program_id1]
        vec2 = self.embeddings[program_id2]
        
        print(f"Interpolating between programs {program_id1} and {program_id2}...")
        
        results = []
        
        for i in range(steps + 1):
            t = i / steps
            
            # Linear interpolation in vector space
            interpolated_vector = (1 - t) * vec1 + t * vec2
            
            # Normalize if requested
            if self.config.normalize:
                norm = np.linalg.norm(interpolated_vector)
                if norm > 0:
                    interpolated_vector = interpolated_vector / norm
            
            # Decode interpolated vector
            decode_result = self.decode_program(interpolated_vector)
            
            # Generate transition annotations
            if i == 0:
                # First step - no transition
                annotations = []
            else:
                # Compare with previous step
                prev_ast = results[-1][1].ast
                curr_ast = decode_result.ast
                annotations = self._generate_transition_annotations(prev_ast, curr_ast, t)
            
            results.append((interpolated_vector, decode_result, annotations))
        
        # Store interpolation history
        interpolation_record = {
            'program_id1': program_id1,
            'program_id2': program_id2,
            'steps': steps,
            'timestamp': self._get_timestamp(),
            'results': [(r[0].tolist(), r[1].success, len(r[2])) for r in results]
        }
        self.interpolation_history.append(interpolation_record)
        
        print(f"Interpolation completed with {len(results)} steps")
        return results
    
    def learn_manifold(self, program_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Learn the manifold structure of program embeddings.
        
        Args:
            program_ids: List of program IDs to include (None for all)
            
        Returns:
            Manifold quality report
        """
        if program_ids is None:
            program_ids = list(self.embeddings.keys())
        
        if len(program_ids) < 2:
            raise ValueError("Need at least 2 programs to learn manifold")
        
        print(f"Learning manifold for {len(program_ids)} programs...")
        
        # Prepare embeddings
        embeddings_matrix = np.array([self.embeddings[pid] for pid in program_ids])
        
        # Fit manifold
        self.manifold.fit(embeddings_matrix)
        
        # Get quality report
        quality_report = self.manifold.get_quality_report()
        
        # Create manifold snapshot
        if self.config.enable_versioning and self.versioning is not None:
            manifold_snapshot = ManifoldSnapshot(
                manifold_config=self.manifold.config.__dict__,
                latent_embeddings=self.manifold.encode(embeddings_matrix),
                quality_metrics=quality_report,
                reconstruction_error=float(self.manifold.reconstruction_error_)
            )
            
            self.versioning.create_snapshot(
                manifold=manifold_snapshot,
                description=f"Manifold learning for {len(program_ids)} programs",
                tags=["manifold", "learning"]
            )
        
        print(f"Manifold learning completed. Overall quality: {quality_report.get('overall_quality', 0):.3f}")
        return quality_report
    
    def create_snapshot(self, description: str = "", tags: Optional[List[str]] = None) -> str:
        """
        Create a snapshot of the current state.
        
        Args:
            description: Human-readable description
            tags: List of tags for categorization
            
        Returns:
            Snapshot ID
        """
        if not self.config.enable_versioning or self.versioning is None:
            raise ValueError("Versioning is not enabled")
        
        # Prepare embedding snapshot
        if self.embeddings:
            embedding_snapshot = EmbeddingSnapshot(
                embeddings=np.array(list(self.embeddings.values())),
                program_ids=list(self.embeddings.keys()),
                config=self.config.__dict__,
                similarity_matrix=self._compute_similarity_matrix()
            )
        else:
            embedding_snapshot = None
        
        # Create snapshot
        snapshot_id = self.versioning.create_snapshot(
            embeddings=embedding_snapshot,
            description=description,
            tags=tags or []
        )
        
        print(f"Created snapshot {snapshot_id}")
        return snapshot_id
    
    def load_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        """
        Load a snapshot.
        
        Args:
            snapshot_id: ID of snapshot to load
            
        Returns:
            Loaded snapshot data
        """
        if not self.config.enable_versioning or self.versioning is None:
            raise ValueError("Versioning is not enabled")
        
        return self.versioning.load_snapshot(snapshot_id)
    
    def list_snapshots(self, **kwargs) -> List[Any]:
        """
        List available snapshots.
        
        Args:
            **kwargs: Filtering arguments
            
        Returns:
            List of snapshots
        """
        if not self.config.enable_versioning or self.versioning is None:
            raise ValueError("Versioning is not enabled")
        
        return self.versioning.list_snapshots(**kwargs)
    
    def _generate_program_id(self, program_code: str) -> str:
        """Generate a unique program ID."""
        import hashlib
        hash_value = hashlib.md5(program_code.encode()).hexdigest()[:8]
        return f"prog_{hash_value}"
    
    def _compute_similarity_matrix(self) -> np.ndarray:
        """Compute similarity matrix for all programs."""
        if len(self.embeddings) < 2:
            return np.array([])
        
        program_ids = list(self.embeddings.keys())
        n = len(program_ids)
        similarity_matrix = np.zeros((n, n))
        
        for i, pid1 in enumerate(program_ids):
            for j, pid2 in enumerate(program_ids):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    similarity_matrix[i, j] = self.compute_similarity(pid1, pid2)
        
        return similarity_matrix
    
    def _generate_transition_annotations(self, ast1: Any, ast2: Any, t: float) -> List[TransitionAnnotation]:
        """Generate annotations explaining transitions between ASTs."""
        annotations = []
        
        # Simple transition analysis
        if isinstance(ast1, list) and isinstance(ast2, list):
            if len(ast1) != len(ast2):
                annotations.append(TransitionAnnotation(
                    step=int(t * 100),
                    structural_changes=[f"AST node count changed from {len(ast1)} to {len(ast2)}"],
                    semantic_changes=["Program structure modified"],
                    type_changes=["Type information may have changed"],
                    confidence=0.8,
                    explanation=f"Structural change at interpolation step {t:.2f}"
                ))
        
        return annotations
    
    def _create_auto_snapshot(self, program_id: str, description: str):
        """Create an automatic snapshot."""
        try:
            self.create_snapshot(f"Auto-save: {description}", tags=["auto-save"])
        except Exception as e:
            print(f"Warning: Auto-snapshot failed: {e}")
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            'total_programs': len(self.programs),
            'total_embeddings': len(self.embeddings),
            'vector_dimension': self.config.vector_dim,
            'neural_embeddings_enabled': self.config.use_neural_embeddings,
            'versioning_enabled': self.config.enable_versioning,
            'manifold_fitted': self.manifold.is_fitted,
            'interpolation_history_count': len(self.interpolation_history),
            'snapshot_count': len(self.versioning.snapshots) if self.versioning else 0
        } 