"""
Vector-to-AST Decoder with Error Recovery and Explainable Transitions

This module provides functionality to convert vector representations back into
Abstract Syntax Trees (ASTs), enabling semantic interpolation and program synthesis.
Includes error recovery feedback and explainable transition annotations.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class DecodeErrorType(Enum):
    """Types of decoding errors."""
    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_ERROR = "semantic_error"
    TYPE_MISMATCH = "type_mismatch"
    STRUCTURE_INVALID = "structure_invalid"
    VECTOR_OUT_OF_BOUNDS = "vector_out_of_bounds"

@dataclass
class DecodeError:
    """Detailed error information for decoding failures."""
    error_type: DecodeErrorType
    message: str
    vector_slice: Optional[np.ndarray] = None
    expected_structure: Optional[str] = None
    actual_structure: Optional[str] = None
    confidence: float = 0.0
    suggestions: Optional[List[str]] = None
    
    def __init__(self, error_type: DecodeErrorType, message: str, 
                 vector_slice: Optional[np.ndarray] = None, 
                 expected_structure: Optional[str] = None, 
                 actual_structure: Optional[str] = None, 
                 confidence: float = 0.0, 
                 suggestions: Optional[List[str]] = None):
        self.error_type = error_type
        self.message = message
        self.vector_slice = vector_slice
        self.expected_structure = expected_structure
        self.actual_structure = actual_structure
        self.confidence = confidence
        self.suggestions = suggestions if suggestions is not None else []

@dataclass
class TransitionAnnotation:
    """Annotation explaining what changed during interpolation."""
    step: int
    structural_changes: List[str]
    semantic_changes: List[str]
    type_changes: List[str]
    confidence: float
    explanation: str

@dataclass
class DecodeResult:
    """Result of decoding operation with metadata."""
    ast: Any
    success: bool
    errors: List[DecodeError]
    confidence: float
    transition_annotations: List[TransitionAnnotation]
    metadata: Dict[str, Any]

class ErrorRecoveryFeedback:
    """Provides comprehensive feedback and suggestions for decoding errors with visualization."""
    
    def __init__(self):
        self.error_patterns = {
            DecodeErrorType.SYNTAX_ERROR: [
                "Check for missing parentheses or brackets",
                "Verify operator precedence",
                "Ensure proper indentation",
                "Check for unmatched quotes",
                "Validate function call syntax",
                "Check for missing colons after control structures"
            ],
            DecodeErrorType.SEMANTIC_ERROR: [
                "Variable may be undefined or out of scope",
                "Function call with wrong number of arguments",
                "Type mismatch in operation",
                "Invalid control flow structure",
                "Missing return statement in function",
                "Infinite loop detected"
            ],
            DecodeErrorType.TYPE_MISMATCH: [
                "Expected numeric type, got string",
                "Boolean expression in numeric context",
                "Function call with wrong parameter types",
                "Assignment type mismatch",
                "Operator not supported for given types",
                "Type conversion needed"
            ],
            DecodeErrorType.STRUCTURE_INVALID: [
                "AST structure is malformed",
                "Missing required child nodes",
                "Invalid node type sequence",
                "Unbalanced tree structure",
                "Unexpected node type",
                "Incomplete expression"
            ],
            DecodeErrorType.VECTOR_OUT_OF_BOUNDS: [
                "Vector contains invalid values (NaN, inf)",
                "Vector dimension mismatch",
                "Vector normalization failed",
                "Vector contains extreme values",
                "Vector encoding corrupted"
            ]
        }
        
        # Error severity levels
        self.severity_levels = {
            DecodeErrorType.SYNTAX_ERROR: "HIGH",
            DecodeErrorType.SEMANTIC_ERROR: "MEDIUM",
            DecodeErrorType.TYPE_MISMATCH: "MEDIUM",
            DecodeErrorType.STRUCTURE_INVALID: "HIGH",
            DecodeErrorType.VECTOR_OUT_OF_BOUNDS: "CRITICAL"
        }
        
        # Recovery strategies
        self.recovery_strategies = {
            DecodeErrorType.SYNTAX_ERROR: [
                "Attempt automatic syntax correction",
                "Use template-based reconstruction",
                "Fall back to nearest valid program"
            ],
            DecodeErrorType.SEMANTIC_ERROR: [
                "Analyze variable scope and definitions",
                "Check function signatures",
                "Validate control flow logic"
            ],
            DecodeErrorType.TYPE_MISMATCH: [
                "Insert type conversion operators",
                "Modify operation to match types",
                "Use type-safe alternatives"
            ],
            DecodeErrorType.STRUCTURE_INVALID: [
                "Rebuild AST from scratch",
                "Use structural templates",
                "Apply structural constraints"
            ],
            DecodeErrorType.VECTOR_OUT_OF_BOUNDS: [
                "Re-encode the vector",
                "Use vector normalization",
                "Apply vector constraints"
            ]
        }
    
    def analyze_error(self, error: DecodeError, vector: np.ndarray) -> List[str]:
        """Analyze error and provide specific suggestions with detailed feedback."""
        suggestions = self.error_patterns.get(error.error_type, [])
        
        # Add vector-specific suggestions
        if error.vector_slice is not None:
            vector_analysis = self._analyze_vector_slice(error.vector_slice)
            suggestions.extend(vector_analysis)
        
        # Add severity-based suggestions
        severity = self.severity_levels.get(error.error_type, "UNKNOWN")
        suggestions.append(f"Error severity: {severity}")
        
        # Add recovery strategies
        strategies = self.recovery_strategies.get(error.error_type, [])
        suggestions.extend([f"Recovery strategy: {strategy}" for strategy in strategies])
        
        # Add confidence-based feedback
        if error.confidence < 0.3:
            suggestions.append("Low confidence - consider re-encoding or using different approach")
        elif error.confidence < 0.7:
            suggestions.append("Medium confidence - partial recovery possible")
        else:
            suggestions.append("High confidence - error should be recoverable")
        
        return suggestions
    
    def _analyze_vector_slice(self, vector_slice: np.ndarray) -> List[str]:
        """Analyze vector slice to provide specific suggestions."""
        suggestions = []
        
        # Check for extreme values
        if np.any(vector_slice > 0.9):
            suggestions.append("Vector contains extreme values (>0.9) - consider normalization")
        if np.any(vector_slice < 0.1):
            suggestions.append("Vector contains very small values (<0.1) - may indicate sparse encoding")
        
        # Check for uniform distribution
        if np.std(vector_slice) < 0.1:
            suggestions.append("Vector is too uniform (std < 0.1) - may lack semantic information")
        
        # Check for NaN or inf values
        if np.any(np.isnan(vector_slice)):
            suggestions.append("Vector contains NaN values - encoding may have failed")
        if np.any(np.isinf(vector_slice)):
            suggestions.append("Vector contains infinite values - numerical overflow detected")
        
        # Check vector norm
        norm = np.linalg.norm(vector_slice)
        if norm < 0.1:
            suggestions.append("Vector norm too small (<0.1) - may be under-encoded")
        elif norm > 10.0:
            suggestions.append("Vector norm too large (>10.0) - may be over-encoded")
        
        return suggestions
    
    def generate_error_report(self, errors: List[DecodeError], vector: np.ndarray) -> Dict[str, Any]:
        """Generate a comprehensive error report with analysis and recommendations."""
        report = {
            'total_errors': len(errors),
            'error_types': {},
            'severity_distribution': {},
            'overall_confidence': 0.0,
            'recovery_recommendations': [],
            'vector_analysis': self._analyze_vector_comprehensive(vector),
            'detailed_errors': []
        }
        
        # Analyze each error
        for i, error in enumerate(errors):
            error_analysis = {
                'index': i,
                'type': error.error_type.value,
                'message': error.message,
                'confidence': error.confidence,
                'severity': self.severity_levels.get(error.error_type, "UNKNOWN"),
                'suggestions': self.analyze_error(error, vector),
                'vector_slice_info': self._analyze_vector_slice_info(error.vector_slice) if error.vector_slice is not None else None
            }
            report['detailed_errors'].append(error_analysis)
            
            # Update statistics
            error_type = error.error_type.value
            report['error_types'][error_type] = report['error_types'].get(error_type, 0) + 1
            
            severity = self.severity_levels.get(error.error_type, "UNKNOWN")
            report['severity_distribution'][severity] = report['severity_distribution'].get(severity, 0) + 1
        
        # Calculate overall confidence
        if errors:
            report['overall_confidence'] = np.mean([e.confidence for e in errors])
        
        # Generate recovery recommendations
        report['recovery_recommendations'] = self._generate_recovery_recommendations(errors)
        
        return report
    
    def _analyze_vector_comprehensive(self, vector: np.ndarray) -> Dict[str, Any]:
        """Comprehensive vector analysis for debugging."""
        try:
            # Handle inf and NaN values properly
            finite_vector = vector[np.isfinite(vector)]
            
            analysis = {
                'dimension': len(vector),
                'norm': float(np.linalg.norm(vector)) if np.isfinite(np.linalg.norm(vector)) else float('inf'),
                'mean': float(np.mean(finite_vector)) if len(finite_vector) > 0 else float('nan'),
                'std': float(np.std(finite_vector)) if len(finite_vector) > 1 else 0.0,
                'min': float(np.min(finite_vector)) if len(finite_vector) > 0 else float('nan'),
                'max': float(np.max(finite_vector)) if len(finite_vector) > 0 else float('nan'),
                'has_nan': bool(np.any(np.isnan(vector))),
                'has_inf': bool(np.any(np.isinf(vector))),
                'zero_count': int(np.sum(vector == 0)),
                'extreme_count': int(np.sum((vector > 0.9) | (vector < 0.1))),
                'distribution_type': self._classify_distribution(finite_vector) if len(finite_vector) > 0 else "INVALID"
            }
            
            return analysis
            
        except Exception as e:
            # Fallback analysis if something goes wrong
            return {
                'dimension': len(vector),
                'norm': float('nan'),
                'mean': float('nan'),
                'std': float('nan'),
                'min': float('nan'),
                'max': float('nan'),
                'has_nan': bool(np.any(np.isnan(vector))),
                'has_inf': bool(np.any(np.isinf(vector))),
                'zero_count': 0,
                'extreme_count': 0,
                'distribution_type': "ERROR"
            }
    
    def _classify_distribution(self, vector: np.ndarray) -> str:
        """Classify the distribution of vector values."""
        try:
            if len(vector) == 0:
                return "EMPTY"
            
            # Check for invalid values
            if not np.all(np.isfinite(vector)):
                return "INVALID"
            
            std_val = np.std(vector)
            if np.isnan(std_val) or np.isinf(std_val):
                return "INVALID"
            
            if std_val < 0.1:
                return "UNIFORM"
            elif std_val > 0.5:
                return "VARIED"
            else:
                return "MODERATE"
                
        except Exception:
            return "ERROR"
    
    def _analyze_vector_slice_info(self, vector_slice: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Analyze specific vector slice information."""
        if vector_slice is None:
            return None
        
        try:
            # Handle inf and NaN values properly
            finite_slice = vector_slice[np.isfinite(vector_slice)]
            
            return {
                'dimension': len(vector_slice),
                'norm': float(np.linalg.norm(vector_slice)) if np.isfinite(np.linalg.norm(vector_slice)) else float('inf'),
                'mean': float(np.mean(finite_slice)) if len(finite_slice) > 0 else float('nan'),
                'std': float(np.std(finite_slice)) if len(finite_slice) > 1 else 0.0,
                'range': [float(np.min(finite_slice)), float(np.max(finite_slice))] if len(finite_slice) > 0 else [float('nan'), float('nan')]
            }
            
        except Exception:
            return {
                'dimension': len(vector_slice),
                'norm': float('nan'),
                'mean': float('nan'),
                'std': float('nan'),
                'range': [float('nan'), float('nan')]
            }
    
    def _generate_recovery_recommendations(self, errors: List[DecodeError]) -> List[str]:
        """Generate prioritized recovery recommendations."""
        recommendations = []
        
        # Count error types
        error_counts = {}
        for error in errors:
            error_type = error.error_type
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        # Prioritize by severity and frequency
        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            strategies = self.recovery_strategies.get(error_type, [])
            
            if count > 1:
                recommendations.append(f"Multiple {error_type.value} errors ({count}) - {strategies[0] if strategies else 'General recovery needed'}")
            else:
                recommendations.append(f"Single {error_type.value} error - {strategies[0] if strategies else 'Standard recovery'}")
        
        # Add general recommendations
        if len(errors) > 5:
            recommendations.append("High error count - consider re-encoding with different parameters")
        
        if any(e.confidence < 0.3 for e in errors):
            recommendations.append("Low confidence errors detected - use fallback decoding strategy")
        
        return recommendations
    
    def visualize_error_patterns(self, errors: List[DecodeError]) -> Dict[str, Any]:
        """Generate visualization data for error patterns."""
        if not errors:
            return {}
        
        # Error type distribution
        error_types = [e.error_type.value for e in errors]
        type_counts = {}
        for error_type in error_types:
            type_counts[error_type] = type_counts.get(error_type, 0) + 1
        
        # Confidence distribution
        confidences = [e.confidence for e in errors]
        
        # Severity distribution
        severities = [self.severity_levels.get(e.error_type, "UNKNOWN") for e in errors]
        severity_counts = {}
        for severity in severities:
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            'error_type_distribution': type_counts,
            'confidence_distribution': {
                'mean': float(np.mean(confidences)),
                'std': float(np.std(confidences)),
                'min': float(np.min(confidences)),
                'max': float(np.max(confidences))
            },
            'severity_distribution': severity_counts,
            'total_errors': len(errors)
        }

def decode_vec_to_ast(vector: np.ndarray, target_structure: Optional[Any] = None,
                      config: Optional[Dict[str, Any]] = None) -> DecodeResult:
    """
    Decode a vector back into an AST structure with comprehensive error recovery.
    
    Args:
        vector: Vector representation to decode
        target_structure: Optional target AST structure to guide decoding
        config: Decoding configuration
        
    Returns:
        DecodeResult with AST, success status, and detailed error information
    """
    if config is None:
        config = {}
    
    errors = []
    confidence = 1.0
    transition_annotations = []
    
    # Initialize error recovery system
    error_feedback = ErrorRecoveryFeedback()
    
    # Validate input vector
    if not _validate_vector(vector):
        errors.append(DecodeError(
            error_type=DecodeErrorType.VECTOR_OUT_OF_BOUNDS,
            message="Vector contains invalid values",
            vector_slice=vector,
            confidence=0.0
        ))
        confidence *= 0.5
    
    try:
        if target_structure is not None:
            ast, step_annotations = _guided_decode(vector, target_structure, config)
            transition_annotations.extend(step_annotations)
        else:
            ast, step_annotations = _free_decode(vector, config)
            transition_annotations.extend(step_annotations)
        
        # Validate generated AST
        ast_errors = _validate_ast(ast)
        errors.extend(ast_errors)
        
        # Adjust confidence based on errors
        for error in ast_errors:
            confidence *= (1.0 - error.confidence)
        
        success = len([e for e in errors if e.error_type in [DecodeErrorType.SYNTAX_ERROR, DecodeErrorType.STRUCTURE_INVALID]]) == 0
        
    except Exception as e:
        errors.append(DecodeError(
            error_type=DecodeErrorType.STRUCTURE_INVALID,
            message=f"Decoding failed: {str(e)}",
            confidence=0.0
        ))
        ast = None
        success = False
        confidence = 0.0
    
    # Generate comprehensive error report
    error_report = error_feedback.generate_error_report(errors, vector)
    error_patterns = error_feedback.visualize_error_patterns(errors)
    
    return DecodeResult(
        ast=ast,
        success=success,
        errors=errors,
        confidence=confidence,
        transition_annotations=transition_annotations,
        metadata={
            'vector_norm': float(np.linalg.norm(vector)) if np.isfinite(np.linalg.norm(vector)) else float('inf'),
            'vector_mean': float(np.mean(vector)) if np.isfinite(np.mean(vector)) else float('nan'),
            'vector_std': float(np.std(vector)) if np.isfinite(np.std(vector)) else float('nan'),
            'error_report': error_report,
            'error_patterns': error_patterns,
            'recovery_suggestions': error_report.get('recovery_recommendations', [])
        }
    )

def _validate_vector(vector: np.ndarray) -> bool:
    """Validate that vector contains reasonable values."""
    try:
        # Check for basic validity
        if vector is None or not isinstance(vector, np.ndarray):
            return False
        
        # Check for finite values (no NaN or inf)
        if not np.all(np.isfinite(vector)):
            return False
        
        # Check for reasonable bounds (allow negative values for some encodings)
        # Don't enforce 0-1 bounds as vectors can be normalized differently
        if np.any(np.abs(vector) > 1e6):  # Check for extreme values
            return False
        
        # Check for reasonable size
        if len(vector) == 0:
            return False
        
        return True
        
    except Exception:
        return False

def _guided_decode(vector: np.ndarray, target_structure: Any, 
                   config: Dict[str, Any]) -> Tuple[Any, List[TransitionAnnotation]]:
    """
    Decode vector using target structure as a guide.
    
    Args:
        vector: Vector representation
        target_structure: Target AST structure to match
        config: Decoding configuration
        
    Returns:
        Tuple of (reconstructed AST, transition annotations)
    """
    annotations = []
    
    if isinstance(target_structure, tuple):
        # Reconstruct tuple structure
        node_type = target_structure[0]
        children = []
        
        # Use vector values to influence child reconstruction
        for i, child in enumerate(target_structure[1:]):
            child_vector = vector[i * 16:(i + 1) * 16]  # Use portion of vector for each child
            reconstructed_child, child_annotations = _guided_decode(child_vector, child, config)
            children.append(reconstructed_child)
            annotations.extend(child_annotations)
        
        result = (node_type,) + tuple(children)
        
        # Add annotation for this step
        annotations.append(TransitionAnnotation(
            step=len(annotations),
            structural_changes=[f"Reconstructed {node_type} node with {len(children)} children"],
            semantic_changes=[f"Maintained {node_type} semantic structure"],
            type_changes=[f"Preserved {node_type} type information"],
            confidence=0.8,
            explanation=f"Guided reconstruction of {node_type} node"
        ))
        
        return result, annotations
    
    elif isinstance(target_structure, list):
        # Reconstruct list structure
        children = []
        for i, child in enumerate(target_structure):
            child_vector = vector[i * 16:(i + 1) * 16]
            reconstructed_child, child_annotations = _guided_decode(child_vector, child, config)
            children.append(reconstructed_child)
            annotations.extend(child_annotations)
        
        annotations.append(TransitionAnnotation(
            step=len(annotations),
            structural_changes=[f"Reconstructed list with {len(children)} elements"],
            semantic_changes=["Maintained list semantic structure"],
            type_changes=["Preserved list type information"],
            confidence=0.8,
            explanation="Guided reconstruction of list structure"
        ))
        
        return children, annotations
    
    else:
        # Leaf node - return as is
        return target_structure, annotations

def _free_decode(vector: np.ndarray, config: Dict[str, Any]) -> Tuple[Any, List[TransitionAnnotation]]:
    """
    Free-form decoding without target structure.
    
    Args:
        vector: Vector representation
        config: Decoding configuration
        
    Returns:
        Tuple of (reconstructed AST, transition annotations)
    """
    annotations = []
    
    # This is a simplified decoder that creates a basic structure
    # In practice, this would be much more sophisticated
    
    # Use vector values to determine structure
    if len(vector) < 16:
        return "atom", annotations
    
    # Simple heuristic: use first few values to determine node type
    node_type_score = np.mean(vector[:8])
    
    if node_type_score > 0.7:
        # Function definition
        result = ("func_def", "main", [], ("return_stmt", ("atom", "0")))
        annotations.append(TransitionAnnotation(
            step=0,
            structural_changes=["Created function definition structure"],
            semantic_changes=["Inferred function semantic context"],
            type_changes=["Assigned function type"],
            confidence=0.7,
            explanation="High vector values suggest function definition"
        ))
    elif node_type_score > 0.5:
        # If statement
        result = ("if_stmt", ("atom", "condition"), ("assignment", "x", ("atom", "1")))
        annotations.append(TransitionAnnotation(
            step=0,
            structural_changes=["Created conditional structure"],
            semantic_changes=["Inferred control flow semantic"],
            type_changes=["Assigned boolean condition type"],
            confidence=0.6,
            explanation="Medium vector values suggest conditional logic"
        ))
    elif node_type_score > 0.3:
        # Assignment
        result = ("assignment", "x", ("expr_binop", "+", ("atom", "a"), ("atom", "b")))
        annotations.append(TransitionAnnotation(
            step=0,
            structural_changes=["Created assignment structure"],
            semantic_changes=["Inferred data flow semantic"],
            type_changes=["Assigned variable assignment type"],
            confidence=0.5,
            explanation="Low-medium vector values suggest assignment"
        ))
    else:
        # Simple atom
        result = ("atom", "value")
        annotations.append(TransitionAnnotation(
            step=0,
            structural_changes=["Created simple atom structure"],
            semantic_changes=["Inferred literal semantic"],
            type_changes=["Assigned atom type"],
            confidence=0.4,
            explanation="Low vector values suggest simple literal"
        ))
    
    return result, annotations

def _validate_ast(ast: Any) -> List[DecodeError]:
    """Validate that an AST has a reasonable structure."""
    errors = []
    
    try:
        # Basic validation: check if it's a reasonable structure
        if isinstance(ast, tuple):
            if len(ast) == 0:
                errors.append(DecodeError(
                    error_type=DecodeErrorType.STRUCTURE_INVALID,
                    message="Empty tuple structure",
                    confidence=0.9
                ))
                return errors
            
            # Check if first element is a string (node type)
            if not isinstance(ast[0], str):
                errors.append(DecodeError(
                    error_type=DecodeErrorType.STRUCTURE_INVALID,
                    message="First element must be string node type",
                    confidence=0.8
                ))
            
            # Recursively validate children
            for i, child in enumerate(ast[1:]):
                child_errors = _validate_ast(child)
                for error in child_errors:
                    error.message = f"Child {i}: {error.message}"
                errors.extend(child_errors)
        
        elif isinstance(ast, list):
            # Validate all list elements
            for i, item in enumerate(ast):
                item_errors = _validate_ast(item)
                for error in item_errors:
                    error.message = f"List item {i}: {error.message}"
                errors.extend(item_errors)
        
        else:
            # Leaf nodes are always valid
            pass
    
    except Exception as e:
        errors.append(DecodeError(
            error_type=DecodeErrorType.STRUCTURE_INVALID,
            message=f"Validation failed: {str(e)}",
            confidence=0.9
        ))
    
    return errors

def decode_to_program(vector: np.ndarray, target_structure: Optional[Any] = None,
                     config: Optional[Dict[str, Any]] = None) -> Tuple[str, DecodeResult]:
    """
    Decode vector directly to program code with error recovery.
    
    Args:
        vector: Vector representation
        target_structure: Optional target structure
        config: Decoding configuration
        
    Returns:
        Tuple of (program code, decode result)
    """
    decode_result = decode_vec_to_ast(vector, target_structure, config)
    
    if decode_result.success and decode_result.ast is not None:
        try:
            program_code = _ast_to_code(decode_result.ast)
            return program_code, decode_result
        except Exception as e:
            decode_result.errors.append(DecodeError(
                error_type=DecodeErrorType.SYNTAX_ERROR,
                message=f"Code generation failed: {str(e)}",
                confidence=0.7
            ))
            decode_result.success = False
            return "", decode_result
    else:
        return "", decode_result

def _ast_to_code(ast: Any) -> str:
    """
    Convert AST back to program code.
    
    Args:
        ast: Abstract Syntax Tree
        
    Returns:
        Program code as string
    """
    if isinstance(ast, tuple):
        node_type = ast[0]
        
        if node_type == "func_def":
            name = ast[1] if len(ast) > 1 else "main"
            params = ast[2] if len(ast) > 2 else []
            body = ast[3] if len(ast) > 3 else ("pass",)
            
            param_str = ", ".join(params) if params else ""
            body_str = _ast_to_code(body)
            return f"def {name}({param_str}):\n    {body_str}"
        
        elif node_type == "if_stmt":
            condition = _ast_to_code(ast[1]) if len(ast) > 1 else "True"
            body = _ast_to_code(ast[2]) if len(ast) > 2 else "pass"
            return f"if {condition}:\n    {body}"
        
        elif node_type == "while_stmt":
            condition = _ast_to_code(ast[1]) if len(ast) > 1 else "True"
            body = _ast_to_code(ast[2]) if len(ast) > 2 else "pass"
            return f"while {condition}:\n    {body}"
        
        elif node_type == "assignment":
            var = ast[1] if len(ast) > 1 else "x"
            value = _ast_to_code(ast[2]) if len(ast) > 2 else "0"
            return f"{var} = {value}"
        
        elif node_type == "return_stmt":
            value = _ast_to_code(ast[1]) if len(ast) > 1 else "None"
            return f"return {value}"
        
        elif node_type == "func_call":
            func_name = ast[1] if len(ast) > 1 else "func"
            args = ast[2] if len(ast) > 2 else []
            args_str = ", ".join(_ast_to_code(arg) for arg in args)
            return f"{func_name}({args_str})"
        
        elif node_type == "expr_binop":
            op = ast[1] if len(ast) > 1 else "+"
            left = _ast_to_code(ast[2]) if len(ast) > 2 else "0"
            right = _ast_to_code(ast[3]) if len(ast) > 3 else "0"
            return f"({left} {op} {right})"
        
        elif node_type == "atom":
            value = ast[1] if len(ast) > 1 else "0"
            return str(value)
        
        else:
            # Unknown node type, return as string
            return str(ast)
    
    elif isinstance(ast, list):
        return "\n".join(_ast_to_code(item) for item in ast)
    
    else:
        return str(ast)

def interpolate_asts(ast1: Any, ast2: Any, t: float) -> Tuple[Any, List[TransitionAnnotation]]:
    """
    Interpolate between two ASTs with explainable transitions.
    
    Args:
        ast1: First AST
        ast2: Second AST
        t: Interpolation parameter (0.0 to 1.0)
        
    Returns:
        Tuple of (interpolated AST, transition annotations)
    """
    annotations = []
    
    # This is a simplified interpolation
    # In practice, this would be much more sophisticated
    
    if t < 0.5:
        result = ast1
        annotations.append(TransitionAnnotation(
            step=0,
            structural_changes=["Maintained structure from first AST"],
            semantic_changes=["Preserved semantic from first AST"],
            type_changes=["Kept type information from first AST"],
            confidence=1.0 - t,
            explanation=f"Interpolation parameter t={t:.2f} favors first AST"
        ))
    else:
        result = ast2
        annotations.append(TransitionAnnotation(
            step=0,
            structural_changes=["Adopted structure from second AST"],
            semantic_changes=["Adopted semantic from second AST"],
            type_changes=["Adopted type information from second AST"],
            confidence=t,
            explanation=f"Interpolation parameter t={t:.2f} favors second AST"
        ))
    
    return result, annotations

def validate_ast(ast: Any) -> bool:
    """
    Validate that an AST has a reasonable structure.
    
    Args:
        ast: AST to validate
        
    Returns:
        True if AST is valid, False otherwise
    """
    errors = _validate_ast(ast)
    return len(errors) == 0

def explain_transition(ast1: Any, ast2: Any) -> List[str]:
    """
    Explain the transition between two ASTs.
    
    Args:
        ast1: First AST
        ast2: Second AST
        
    Returns:
        List of explanation strings
    """
    explanations = []
    
    # Compare structures
    if isinstance(ast1, tuple) and isinstance(ast2, tuple):
        if ast1[0] != ast2[0]:
            explanations.append(f"Node type changed from {ast1[0]} to {ast2[0]}")
        if len(ast1) != len(ast2):
            explanations.append(f"Number of children changed from {len(ast1)} to {len(ast2)}")
    
    # Compare content (for enhanced AST nodes)
    if hasattr(ast1, 'content') and hasattr(ast2, 'content'):
        if ast1.content != ast2.content:  # type: ignore
            explanations.append(f"Content changed from '{ast1.content}' to '{ast2.content}'")  # type: ignore
    
    # Compare types (for enhanced AST nodes)
    if hasattr(ast1, 'type_info') and hasattr(ast2, 'type_info'):
        if ast1.type_info.category != ast2.type_info.category:  # type: ignore
            explanations.append(f"Type changed from {ast1.type_info.category.value} to {ast2.type_info.category.value}")  # type: ignore
    
    return explanations 