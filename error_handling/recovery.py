"""
Error Recovery and Feedback System

This module provides comprehensive error recovery and feedback capabilities
for the HSC system, including error analysis, suggestions, and recovery strategies.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from enum import Enum

class DecodeErrorType(Enum):
    """Types of decoding errors."""
    SYNTAX_ERROR = "syntax_error"
    SEMANTIC_ERROR = "semantic_error"
    TYPE_MISMATCH = "type_mismatch"
    STRUCTURE_INVALID = "structure_invalid"
    VECTOR_OUT_OF_BOUNDS = "vector_out_of_bounds"

class DecodeError:
    """Detailed error information for decoding failures."""
    
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