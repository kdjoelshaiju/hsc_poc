"""
DSL Parser with Abstract Type System

This module provides a parser for the domain-specific language with
abstract type inference and type-aware semantic analysis.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
from dataclasses import dataclass
from enum import Enum

class TypeCategory(Enum):
    """Abstract type categories for type inference."""
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    STRING = "string"
    FUNCTION = "function"
    UNKNOWN = "unknown"
    ANY = "any"

@dataclass
class TypeInfo:
    """Type information for AST nodes."""
    category: TypeCategory
    is_literal: bool = False
    is_variable: bool = False
    is_function: bool = False
    inferred_from: Optional[str] = None
    confidence: float = 1.0

@dataclass
class ASTNode:
    """Enhanced AST node with type information."""
    node_type: str
    content: str
    type_info: TypeInfo
    children: List['ASTNode']
    line_number: int
    semantic_annotations: Dict[str, Any]

    def __init__(self, node_type: str, content: str, type_info: TypeInfo, 
                 children: Optional[List['ASTNode']] = None, line_number: int = 0, 
                 semantic_annotations: Optional[Dict[str, Any]] = None):
        self.node_type = node_type
        self.content = content
        self.type_info = type_info
        self.children = children if children is not None else []
        self.line_number = line_number
        self.semantic_annotations = semantic_annotations if semantic_annotations is not None else {}

class TypeInferrer:
    """Abstract type inference system."""
    
    def __init__(self):
        self.type_patterns = {
            r'\b\d+\.\d+\b': TypeCategory.NUMERIC,  # Float literals
            r'\b\d+\b': TypeCategory.NUMERIC,       # Integer literals
            r'\bTrue\b|\bFalse\b': TypeCategory.BOOLEAN,  # Boolean literals
            r'"[^"]*"': TypeCategory.STRING,        # String literals
            r'\bdef\s+\w+': TypeCategory.FUNCTION,  # Function definitions
        }
        
        self.variable_types: Dict[str, TypeInfo] = {}
        self.function_signatures: Dict[str, List[TypeInfo]] = {}
    
    def infer_type(self, content: str, context: str = "") -> TypeInfo:
        """Infer type from content and context."""
        # Check literal patterns
        for pattern, type_cat in self.type_patterns.items():
            if re.match(pattern, content.strip()):
                return TypeInfo(
                    category=type_cat,
                    is_literal=True,
                    confidence=0.9
                )
        
        # Check for variables
        if re.match(r'\b[a-zA-Z_]\w*\b', content):
            var_name = content.strip()
            if var_name in self.variable_types:
                return self.variable_types[var_name]
            else:
                # Infer from context
                inferred_type = self._infer_from_context(content, context)
                self.variable_types[var_name] = inferred_type
                return inferred_type
        
        return TypeInfo(category=TypeCategory.UNKNOWN, confidence=0.5)
    
    def _infer_from_context(self, var_name: str, context: str) -> TypeInfo:
        """Infer variable type from surrounding context."""
        context_lower = context.lower()
        
        if any(word in context_lower for word in ['count', 'sum', 'total', 'number', 'index']):
            return TypeInfo(category=TypeCategory.NUMERIC, is_variable=True, confidence=0.8)
        elif any(word in context_lower for word in ['flag', 'is_', 'has_', 'valid']):
            return TypeInfo(category=TypeCategory.BOOLEAN, is_variable=True, confidence=0.8)
        elif any(word in context_lower for word in ['name', 'text', 'message', 'string']):
            return TypeInfo(category=TypeCategory.STRING, is_variable=True, confidence=0.8)
        elif 'def ' in context:
            return TypeInfo(category=TypeCategory.FUNCTION, is_variable=True, confidence=0.7)
        
        return TypeInfo(category=TypeCategory.UNKNOWN, is_variable=True, confidence=0.5)

def parse_program(program_code: str) -> List[ASTNode]:
    """
    Parse a DSL program into an enhanced AST with type information.
    
    Args:
        program_code: The program code as a string
        
    Returns:
        List of ASTNode objects with type information
    """
    type_inferrer = TypeInferrer()
    lines = program_code.strip().split('\n')
    ast_nodes = []
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        # Determine node type and content
        node_type, content = _parse_line(line)
        
        # Infer type information
        type_info = type_inferrer.infer_type(content, line)
        
        # Create AST node
        ast_node = ASTNode(
            node_type=node_type,
            content=content,
            type_info=type_info,
            line_number=line_num,
            semantic_annotations={
                'complexity': _estimate_complexity(line),
                'has_side_effects': _has_side_effects(line),
                'is_pure': _is_pure_function(line)
            }
        )
        
        ast_nodes.append(ast_node)
    
    return ast_nodes

def _parse_line(line: str) -> Tuple[str, str]:
    """Parse a single line to determine node type and content."""
    if line.startswith('def '):
        return 'func_def', line
    elif line.startswith('return '):
        return 'return_stmt', line
    elif line.startswith('if '):
        return 'if_stmt', line
    elif line.startswith('while '):
        return 'while_stmt', line
    elif line.startswith('else'):
        return 'else_stmt', line
    elif '=' in line and not line.startswith('if ') and not line.startswith('while '):
        return 'assignment', line
    else:
        return 'expr', line

def _estimate_complexity(line: str) -> int:
    """Estimate cyclomatic complexity of a line."""
    complexity = 1
    complexity += line.count('if') + line.count('while') + line.count('for')
    complexity += line.count('and') + line.count('or')
    complexity += line.count('+') + line.count('-') + line.count('*') + line.count('/')
    return complexity

def _has_side_effects(line: str) -> bool:
    """Check if line has side effects."""
    side_effect_patterns = [
        r'\bprint\b', r'\binput\b', r'\.append\b', r'\.extend\b',
        r'\.insert\b', r'\.remove\b', r'\.pop\b', r'\.clear\b'
    ]
    return any(re.search(pattern, line) for pattern in side_effect_patterns)

def _is_pure_function(line: str) -> bool:
    """Check if line is part of a pure function."""
    return not _has_side_effects(line) and 'return' in line

def tokenize(program_code: str) -> List[str]:
    """
    Tokenize a DSL program with type-aware tokenization.
    
    Args:
        program_code: The program code as a string
        
    Returns:
        List of tokens with type annotations
    """
    # Enhanced tokenization with type awareness
    tokens = []
    
    # Split by whitespace and operators
    raw_tokens = re.findall(r'\b\w+\b|[+\-*/=<>!&|(){}[\],;:]', program_code)
    
    for token in raw_tokens:
        # Add type annotation to tokens
        if re.match(r'\b\d+\.\d+\b', token):
            tokens.append(f"{token}:float")
        elif re.match(r'\b\d+\b', token):
            tokens.append(f"{token}:int")
        elif re.match(r'\bTrue\b|\bFalse\b', token):
            tokens.append(f"{token}:bool")
        elif re.match(r'"[^"]*"', token):
            tokens.append(f"{token}:str")
        elif re.match(r'\bdef\b', token):
            tokens.append(f"{token}:keyword")
        elif re.match(r'\bif\b|\bwhile\b|\belse\b|\breturn\b', token):
            tokens.append(f"{token}:keyword")
        else:
            tokens.append(f"{token}:identifier")
    
    return tokens

def get_type_summary(ast_nodes: List[ASTNode]) -> Dict[str, Any]:
    """Generate a summary of types used in the program."""
    type_counts = {}
    variable_types = {}
    function_types = {}
    
    for node in ast_nodes:
        # Count type categories
        type_cat = node.type_info.category.value
        type_counts[type_cat] = type_counts.get(type_cat, 0) + 1
        
        # Track variable types
        if node.type_info.is_variable:
            var_name = node.content.split('=')[0].strip() if '=' in node.content else node.content
            variable_types[var_name] = node.type_info
        
        # Track function types
        if node.type_info.is_function:
            func_name = node.content.split('(')[0].replace('def ', '').strip()
            function_types[func_name] = node.type_info
    
    return {
        'type_distribution': type_counts,
        'variable_types': {k: v.category.value for k, v in variable_types.items()},
        'function_types': {k: v.category.value for k, v in function_types.items()},
        'type_coverage': len(type_counts) / len(TypeCategory),
        'type_confidence': sum(node.type_info.confidence for node in ast_nodes) / len(ast_nodes)
    } 