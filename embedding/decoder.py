"""
Vector-to-AST Decoder

This module provides functionality to convert vector representations back into
Abstract Syntax Trees (ASTs), enabling semantic interpolation and program synthesis.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import json

def decode_vec_to_ast(vector: np.ndarray, target_structure: Optional[Any] = None) -> Any:
    """
    Decode a vector back into an AST structure.
    
    Args:
        vector: Vector representation to decode
        target_structure: Optional target AST structure to guide decoding
        
    Returns:
        Reconstructed AST
    """
    if target_structure is not None:
        return _guided_decode(vector, target_structure)
    else:
        return _free_decode(vector)

def _guided_decode(vector: np.ndarray, target_structure: Any) -> Any:
    """
    Decode vector using target structure as a guide.
    
    Args:
        vector: Vector representation
        target_structure: Target AST structure to match
        
    Returns:
        Reconstructed AST matching target structure
    """
    if isinstance(target_structure, tuple):
        # Reconstruct tuple structure
        node_type = target_structure[0]
        children = []
        
        # Use vector values to influence child reconstruction
        for i, child in enumerate(target_structure[1:]):
            child_vector = vector[i * 16:(i + 1) * 16]  # Use portion of vector for each child
            reconstructed_child = _guided_decode(child_vector, child)
            children.append(reconstructed_child)
        
        return (node_type,) + tuple(children)
    
    elif isinstance(target_structure, list):
        # Reconstruct list structure
        children = []
        for i, child in enumerate(target_structure):
            child_vector = vector[i * 16:(i + 1) * 16]
            reconstructed_child = _guided_decode(child_vector, child)
            children.append(reconstructed_child)
        
        return children
    
    else:
        # Leaf node - return as is
        return target_structure

def _free_decode(vector: np.ndarray) -> Any:
    """
    Free-form decoding without target structure.
    
    Args:
        vector: Vector representation
        
    Returns:
        Reconstructed AST
    """
    # This is a simplified decoder that creates a basic structure
    # In practice, this would be much more sophisticated
    
    # Use vector values to determine structure
    if len(vector) < 16:
        return "atom"
    
    # Simple heuristic: use first few values to determine node type
    node_type_score = np.mean(vector[:8])
    
    if node_type_score > 0.7:
        # Function definition
        return ("func_def", "main", [], ("return_stmt", ("atom", "0")))
    elif node_type_score > 0.5:
        # If statement
        return ("if_stmt", ("atom", "condition"), ("assignment", "x", ("atom", "1")))
    elif node_type_score > 0.3:
        # Assignment
        return ("assignment", "x", ("expr_binop", "+", ("atom", "a"), ("atom", "b")))
    else:
        # Simple atom
        return ("atom", "value")

def decode_to_program(vector: np.ndarray, target_structure: Optional[Any] = None) -> str:
    """
    Decode vector directly to program code.
    
    Args:
        vector: Vector representation
        target_structure: Optional target structure
        
    Returns:
        Program code as string
    """
    ast = decode_vec_to_ast(vector, target_structure)
    return _ast_to_code(ast)

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

def interpolate_asts(ast1: Any, ast2: Any, t: float) -> Any:
    """
    Interpolate between two ASTs.
    
    Args:
        ast1: First AST
        ast2: Second AST
        t: Interpolation parameter (0.0 to 1.0)
        
    Returns:
        Interpolated AST
    """
    # This is a simplified interpolation
    # In practice, this would be much more sophisticated
    
    if t < 0.5:
        return ast1
    else:
        return ast2

def validate_ast(ast: Any) -> bool:
    """
    Validate that an AST has a reasonable structure.
    
    Args:
        ast: AST to validate
        
    Returns:
        True if AST is valid, False otherwise
    """
    try:
        # Basic validation: check if it's a reasonable structure
        if isinstance(ast, tuple):
            if len(ast) == 0:
                return False
            
            # Check if first element is a string (node type)
            if not isinstance(ast[0], str):
                return False
            
            # Recursively validate children
            for child in ast[1:]:
                if not validate_ast(child):
                    return False
            
            return True
        
        elif isinstance(ast, list):
            # Validate all list elements
            return all(validate_ast(item) for item in ast)
        
        else:
            # Leaf nodes are always valid
            return True
    
    except Exception:
        return False 