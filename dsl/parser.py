"""
DSL Parser

This module provides a simple parser for the domain-specific language.
"""

from typing import Any, Dict, List, Union
import re

def parse_program(program_code: str) -> Any:
    """
    Parse a DSL program into an abstract syntax tree.
    
    Args:
        program_code: The program code as a string
        
    Returns:
        Abstract syntax tree representation
    """
    # Simple parser implementation
    # In a full implementation, this would use Lark or similar
    
    lines = program_code.strip().split('\n')
    ast = []
    
    for line in lines:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
            
        # Parse function definitions
        if line.startswith('def '):
            ast.append(('func_def', line))
        # Parse return statements
        elif line.startswith('return '):
            ast.append(('return_stmt', line))
        # Parse if statements
        elif line.startswith('if '):
            ast.append(('if_stmt', line))
        # Parse while statements
        elif line.startswith('while '):
            ast.append(('while_stmt', line))
        # Parse else statements
        elif line.startswith('else'):
            ast.append(('else_stmt', line))
        # Parse assignments
        elif '=' in line and not line.startswith('if ') and not line.startswith('while '):
            ast.append(('assignment', line))
        # Parse other expressions
        else:
            ast.append(('expr', line))
    
    return ast

def tokenize(program_code: str) -> List[str]:
    """
    Tokenize a DSL program.
    
    Args:
        program_code: The program code as a string
        
    Returns:
        List of tokens
    """
    # Simple tokenization
    tokens = re.findall(r'\b\w+\b|[+\-*/=<>!&|(){}[\],;:]', program_code)
    return tokens 