"""
Example DSL Programs

This module contains example programs in the DSL for testing and demonstration.
"""

from typing import Optional

examples = {
    "square": """
    def square(x):
        return x * x

    y = square(5)
    """,

    "factorial": """
    def fact(n):
        if n <= 1:
            return 1
        else:
            return n * fact(n - 1)

    result = fact(5)
    """,

    "fibonacci": """
    def fib(n):
        if n <= 1:
            return n
        else:
            return fib(n - 1) + fib(n - 2)

    x = fib(6)
    """,

    "loop_sum": """
    def loop_sum(n):
        total = 0
        while n > 0:
            total = total + n
            n = n - 1
        return total

    result = loop_sum(10)
    """,

    "logic_test": """
    def logic(x, y):
        if x > 0 && y > 0:
            return 1
        else:
            return 0

    result = logic(1, 2)
    """
}

def list_examples():
    """Return a list of available example names."""
    return list(examples.keys())

def get_example(name: str) -> Optional[str]:
    """Get an example program by name."""
    return examples.get(name) 