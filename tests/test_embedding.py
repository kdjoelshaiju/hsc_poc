"""
Tests for the embedding module.
"""

import unittest
from embedding.engine import DSLToVectorEngine, EmbeddingConfig


class TestEmbeddingEngine(unittest.TestCase):
    """Test cases for the embedding engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = DSLToVectorEngine()
        self.test_program = "def square(x): return x * x"
    
    def test_encode_program(self):
        """Test program encoding."""
        embedding = self.engine.encode_program(self.test_program)
        self.assertIsNotNone(embedding)
        self.assertEqual(len(embedding), 64)  # Default vector dimension
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        program1 = "def square(x): return x * x"
        program2 = "def cube(x): return x * x * x"
        
        similar = self.engine.find_most_similar(program1, [program2], top_k=1)
        self.assertEqual(len(similar), 1)
        self.assertGreater(similar[0][1], 0)  # Similarity should be positive


if __name__ == '__main__':
    unittest.main() 