"""
Tests for the embedding module.
"""

import unittest
from embedding.engine import DSLToVectorEngine


class TestEmbeddingEngine(unittest.TestCase):
    """Test cases for the embedding engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = DSLToVectorEngine()
        self.test_program = "def square(x): return x * x"
    
    def test_encode_program(self):
        """Test program encoding."""
        result = self.engine.encode_program(self.test_program)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.vector), 128)  # Default vector dimension
        self.assertGreater(result.confidence, 0)
    
    def test_similarity_computation(self):
        """Test similarity computation."""
        # Encode two programs
        self.engine.encode_program("def square(x): return x * x")
        self.engine.encode_program("def cube(x): return x * x * x")
        
        # Get program IDs
        program_id1 = list(self.engine.embeddings.keys())[0]
        program_id2 = list(self.engine.embeddings.keys())[1]
        
        # Test similarity
        similarity = self.engine.compute_similarity(program_id1, program_id2)
        self.assertGreater(similarity, -1)  # Similarity should be between -1 and 1
        self.assertLess(similarity, 1)


if __name__ == '__main__':
    unittest.main() 