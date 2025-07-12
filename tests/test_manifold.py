"""
Tests for the manifold module.
"""

import unittest
import numpy as np
from manifold.autoencoder import AutoencoderManifold


class TestAutoencoderManifold(unittest.TestCase):
    """Test cases for the autoencoder manifold."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.autoencoder = AutoencoderManifold(input_dim=64, latent_dim=16)
        self.test_embeddings = np.random.randn(10, 64)
    
    def test_fit(self):
        """Test manifold fitting."""
        self.autoencoder.fit(self.test_embeddings)
        self.assertTrue(self.autoencoder.is_fitted)
    
    def test_encode_decode(self):
        """Test encoding and decoding."""
        self.autoencoder.fit(self.test_embeddings)
        
        encoded = self.autoencoder.encode(self.test_embeddings)
        decoded = self.autoencoder.decode(encoded)
        
        self.assertEqual(encoded.shape, (10, 16))
        self.assertEqual(decoded.shape, (10, 64))


if __name__ == '__main__':
    unittest.main() 