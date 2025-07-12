"""
Tests for the interpolation module.
"""

import unittest
from interpolation.geodesic import GeodesicInterpolator
from manifold.autoencoder import AutoencoderManifold


class TestGeodesicInterpolator(unittest.TestCase):
    """Test cases for the geodesic interpolator."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.manifold = AutoencoderManifold(input_dim=64, latent_dim=16)
        self.interpolator = GeodesicInterpolator(self.manifold)
        self.start_program = "def square(x): return x * x"
        self.end_program = "def cube(x): return x * x * x"
    
    def test_interpolation(self):
        """Test program interpolation."""
        interpolated = self.interpolator.interpolate(
            self.start_program, self.end_program, steps=5
        )
        self.assertEqual(len(interpolated), 5)
        self.assertIsInstance(interpolated[0], str)


if __name__ == '__main__':
    unittest.main() 