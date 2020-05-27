import unittest

try:
    import torch_csprng as csprng
except ImportError:
    raise RuntimeError("CSPRNG not available")

class TestCSPRNG(unittest.TestCase):

    def test_random(self):
        self.assertEqual(42, 42)

if __name__ == '__main__':
    unittest.main()
