"""
test file for orientation loss
"""
import unittest
from torch_lib.Model import orientation_loss


class OrientationLossTestCase(unittest.TestCase):
    """ test class for orientation loss """
    def test_invalid_input(self):
        """ test function for orientation loss """
        with self.assertRaises(AttributeError):
            val_input = 0
            orientation_loss(val_input, val_input, val_input)


if __name__ == "__main__":
    unittest.main()
