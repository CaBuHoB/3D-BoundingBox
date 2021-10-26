"""
file for test Math.py
"""
import unittest
import numpy as np
from library.Math import rotation_matrix


class MathTestCase(unittest.TestCase):
    """ class rotation matrix test """
    def test_rotation_matrix(self):
        """ test rotation matrix """
        val_input = 5
        valid_output = np.array([[0.28366219, 0, -0.95892427],\
                                [0, 1, 0],\
                                [0.95892427, 0, 0.28366219]])
        valid_output = valid_output.reshape([3, 3])
        output = rotation_matrix(val_input)
        self.assertTrue(np.allclose(output, valid_output))

        val_input = 'string'
        with self.assertRaises(TypeError):
            output = rotation_matrix(val_input)


if __name__ == "__main__":
    unittest.main()
