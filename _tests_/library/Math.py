from library.Math import rotation_matrix
import numpy as np
import unittest


class MathTestCase(unittest.TestCase):

    def test_rotation_matrix(self):
        input = 5
        valid_output = np.array([[0.28366219, 0, -0.95892427],
                                [0, 1, 0],
                                [0.95892427, 0, 0.28366219]])
        valid_output = valid_output.reshape([3,3])
        output = rotation_matrix(input)
        self.assertTrue(np.allclose(output, valid_output))

        input = 'string'
        with self.assertRaises(TypeError):
            output = rotation_matrix(input)


if __name__ == "__main__":
    unittest.main()
