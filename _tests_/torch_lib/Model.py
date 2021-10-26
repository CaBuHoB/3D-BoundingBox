import unittest
from torch_lib.Model import Model, OrientationLoss


class OrientationLossTestCase(unittest.TestCase):
    def test_invalid_input(self):
        with self.assertRaises(AttributeError):
            input = 0
            OrientationLoss(input, input, input)


if __name__ == "__main__":
    unittest.main()
