"""
file fot testing plotting.py
"""
import unittest
from library.Plotting import constraint_to_color, create_2d_box


class PlotsTestCase(unittest.TestCase):
    """ class for test plots """
    def test_constraint_to_color(self):
        """ test constraints """
        invalid_index = -1
        with self.assertRaises(KeyError):
            constraint_to_color(invalid_index)

        invalid_index = 5
        with self.assertRaises(KeyError):
            constraint_to_color(invalid_index)

        invalid_index = 'string'
        with self.assertRaises(KeyError):
            constraint_to_color(invalid_index)

    def test_2d_box(self):
        """ test 2d box """
        val_input = [(0, 0), (5, 5)]
        valid_output = (0, 0), (0, 5), (5, 5), (5, 0)
        output = create_2d_box(val_input)
        self.assertEqual(output, valid_output)


if __name__ == "__main__":
    unittest.main()
