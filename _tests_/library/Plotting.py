from library.Plotting import constraint_to_color, create_2d_box
import unittest


class PlotsTestCase(unittest.TestCase):

    def test_constraint_to_color(self):
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
        input = [(0, 0), (5, 5)]
        valid_output = (0, 0), (0, 5), (5, 5), (5, 0)
        output = create_2d_box(input)
        self.assertEqual(output, valid_output)


if __name__ == "__main__":
    unittest.main()
