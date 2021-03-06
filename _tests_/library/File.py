"""
test file for File.py
"""
import unittest
import os
import warnings
import numpy as np
from library.File import get_calibration_cam_to_image, get_p, get_r0


class CabFileTestCase(unittest.TestCase):
    """ test class """
    def setUp(self) -> None:
        """ function with setup """
        self.path_to_fake_file_type_1 = './cab_file_type_1.txt'
        file_content = 'P2: 1 2 3 4 5 6 7 8 9 10 11 12'
        with open(self.path_to_fake_file_type_1, 'w') as file:
            file.write(file_content)

        self.path_to_fake_file_type_2 = './cab_file_type_2.txt'
        file_content = 'P_rect_02: 0 5 10 15 1 2 3 4 6 7 8 9'
        with open(self.path_to_fake_file_type_2, 'w') as file:
            file.write(file_content)

        self.path_to_fake_file_type_3 = './cab_file_type_3.txt'
        file_content = 'R0_rect: 1 2 3 4 5 6 7 8 9'
        with open(self.path_to_fake_file_type_3, 'w') as file:
            file.write(file_content)

    def tearDown(self) -> None:
        """ function with teardown """
        os.remove(self.path_to_fake_file_type_1)
        os.remove(self.path_to_fake_file_type_2)
        os.remove(self.path_to_fake_file_type_3)

    def test_get_calibration_valid(self):
        """ test with test calibtarion """
        output = get_calibration_cam_to_image(self.path_to_fake_file_type_1)
        valid_output = np.array([[1, 2, 3, 4],\
                                [5, 6, 7, 8],\
                                [9, 10, 11, 12]])
        self.assertTrue(np.allclose(output, valid_output))

    def test_get_calibration_invalid(self):
        """ test get calibration """
        invalid_path = 'invalid_path'
        with self.assertRaises(FileNotFoundError):
            get_calibration_cam_to_image(invalid_path)

    def test_get_p_filetype2(self):
        """ test get p """
        warnings.simplefilter("ignore")
        output = get_p(self.path_to_fake_file_type_2)
        valid_output_type_2 = np.array([[0, 5, 10, 15],\
                                        [1, 2, 3, 4],\
                                        [6, 7, 8, 9]])
        self.assertTrue(np.allclose(output, valid_output_type_2))

    def test_get_p_filetype1(self):
        """ test get p filetype 1"""
        warnings.simplefilter("ignore")
        output = get_p(self.path_to_fake_file_type_1)
        valid_output_type_1 = np.array([[1, 2, 3, 4],
                                        [5, 6, 7, 8],
                                        [9, 10, 11, 12]])
        with self.assertRaises(TypeError):
            np.allclose(output, valid_output_type_1)

    def test_get_r0(self):
        """ test get r0"""
        warnings.simplefilter("ignore")

        output = get_r0(self.path_to_fake_file_type_3)
        valid_output = np.array([[1, 2, 3, 0],\
                                [4, 5, 6, 0],\
                                [7, 8, 9, 0],\
                                [0, 0, 0, 1]])
        self.assertTrue(np.allclose(output, valid_output))


if __name__ == "__main__":
    unittest.main()
    