"""
test file for yolo
"""
import warnings
import unittest
import cv2
from yolo.yolo import CvYolo


class YoloTestCase(unittest.TestCase):
    """ class for yolo test """
    def test_invalid_path(self):
        """ test invalid path """
        invalid_path = 'unexisted_path'
        with self.assertRaises(FileNotFoundError):
            CvYolo(invalid_path)

        invalid_path = 0
        with self.assertRaises(TypeError):
            CvYolo(invalid_path)

    def test_valid_detections(self):
        """ test valid detections """
        warnings.simplefilter("ignore")
        valid_path = './weights'
        with self.assertRaises(cv2.error):
            CvYolo(valid_path)


if __name__ == "__main__":
    unittest.main()
