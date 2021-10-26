import warnings
import unittest
import cv2
from yolo.yolo import cv_Yolo


class YoloTestCase(unittest.TestCase):

    def test_invalid_path(self):
        invalid_path = 'unexisted_path'
        with self.assertRaises(FileNotFoundError):
            cv_Yolo(invalid_path)

        invalid_path = 0
        with self.assertRaises(TypeError):
            cv_Yolo(invalid_path)

    def test_valid_detections(self):
        warnings.simplefilter("ignore")
        valid_path = './weights'
        with self.assertRaises(cv2.error):
            cv_Yolo(valid_path)


if __name__ == "__main__":
    unittest.main()
