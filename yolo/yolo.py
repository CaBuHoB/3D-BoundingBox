"""
Will use opencv's built in darknet api to do 2D object detection which will
then get passed into the torch net

source: https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
"""

import os
import numpy as np
import cv2

class CvYolo:
    """ class CvYolo"""
    def __init__(self, yolo_path, confidence=0.5, threshold=0.3):
        self.confidence = confidence
        self.threshold = threshold

        labels_path = os.path.sep.join([yolo_path, "coco.names"])
        self.labels = open(labels_path).read().split("\n")

        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.labels), 3), dtype="uint8")

        weights_path = os.path.sep.join([yolo_path, "yolov3.weights"])
        cfg_path = os.path.sep.join([yolo_path, "yolov3.cfg"])

        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)

    def detect(self, image):
        """ detection function """
        # assert image is opencv
        (val_h, val_w) = image.shape[:2]

        layer_num = self.net.getLayerNames()
        layer_num = [layer_num[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # prepare input
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)

        self.net.setInput(blob)
        output = self.net.forward(layer_num)

        detections = []

        boxes = []
        confidences = []
        class_ids = []

        for output in output:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > self.confidence:

                    box = detection[0:4] * np.array([val_w, val_h, val_w, val_h])
                    (center_x, center_y, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    val_x = int(center_x - (width / 2))
                    val_y = int(center_y - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs

                    boxes.append([val_x, val_y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)



        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)

        if len(idxs) > 0:
            for i in idxs.flatten():

                top_left = (boxes[i][0], boxes[i][1])
                bottom_right = (top_left[0] + boxes[i][2], top_left[1] + boxes[i][3])

                box_2d = [top_left, bottom_right]
                class_ = self.get_class(class_ids[i])
                if class_ == "person":
                    class_ = "pedestrian"

                detections.append(Detection(box_2d, class_))

        return detections

    def get_class(self, class_id):
        """ get class function """
        return self.labels[class_id]



class Detection:
    """ class detection"""
    def __init__(self, box_2d, class_):
        """ initialization function """
        self.box_2d = box_2d
        self.detected_class = class_
