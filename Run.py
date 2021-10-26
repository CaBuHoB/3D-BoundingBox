"""
Images must be in ./Kitti/testing/image_2/ and camera matricies in ./Kitti/testing/calib/

Uses YOLO to obtain 2D box, PyTorch to get 3D box, plots both

SPACE bar for next image, any other key to exit
"""


import os
import sys
import time
import argparse
import numpy as np

import cv2

import torch
from torchvision.models import vgg
from torch_lib.Dataset import DetectedObject, generate_bins
from torch_lib import Model, ClassAverages
from library.Math import calc_location
from library.Plotting import plot_2d_box, plot_3d_box
from yolo.yolo import CvYolo


PARSER = argparse.ArgumentParser()

PARSER.add_argument("--dataset-path", default="eval/image_2/",
                    help="Relative path to the directory containing images to detect. Default \
                        is eval/image_2/")

PARSER.add_argument("--calib-path", default="camera_cal/calib_cam_to_cam.txt",
                    help="Path file with calibrating data for camera")

PARSER.add_argument("--video", action="store_true",
                    help="Weather or not to advance frame-by-frame as fast as possible. \
                        By default, this will pull images from ./eval/video")

PARSER.add_argument("--show-yolo", action="store_true",
                    help="Show the 2D BoundingBox detecions on a separate image")

PARSER.add_argument("--hide-debug", action="store_true",
                    help="Supress the printing of each 3d location")

PARSER.add_argument("--imwrite", action="store_true",
                    help="Flag for running the code in the mode of saving images to a folder. \
                        If this flag is used, the files are saved in output_dir. \
                        By default, images are displayed using cv2.imshow.")

PARSER.add_argument("--output-dir", default="output_dir/",
                    help="If the imwrite flag is True, the images will be saved to this directory. \
                        By default, this is output_dir/")

PARSER.add_argument("--weights-path", default="weights/",
                    help="Path to folder, where weights will be saved. \
                        By default, this is weights/")

PARSER.add_argument("--device", default="cuda",
                    help="PyTorch device: cuda/cpu")


def plot_regressed_3d_bbox(img, cam_to_img, box_2d, dimensions, alpha, theta_ray, img_2d=None):
    """ plots regressed 3d box """
    # the math! returns X, the corners used for constraint
    location, _ = calc_location(
        dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    if img_2d is not None:
        plot_2d_box(img_2d, box_2d)

    plot_3d_box(img, cam_to_img, orient, dimensions, location)  # 3d boxes

    return location


def main():
    """ main function """
    flags = PARSER.parse_args()

    device = flags.device

    # load torch
    weights_path = flags.weights_path
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        sys.exit()
    else:
        print('Using previous model %s' % model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).to(device)
        checkpoint = torch.load(os.path.join(\
            weights_path, model_lst[-1]), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # load yolo
    yolo = CvYolo(weights_path)

    averages = ClassAverages.ClassAverages()

    angle_bins = generate_bins(2)

    img_path = flags.dataset_path
    if flags.video and flags.dataset_path == "eval/image_2/":
        img_path = "eval/video/2011_09_26/image_2/"

    if flags.imwrite:
        os.makedirs(flags.output_dir, exist_ok=True)

    # using P_rect from global calibration file
    calib_file = flags.calib_path

    # using P from each frame
    # calib_path = os.path.abspath(os.path.dirname(__file__)) + '/Kitti/testing/calib/'

    try:
        ids = [x.split('.')[0] for x in sorted(os.listdir(img_path))]
    except ValueError:
        print("\nError: no images in %s" % img_path)
        sys.exit()

    for i, img_id in enumerate(ids):

        start_time = time.time()

        img_file = os.path.join(img_path, img_id + ".png")

        # P for each frame
        # calib_file = calib_path + id + ".txt"

        truth_img = cv2.imread(img_file)
        img = np.copy(truth_img)
        yolo_img = np.copy(truth_img)

        detections = yolo.detect(yolo_img)

        for detection in detections:

            if not averages.recognized_class(detection.detected_class):
                continue

            # this is throwing when the 2d bbox is invalid
            try:
                det_obj = DetectedObject(\
                    img, detection.detected_class, detection.box_2d, calib_file)
            except ValueError:
                continue

            theta_ray = det_obj.theta_ray
            input_img = det_obj.img
            proj_matrix = det_obj.proj_matrix
            box_2d = detection.box_2d
            detected_class = detection.detected_class

            input_tensor = torch.zeros([1, 3, 224, 224]).to(device)
            input_tensor[0, :, :, :] = input_img

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(detected_class)

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += angle_bins[argmax]
            alpha -= np.pi

            if flags.show_yolo:
                location = plot_regressed_3d_bbox(\
                    img, proj_matrix, box_2d, dim, alpha, theta_ray, truth_img)
            else:
                location = plot_regressed_3d_bbox(\
                    img, proj_matrix, box_2d, dim, alpha, theta_ray)

            if not flags.hide_debug:
                print('Estimated pose: %s' % location)

        if flags.show_yolo:
            img = np.concatenate((truth_img, img), axis=0)
        if flags.imwrite:
            cv2.imwrite(os.path.join(flags.output_dir, f'im_{img_id}.jpg'), img)
        elif flags.show_yolo:
            cv2.imshow('SPACE for next image, any other key to exit', img)
        else:
            cv2.imshow('3D detections', img)
        print(f'Done: [{i}/{len(ids)}]')

        if not flags.hide_debug:
            print("\n")
            print('Got %s poses in %.3f seconds' %
                  (len(detections), time.time() - start_time))
            print('-------------')

        if flags.imwrite:
            continue
        if flags.video:
            cv2.waitKey(1)
        elif cv2.waitKey(0) != 32:  # space bar
            sys.exit()


if __name__ == '__main__':
    main()
