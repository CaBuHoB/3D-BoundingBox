"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""
import os
import sys
import time
import argparse
import numpy as np
import cv2

import torch
from torch_lib.Dataset import Dataset
from torch_lib import Model, ClassAverages
from torchvision.models import vgg
from library.Math import calc_location
from library.Plotting import plot_2d_box, plot_3d_box





# to run car by car
SINGLECAR = False

PARSER = argparse.ArgumentParser()

PARSER.add_argument("--dataset-path", default="eval/",
                    help="Path to directory with dataset")

PARSER.add_argument("--calib-path", default="camera_cal/calib_cam_to_cam.txt",
                    help="Path file with calibrating data for camera")

PARSER.add_argument("--weights-path", default="weights/",
                    help="Path to folder, where weights will be saved. \
                        By default, this is weights/")

PARSER.add_argument("--imwrite", action="store_true",
                    help="Flag for running the code in the mode of saving images to a folder. \
                        If this flag is used, the files are saved in output_dir. \
                        By default, images are displayed using cv2.imshow.")

PARSER.add_argument("--output-dir", default="output_dir/",
                    help="If the imwrite flag is True, the images will be saved to this directory. \
                        By default, this is output_dir/")

PARSER.add_argument("--device", default="cuda",
                    help="PyTorch device: cuda/cpu")


def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):
    """ plot regressed 3d box """
    # the math! returns X, the corners used for constraint
    location, _ = calc_location(
        dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location)  # 3d boxes

    return location


def main():
    """ main """
    flags = PARSER.parse_args()

    device = flags.device

    if flags.imwrite:
        os.makedirs(flags.output_dir, exist_ok=True)

    weights_path = flags.weights_path
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        sys.exit()
    else:
        print('Using previous model %s' % model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).to(device)
        checkpoint = torch.load(os.path.join(weights_path, model_lst[-1]), map_location=torch.device(device))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # defaults to /eval
    dataset = Dataset(flags.dataset_path, flags.calib_path)
    averages = ClassAverages.ClassAverages()

    all_images = dataset.all_objects()
    for key in sorted(all_images.keys()):

        start_time = time.time()

        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for i, detected_obj in enumerate(objects):
            label = detected_obj.label
            theta_ray = detected_obj.theta_ray
            input_img = detected_obj.img

            input_tensor = torch.zeros([1, 3, 224, 224]).to(device)
            input_tensor[0, :, :, :] = input_img
            input_tensor.to(device)

            [orient, conf, dim] = model(input_tensor)
            orient = orient.cpu().data.numpy()[0, :, :]
            conf = conf.cpu().data.numpy()[0, :]
            dim = dim.cpu().data.numpy()[0, :]

            dim += averages.get_item(label['Class'])

            argmax = np.argmax(conf)
            orient = orient[argmax, :]
            cos = orient[0]
            sin = orient[1]
            alpha = np.arctan2(sin, cos)
            alpha += dataset.angle_bins[argmax]
            alpha -= np.pi

            location = plot_regressed_3d_bbox(\
                img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

            print('Estimated pose: %s' % location)
            print('Truth pose: %s' % label['Location'])
            print('-------------')

            # plot car by car
            if not SINGLECAR:
                continue
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            if not flags.imwrite:
                cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
                cv2.waitKey(1)
            else:
                cv2.imwrite(os.path.join(flags.output_dir, f'im_sc_{key}_{i}.jpg'), numpy_vertical)

        print('Got %s poses in %.3f seconds\n' % (len(objects), time.time() - start_time))

        # plot image by image
        if SINGLECAR:
            continue
        numpy_vertical = np.concatenate((truth_img, img), axis=0)
        if not flags.imwrite:
            cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
            cv2.waitKey(1)
        else:
            cv2.imwrite(os.path.join(flags.output_dir, f'im_sc_{key}.jpg'), numpy_vertical)


if __name__ == '__main__':
    main()
