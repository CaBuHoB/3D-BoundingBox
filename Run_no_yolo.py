"""
This script will use the 2D box from the label rather than from YOLO,
but will still use the neural nets to get the 3D position and plot onto the
image. Press space for next image and escape to quit
"""

import os
import sys
import time
import cv2
import numpy as np

import torch
from torchvision.models import vgg
from torch_lib.Dataset import Dataset
from torch_lib import Model, ClassAverages
from library.Math import calc_location
from library.Plotting import plot_2d_box, plot_3d_box



# to run car by car
CAR = False

def plot_regressed_3d_bbox(img, truth_img, cam_to_img, box_2d, dimensions, alpha, theta_ray):
    """ plots regressed 3d box """
    # the math! returns X, the corners used for constraint
    location, [] = calc_location(dimensions, cam_to_img, box_2d, alpha, theta_ray)

    orient = alpha + theta_ray

    plot_2d_box(truth_img, box_2d)
    plot_3d_box(img, cam_to_img, orient, dimensions, location) # 3d boxes

    return location

def main():
    """ main function """
    weights_path = os.path.abspath(os.path.dirname(__file__)) + '/weights'
    model_lst = [x for x in sorted(os.listdir(weights_path)) if x.endswith('.pkl')]
    if len(model_lst) == 0:
        print('No previous model found, please train first!')
        sys.exit()
    else:
        print('Using previous model %s'%model_lst[-1])
        my_vgg = vgg.vgg19_bn(pretrained=True)
        model = Model.Model(features=my_vgg.features, bins=2).cuda()
        checkpoint = torch.load(weights_path + '/%s'%model_lst[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    # defaults to /eval
    dataset = Dataset(os.path.abspath(os.path.dirname(__file__)) + '/eval')
    averages = ClassAverages.ClassAverages()

    all_images = dataset.all_objects()
    for key in sorted(all_images.keys()):

        start_time = time.time()

        data = all_images[key]

        truth_img = data['Image']
        img = np.copy(truth_img)
        objects = data['Objects']
        cam_to_img = data['Calib']

        for det_obj in objects:
            label = det_obj.label
            theta_ray = det_obj.theta_ray
            input_img = det_obj.img

            input_tensor = torch.zeros([1, 3, 224, 224]).cuda()
            input_tensor[0, :, :, :] = input_img
            input_tensor.cuda()

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

            location = plot_regressed_3d_bbox( \
                img, truth_img, cam_to_img, label['Box_2D'], dim, alpha, theta_ray)

            print('Estimated pose: %s'%location)
            print('Truth pose: %s'%label['Location'])
            print('-------------')

            # plot car by car
            if CAR:
                numpy_vertical = np.concatenate((truth_img, img), axis=0)
                cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
                cv2.waitKey(0)

        print('Got %s poses in %.3f seconds\n'%(len(objects), time.time() - start_time))

        # plot image by image
        if not CAR:
            numpy_vertical = np.concatenate((truth_img, img), axis=0)
            cv2.imshow('2D detection on top, 3D prediction on bottom', numpy_vertical)
            if cv2.waitKey(0) == 27:
                return

if __name__ == '__main__':
    main()
