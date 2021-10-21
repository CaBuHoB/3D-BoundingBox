"""
Functions to read from files
TODO: move the functions that read label from Dataset into here
"""
import sys
import numpy as np


def get_calibration_cam_to_image(cab_f):
    """ Gets calibration cam to image """
    for line in open(cab_f):
        if 'P2:' in line:
            cam_to_img = line.strip().split(' ')
            cam_to_img = np.asarray([float(number) for number in cam_to_img[1:]])
            cam_to_img = np.reshape(cam_to_img, (3, 4))
            return cam_to_img

    file_not_found(cab_f)
    return None

def get_p(cab_f):
    """ Gets P """
    for line in open(cab_f):
        if 'P_rect_02' in line:
            cam_p = line.strip().split(' ')
            cam_p = np.asarray([float(cam_p) for cam_p in cam_p[1:]])
            return_matrix = np.zeros((3, 4))
            return_matrix = cam_p.reshape((3, 4))
            return return_matrix

    # try other type of file
    return get_calibration_cam_to_image

def get_val_r0(cab_f):
    """ Gets val_r0 """
    for line in open(cab_f):
        if 'val_r0_rect:' in line:
            val_r0 = line.strip().split(' ')
            val_r0 = np.asarray([float(number) for number in val_r0[1:]])
            val_r0 = np.reshape(val_r0, (3, 3))

            val_r0_rect = np.zeros([4, 4])
            val_r0_rect[3, 3] = 1
            val_r0_rect[:3, :3] = val_r0

            return val_r0_rect
    return None

def get_tr_to_velo(cab_f):
    """ Gets tr to velo """
    for line in open(cab_f):
        if 'Tr_velo_to_cam:' in line:
            val_tr = line.strip().split(' ')
            val_tr = np.asarray([float(number) for number in val_tr[1:]])
            val_tr = np.reshape(val_tr, (3, 4))

            tr_to_velo = np.zeros([4, 4])
            tr_to_velo[3, 3] = 1
            tr_to_velo[:3, :4] = val_tr

            return tr_to_velo
    return None

def file_not_found(filename):
    """ Checks if file exists """
    print("\nError! Can't read calibration file, does %s exist?"%filename)
    sys.exit()
