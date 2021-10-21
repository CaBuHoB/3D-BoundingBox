"""
Plotting file
"""

from enum import Enum
import cv2
import numpy as np


from .File import get_calibration_cam_to_image
from .Math import create_corners, rotation_matrix

class CvColors(Enum):
    """ Color values """
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    PURPLE = (247, 44, 200)
    ORANGE = (44, 162, 247)
    MINT = (239, 255, 66)
    YELLOW = (2, 255, 250)

def constraint_to_color(constraint_idx):
    """ Constraints to color """
    return {
        0 : CvColors.PURPLE.value, #left
        1 : CvColors.ORANGE.value, #top
        2 : CvColors.MINT.value, #right
        3 : CvColors.YELLOW.value #bottom
    }[constraint_idx]


# from the 2 corners, return the 4 corners of a box in CCW order
# coulda just used cv2.rectangle haha
def create_2d_box(box_2d):
    """ Creates 2d box """
    corner1_2d = box_2d[0]
    corner2_2d = box_2d[1]

    pt1 = corner1_2d
    pt2 = (corner1_2d[0], corner2_2d[1])
    pt3 = corner2_2d
    pt4 = (corner2_2d[0], corner1_2d[1])

    return pt1, pt2, pt3, pt4


# takes in a 3d point and projects it into 2d
def project_3d_pt(val_pt, cam_to_img, calib_file=None):
    """ Project 3d pts """
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)
        #r0_rect = get_R0(calib_file)
        #tr_velo_to_cam = get_tr_to_velo(calib_file)

    point = np.array(val_pt)
    point = np.append(point, 1)

    point = np.dot(cam_to_img, point)
    # point = np.dot(np.dot(np.dot(cam_to_img, r0_rect), tr_velo_to_cam), point)

    point = point[:2]/point[2]
    point = point.astype(np.int16)

    return point



# take in 3d points and plot them on image as red circles
def plot_3d_pts(\
    img, val_pts, center, calib_file=None, cam_to_img=None, relative=False, constraint_idx=None):
    """ Plots 3d pts """
    if calib_file is not None:
        cam_to_img = get_calibration_cam_to_image(calib_file)

    for count_pt in val_pts:
        if relative:
            count_pt = [i + center[j] for j, i in enumerate(count_pt)] # more pythonic

        point = project_3d_pt(count_pt, cam_to_img)

        color = CvColors.RED.value

        if constraint_idx is not None:
            color = constraint_to_color(constraint_idx)

        cv2.circle(img, (point[0], point[1]), 3, color, thickness=-1)



def plot_3d_box(img, cam_to_img, val_ry, dimension, center):
    """ Plots 3D box """
    # plot_3d_pts(img, [center], center, calib_file=calib_file, cam_to_img=cam_to_img)

    val_r = rotation_matrix(val_ry)

    corners = create_corners(dimension, location=center, val_r=val_r)

    # to see the corners on image as red circles
    # plot_3d_pts(img, corners, center,cam_to_img=cam_to_img, relative=False)

    box_3d = []
    for corner in corners:
        point = project_3d_pt(corner, cam_to_img)
        box_3d.append(point)

    cv2.line(img, (box_3d[0][0], box_3d[0][1]), \
        (box_3d[2][0], box_3d[2][1]), CvColors.GREEN.value, 1)
    cv2.line(img, (box_3d[4][0], box_3d[4][1]), \
        (box_3d[6][0], box_3d[6][1]), CvColors.GREEN.value, 1)
    cv2.line(img, (box_3d[0][0], box_3d[0][1]), \
        (box_3d[4][0], box_3d[4][1]), CvColors.GREEN.value, 1)
    cv2.line(img, (box_3d[2][0], box_3d[2][1]), \
        (box_3d[6][0], box_3d[6][1]), CvColors.GREEN.value, 1)

    cv2.line(img, (box_3d[1][0], box_3d[1][1]), \
        (box_3d[3][0], box_3d[3][1]), CvColors.GREEN.value, 1)
    cv2.line(img, (box_3d[1][0], box_3d[1][1]), \
        (box_3d[5][0], box_3d[5][1]), CvColors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), \
        (box_3d[3][0], box_3d[3][1]), CvColors.GREEN.value, 1)
    cv2.line(img, (box_3d[7][0], box_3d[7][1]), \
        (box_3d[5][0], box_3d[5][1]), CvColors.GREEN.value, 1)

    for i in range(0, 7, 2):
        cv2.line(img, (box_3d[i][0], box_3d[i][1]), \
            (box_3d[i+1][0], box_3d[i+1][1]), CvColors.GREEN.value, 1)

    front_mark = [(box_3d[i][0], box_3d[i][1]) for i in range(4)]

    cv2.line(img, front_mark[0], front_mark[3], CvColors.BLUE.value, 1)
    cv2.line(img, front_mark[1], front_mark[2], CvColors.BLUE.value, 1)

def plot_2d_box(img, box_2d):
    """ Plots 2D box """
    # create a square from the corners
    pt1, pt2, pt3, pt4 = create_2d_box(box_2d)

    # plot the 2d box
    cv2.line(img, pt1, pt2, CvColors.BLUE.value, 2)
    cv2.line(img, pt2, pt3, CvColors.BLUE.value, 2)
    cv2.line(img, pt3, pt4, CvColors.BLUE.value, 2)
    cv2.line(img, pt4, pt1, CvColors.BLUE.value, 2)
