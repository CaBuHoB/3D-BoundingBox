"""
dataset file
"""
import os
import numpy as np
import cv2

from torchvision import transforms
from torch.utils import data

from library.File import get_p, get_calibration_cam_to_image
from .ClassAverages import ClassAverages


def calc_theta_ray(img, box_2d, proj_matrix):
    """ calculates theta ray """
    width = img.shape[1]
    fovx = 2 * np.arctan(width / (2 * proj_matrix[0][0]))
    center = (box_2d[1][0] + box_2d[0][0]) / 2
    val_dx = center - (width / 2)

    mult = 1
    if val_dx < 0:
        mult = -1
    val_dx = abs(val_dx)
    angle = np.arctan((2*val_dx*np.tan(fovx/2)) / width)
    angle = angle * mult

    return angle

def format_img(img, box_2d):
    """ formats image """
    # Should this happen? or does normalize take care of it. YOLO doesnt like
    # img=img.astype(np.float) / 255

    # torch transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                            std=[0.229, 0.224, 0.225])
    process = transforms.Compose([\
        transforms.ToTensor(),\
        normalize\
    ])

    # crop image
    pt1 = box_2d[0]
    pt2 = box_2d[1]
    crop = img[pt1[1]:pt2[1]+1, pt1[0]:pt2[0]+1]
    crop = cv2.resize(src=crop, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)

    # recolor, reformat
    batch = process(crop)

    return batch

def generate_bins(bins):
    """ generates bins """
    angle_bins = np.zeros(bins)
    interval = 2 * np.pi / bins
    for i in range(1, bins):
        angle_bins[i] = i * interval
    angle_bins += interval / 2 # center of the bin

    return angle_bins

def parse_label(label_path):
    """ parses label """
    buf = []
    with open(label_path, 'r') as file:
        for line in file:
            line = line[:-1].split(' ')

            val_class = line[0]
            if val_class == "DontCare":
                continue

            for i in range(1, len(line)):
                line[i] = float(line[i])

            alpha = line[3] # what we will be regressing
            val_ry = line[14]
            top_left = (int(round(line[4])), int(round(line[5])))
            bottom_right = (int(round(line[6])), int(round(line[7])))
            box_2d = [top_left, bottom_right]

            dimension = [line[8], line[9], line[10]] # height, width, length
            location = [line[11], line[12], line[13]] # x, y, z
            location[1] -= dimension[0] / 2
            buf.append({\
                    'Class': val_class,\
                    'Box_2D': box_2d,\
                    'Dimensions': dimension,\
                    'Location': location,\
                    'Alpha': alpha,\
                    'Ry': val_ry\
                })
    return buf


class Dataset(data.Dataset):
    """ class for data set """
    def __init__(self, path, calib_path, bins=2, overlap=0.1):
        """ initialization """
        self.top_label_path = path + "/label_2/"
        self.top_img_path = path + "/image_2/"
        self.top_calib_path = path + "/calib/"
        # use a relative path instead?

        self.proj_matrix = get_p(calib_path)

        self.ids = [x.split('.')[0] for x in sorted(os.listdir(self.top_img_path))] # name of file
        self.num_images = len(self.ids)

        # create angle bins
        self.bins = bins
        self.angle_bins = np.zeros(bins)
        self.interval = 2 * np.pi / bins
        for i in range(1, bins):
            self.angle_bins[i] = i * self.interval
        self.angle_bins += self.interval / 2 # center of the bin

        self.overlap = overlap
        # ranges for confidence
        # [(min angle in bin, max angle in bin), ... ]
        self.bin_ranges = []
        for i in range(0, bins):
            self.bin_ranges.append(((i*self.interval - overlap) % (2*np.pi),\
                                (i*self.interval + self.interval + overlap) % (2*np.pi)))

        # hold average dimensions
        class_list = ['Car', 'Van', 'Truck', \
            'Pedestrian', 'Person_sitting', 'Cyclist', 'Tram', 'Misc']
        self.averages = ClassAverages(class_list)

        self.object_list = self.get_objects(self.ids)

        # pre-fetch all labels
        self.labels = {}
        last_id = ""
        for obj in self.object_list:
            val_id = obj[0]
            line_num = obj[1]
            label = self.get_label(val_id, line_num)
            if val_id != last_id:
                self.labels[val_id] = {}
                last_id = val_id

            self.labels[val_id][str(line_num)] = label

        # hold one image at a time
        self.curr_id = ""
        self.curr_img = None


    # should return (Input, Label)
    def __getitem__(self, index):
        """ gets item """
        val_id = self.object_list[index][0]
        line_num = self.object_list[index][1]

        if val_id != self.curr_id:
            self.curr_id = val_id
            self.curr_img = cv2.imread(self.top_img_path + '%s.png'%val_id)

        label = self.labels[val_id][str(line_num)]
        # P doesn't matter here
        obj = DetectedObject(\
            self.curr_img, label['Class'], label['Box_2D'], self.proj_matrix, label=label)

        return obj.img, label

    def __len__(self):
        """ gets length """
        return len(self.object_list)

    def get_objects(self, ids):
        """gets object """
        objects = []
        for it_id in ids:
            with open(self.top_label_path + '%s.txt'%it_id) as file:
                for line_num, line in enumerate(file):
                    line = line[:-1].split(' ')
                    obj_class = line[0]
                    if obj_class == "DontCare":
                        continue

                    dimension = np.array(\
                        [float(line[8]), float(line[9]), float(line[10])], dtype=np.double)
                    self.averages.add_item(obj_class, dimension)

                    objects.append((it_id, line_num))


        self.averages.dump_to_file()
        return objects


    def get_label(self, it_id, line_num):
        """ gets object """
        lines = open(self.top_label_path + '%s.txt'%it_id).read().splitlines()
        label = self.format_label(lines[line_num])

        return label

    def get_bin(self, angle):
        """ gets bin """
        bin_idxs = []

        def is_between(val_min, val_max, angle):
            val_max = (val_max - val_min) \
                if (val_max - val_min) > 0 else (val_max - val_min) + 2*np.pi
            angle = (angle - val_min) if (angle - val_min) > 0 \
                else (angle - val_min) + 2*np.pi
            return angle < val_max

        for bin_idx, bin_range in enumerate(self.bin_ranges):
            if is_between(bin_range[0], bin_range[1], angle):
                bin_idxs.append(bin_idx)

        return bin_idxs

    def format_label(self, line):
        """ formates label """
        line = line[:-1].split(' ')

        val_class = line[0]

        for i in range(1, len(line)):
            line[i] = float(line[i])

        val_alpha = line[3] # what we will be regressing
        #Ry = line[14]
        box_2d = [(int(round(line[4])), int(round(line[5]))), \
            (int(round(line[6])), int(round(line[7])))]

        dimension = np.array([line[8], line[9], line[10]], dtype=np.double) # height, width, length
        # modify for the average
        dimension -= self.averages.get_item(val_class)

        location = [line[11], line[12], line[13]] # x, y, z
        location[1] -= dimension[0] / 2 # bring the KITTI center up to the middle of the object

        orientation = np.zeros((self.bins, 2))
        confidence = np.zeros(self.bins)

        # alpha is [-pi..pi], shift it to be [0..2pi]
        angle = val_alpha + np.pi

        bin_idxs = self.get_bin(angle)

        for i in bin_idxs:
            angle_diff = angle - self.angle_bins[i]

            orientation[i, :] = np.array([np.cos(angle_diff), np.sin(angle_diff)])
            confidence[i] = 1

        label = {\
                'Class': val_class,\
                'Box_2D': box_2d,\
                'Dimensions': dimension,\
                'Alpha': val_alpha,\
                'Orientation': orientation,\
                'Confidence': confidence\
                }

        return label

    # will be deprc soon
    def all_objects(self):
        """ all objects """
        val_data = {}
        for it_id in self.ids:
            val_data[it_id] = {}
            img_path = self.top_img_path + '%s.png'%it_id
            img = cv2.imread(img_path)
            val_data[it_id]['Image'] = img

            # using p per frame
            calib_path = self.top_calib_path + '%s.txt'%it_id
            proj_matrix = get_calibration_cam_to_image(calib_path)

            # using P_rect from global calib file
            proj_matrix = self.proj_matrix

            val_data[it_id]['Calib'] = proj_matrix

            label_path = self.top_label_path + '%s.txt'%it_id
            labels = parse_label(label_path)
            objects = []
            for label in labels:
                box_2d = label['Box_2D']
                detection_class = label['Class']
                objects.append(DetectedObject(\
                    img, detection_class, box_2d, proj_matrix, label=label))

            val_data[it_id]['Objects'] = objects

        return val_data


class DetectedObject:
    """ class for detected object """
    def __init__(self, img, detection_class, box_2d, proj_matrix, label=None):
        """ initialization """
        if isinstance(proj_matrix, str): # filename
            proj_matrix = get_p(proj_matrix)
            # proj_matrix = get_calibration_cam_to_image(proj_matrix)

        self.proj_matrix = proj_matrix
        self.theta_ray = calc_theta_ray(img, box_2d, proj_matrix)
        self.img = format_img(img, box_2d)
        self.label = label
        self.detection_class = detection_class
