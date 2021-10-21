"""
Class averages
"""
import os
import json
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """ Enables writing json with numpy arrays to file """
    def default(self, obj):
        """ default """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class ClassAverages:
    """ Class will hold the average dimension for a class, regressed value is the residual """
    def __init__(self, classes=None):
        if classes is None:
            classes = []
        self.dimension_map = {}
        self.filename = os.path.abspath(os.path.dirname(__file__)) + '/class_averages.txt'

        if len(classes) == 0: # eval mode
            self.load_items_from_file()

        for detection_class in classes:
            class_ = detection_class.lower()
            if class_ in self.dimension_map.keys():
                continue
            self.dimension_map[class_] = {}
            self.dimension_map[class_]['count'] = 0
            self.dimension_map[class_]['total'] = np.zeros(3, dtype=np.double)


    def add_item(self, class_, dimension):
        """ Add item"""
        class_ = class_.lower()
        self.dimension_map[class_]['count'] += 1
        self.dimension_map[class_]['total'] += dimension
        # self.dimension_map[class_]['total'] /= self.dimension_map[class_]['count']

    def get_item(self, class_):
        """ Get item """
        class_ = class_.lower()
        return self.dimension_map[class_]['total'] / self.dimension_map[class_]['count']

    def dump_to_file(self):
        """ Dump to file """
        file = open(self.filename, "w")
        file.write(json.dumps(self.dimension_map, cls=NumpyEncoder))
        file.close()

    def load_items_from_file(self):
        """ Load items from file """
        file = open(self.filename, 'r')
        dimension_map = json.load(file)

        for class_ in dimension_map:
            dimension_map[class_]['total'] = np.asarray(dimension_map[class_]['total'])

        self.dimension_map = dimension_map

    def recognized_class(self, class_):
        """ Recognize class """
        return class_.lower() in self.dimension_map
