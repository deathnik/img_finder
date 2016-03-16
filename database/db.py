import json
import logging
import os
import struct
from sys import modules
import cv2
from features import HistDescriptor, LocalBinaryPatternsDescriptor, crop_image

try:
    module = modules['main']
except KeyError:
    import main

ALLOWED_DESCRIPTOR_TYPES = {
    'hist': HistDescriptor,
    'lbp': LocalBinaryPatternsDescriptor
}


class DBConfig(object):
    def __init__(self, path):
        self.img_size = [256, 256]
        self.sizes = []
        self.images = []
        self.descriptor_suffix = ''
        self.descriptor_type = 'hist'
        with open(path, 'rb') as _input:
            data = json.loads(_input)
            self.sizes = data['sizes']
            self.images = data['images']
            if 'descriptor_suffix' in data:
                self.descriptor_suffix = data['descriptor_suffix']
            if 'descriptor_type' in data:
                if data['descriptor_type'].lower() not in ALLOWED_DESCRIPTOR_TYPES:
                    logging.info(
                        'No such descriptor as {}  registered, falling back to {}'.format(data['descriptor_type'],
                                                                                          self.descriptor_type))
                else:
                    self.descriptor_type = data['descriptor_type'].lower()
            if 'img_size' in data:
                self.img_size = data['img_size']


class DB(object):
    def __init__(self):
        self.images = dict()
        self.cfg = None
        self.load_metainfo()

    def load_metainfo(self):
        db_path = os.path.join(main.DATABASE_LOCATION, 'db.txt')
        self.cfg = DBConfig(db_path)
        for image_name in self.cfg.images:
            image_location = os.path.join(main.DATABASE_LOCATION, image_name) + self.cfg.descriptor_suffix
            self.images.update({image_location: os.path.exists(image_location)})

    def calculate_descriptors(self, img_path):
        descriptor = ALLOWED_DESCRIPTOR_TYPES[self.cfg.descriptor_suffix]()
        img = cv2.imread(img_path, 1)
        for w, h in self.cfg.sizes:
            calculated_descriptors = []
            for i in range(0, img.shape[0] / w):
                for j in range(0, img.shape[1] / h):
                    img_part = crop_image(w * i, h * j, w * (i + 1), h * (j + 1), img)
                    calculated_descriptors.append(descriptor.calculate_descriptor(img_part))
            yield w, h, calculated_descriptors

    def make_descriptors(self, img_path):
        descriptor_type = ALLOWED_DESCRIPTOR_TYPES[self.cfg.descriptor_suffix]()
        pack_template = '{}f'.format(descriptor_type.size())
        with open(img_path + self.cfg.descriptor_suffix, 'wb') as descrfile:
            for _, _, descriptors in self.calculate_descriptors(img_path):
                for descriptor in descriptors:
                    descrfile.write(struct.pack(pack_template, descriptor))
