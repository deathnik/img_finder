import json
import logging
import os
import struct
from sys import modules

import cv2

from features import HistDescriptor, LocalBinaryPatternsDescriptor, crop_image

DATABASE_LOCATION = '/home/deathnik/src/my/magister/webcrdf-testbed/webcrdf-testbed/data/datadb.segmxr/'

ALLOWED_DESCRIPTOR_TYPES = {
    '.hist': HistDescriptor,
    '.lbp': LocalBinaryPatternsDescriptor
}


class DBConfig(object):
    def __init__(self, path):
        self.img_size = [256, 256]
        self.sizes = []
        self.images = []
        self.descriptor_suffix = ''
        self.descriptor_type = 'hist'
        with open(path, 'rb') as _input:
            data = json.loads(_input.read(-1))
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


ELEMENT_SIZE = 4


class DescriptorsDB(object):
    def __init__(self):
        self.images = dict()
        self.cfg = None
        db_path = os.path.join(DATABASE_LOCATION, 'db.txt')
        self.cfg = DBConfig(db_path)
        self.load_metainfo()

    def _get_descriptor_location(self, image_name):
        return os.path.join(DATABASE_LOCATION, image_name) + self.cfg.descriptor_suffix

    def _get_descriptor(self):
        return ALLOWED_DESCRIPTOR_TYPES[self.cfg.descriptor_suffix]()

    def load_metainfo(self):
        db_path = os.path.join(DATABASE_LOCATION, 'db.txt')
        self.cfg = DBConfig(db_path)
        for image_name in self.cfg.images:
            image_location = self._get_descriptor_location(image_name)
            self.images.update({image_location: os.path.exists(image_location)})

    def calculate_descriptors(self, img_path):
        descriptor = self._get_descriptor()
        img = cv2.imread(img_path, 1)
        for w, h in self.cfg.sizes:
            calculated_descriptors = []
            for i in range(0, img.shape[0] / w):
                for j in range(0, img.shape[1] / h):
                    img_part = crop_image(w * i, h * j, w * (i + 1), h * (j + 1), img)
                    calculated_descriptors.append(descriptor.calculate_descriptor(img_part))
            yield w, h, calculated_descriptors

    def make_descriptors(self, img_id):
        img_path = DATABASE_LOCATION + img_id
        _descriptor = self._get_descriptor()
        pack_template = '{}f'.format(_descriptor.size())
        with open(img_path + self.cfg.descriptor_suffix, 'wb') as descrfile:
            for _, _, descriptors in self.calculate_descriptors(img_path):
                for descriptor in descriptors:
                    descrfile.write(struct.pack(pack_template, *descriptor))

    def get_descriptors(self, size, position):
        offset = 0
        size = list(size)
        for sz in self.cfg.sizes:
            if sz != size:
                offset += self.cfg.img_size[0] / sz[0] * self.cfg.img_size[1] / sz[1]
            else:
                offset += position[1] * self.cfg.img_size[0] / sz[0] + position[0]
                break
        descriptor_size = self._get_descriptor().size()

        pack_template = '{}f'.format(descriptor_size)
        offset *= descriptor_size * ELEMENT_SIZE
        for image_name in self.cfg.images:
            with open(self._get_descriptor_location(image_name), 'rb') as f:
                f.seek(offset)
                yield image_name, struct.unpack(pack_template, f.read(descriptor_size * ELEMENT_SIZE))
