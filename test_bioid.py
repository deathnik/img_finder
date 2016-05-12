#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import os
import cv2
from database import DescriptorsDB
from database.db import DBConfig
from features import LocalBinaryPatternsDescriptor, crop_image
from features import HistDescriptor
from features.helpers import Heap
from main_module.main import BoundingBox

db_path = '//home/deathnik/Documents/magistratura/bioid/part'


def imread(path, normalize=True):
    pass


def show(ind, upper, lower):
    img = cv2.imread(db_path + '/' + ind + '.pgm')
    cv2.rectangle(img, upper, lower, color=[110, 50, 50])
    # print upper, lower
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def search(descriptors_db, img_template, upper, lower, heap_size=3):
    # print upper, lower
    bound = BoundingBox(map(float, [upper[0], upper[1], upper[0], lower[1], lower[0], lower[1], lower[0], upper[1]]))

    img_path = img_template.format('BioID_0000')
    lbp = LocalBinaryPatternsDescriptor()
    hist = HistDescriptor()
    p = np.ndarray((4, 2), buffer=bound.data(), dtype=float)
    img = cv2.imread(img_path, 1)
    orig_img_part = crop_image(p[0][0], p[0][1], p[2][0], p[2][1], img)
    # orig_lbp_desc = lbp.calculate_descriptor(orig_img_part)
    # orig_hist_desc = hist.calculate_descriptor(orig_img_part)

    area = bound.area()
    # finding closest scale. we sup closest scale - scale with closest area size
    scale = min(descriptors_db.cfg.sizes, key=lambda x: abs(area - x[0] * x[1]))
    center = bound.center()
    descriptor_coordinates = (center / scale).astype(np.int)

    orig_desc = descriptors_db.calculate_one_descriptor(img, scale, descriptor_coordinates[0],
                                                        descriptor_coordinates[1])
    heap = Heap(heap_size)
    for ind, descriptor_value, (x_pos, y_pos) in descriptors_db.get_descriptors(scale, descriptor_coordinates,
                                                                                bounding_size=3):
        descriptor_value = np.asarray(descriptor_value)
        dist = abs(sum([abs(x - y) for x, y in zip(orig_desc, descriptor_value)]))
        ind = ind.split('.')[0]
        heap.push((- dist, ind, (x_pos, y_pos), descriptor_value))
    show_u = tuple(descriptor_coordinates * scale)
    show_l = tuple(descriptor_coordinates * scale + scale)
    show('BioID_0000', show_u, show_l)
    for dist, ind, (x_pos, y_pos), descriptor_value in reversed(sorted(heap.data())):
        # check descriptor is ok
        img = cv2.imread(img_template.format(ind), 1)
        descriptor_value2 = descriptors_db.calculate_one_descriptor(img, scale, x_pos, y_pos)
        s = sum(descriptor_value2 - descriptor_value)
        if abs(s) > 0.0001:
            raise Exception('wrong descr loaded')
        print dist, ind
        show(ind, (x_pos * scale[0], y_pos * scale[1]), ((x_pos + 1) * scale[0], (y_pos + 1) * scale[1]))


points = {
    0: "right eye pupil",
    1: "left eye pupil",
    2: "right mouth corner",
    3: "left mouth corner",
    4: "outer end of right eye brow",
    5: "inner end of right eye brow",
    6: "inner end of left eye brow",
    7: "outer end of left eye brow",
    8: "right temple",
    9: "outer corner of right eye",
    10: "inner corner of right eye",
    11: "inner corner of left eye",
    12: "outer corner of left eye",
    13: "left temple",
    14: "tip of nose",
    15: "right nostril",
    16: "left nostril",
    17: "centre point on outer edge of upper lip",
    18: "centre point on outer edge of lower lip",
    19: "tip of chin"
}


def calc(descr, img_path, coords, size=[32, 32]):
    img = cv2.imread(img_path, 1)
    x, y = coords
    x_move, y_move = [_ / 2 for _ in size]
    part = crop_image(x - x_move, y - y_move, x + x_move, y + y_move, img)
    return descr.calculate_descriptor(part)


def process_data(descr):
    _dir = '/home/deathnik/Documents/magistratura/bioid/points_20'
    for fname in os.listdir(_dir):
        with open(os.path.join(_dir, fname), 'rb') as f:
            data = f.read(-1)
            coords = [map(float, co.split(' ')) for co in data.split('\n')[3:-2]]
            nose_tip = coords[14]
            chin = coords[19]
            mouse = (coords[2][0] + coords[3][0]) / 2, (coords[2][1] + coords[3][1]) / 2

            img_path = os.path.join('/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/',
                                    'BioID_{}.pgm'.format(fname.split('_')[1].split('.')[0]))

            yield calc(descr, img_path, nose_tip), 1
            yield calc(descr, img_path, chin), 2
            yield calc(descr, img_path, mouse), 3
            return

def main():
    for i in process_data(HistDescriptor()):
        print i
    return
    # img = cv2.imread('/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/BioID_0000.pgm')
    # print img.shape
    # return
    # #print img
    # part = crop_image(160, 160, 176, 176, img)
    # detector = cv2.ORB()
    # kp1, des1  = detector.detectAndCompute(part,None)
    # surfDetector = cv2.FeatureDetector_create("SURF")
    # surfDescriptorExtractor = cv2.DescriptorExtractor_create("SURF")
    # keypoints = surfDetector.detect(img)
    # (keypoints, descriptors) = surfDescriptorExtractor.compute(img,keypoints)
    # print keypoints, descriptors
    # return
    # kp1, des1 = sift.detectAndCompute(part, None)
    # return
    size = (32, 32)
    global db_path
    # db_path = '/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2'

    DBConfig.default_img_size = [286, 384]
    DBConfig.create_for_path(db_path, '.pgm', descriptor_type='.lbp', sizes=[size])

    desc_db = DescriptorsDB(db_location=db_path)
    # desc_db.cfg.images = [
    #     'BioID_1080.pgm'
    # ]
    # desc_db.make_all_descriptors(force=True)
    # descrs = desc_db.get_descriptors([16, 16], [12, 11], bounding_size=1)
    # for _, desc, ind in descrs:
    #    print ind, np.asarray(desc)
    # return
    # return
    # desc_db.make_all_descriptors(force=True)

    # print 'staring search'

    default_pos = (5, 5)
    movex = 0
    movey = 0
    search(descriptors_db=desc_db,
           img_template='/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/{}.pgm',
           upper=((default_pos[0] + movex) * size[0], (default_pos[1] + movey) * size[1]),
           lower=((default_pos[0] + movex + 1) * size[0], (default_pos[1] + movey + 1) * size[1]),
           heap_size=20)

    # for i in xrange(-6, 5):
    #     for j in xrange(-6, 5):
    #         default_pos = (10, 10)
    #         movex = i
    #         movey = j
    #         print '!!!!', i, j
    #         search(descriptors_db=desc_db,
    #                img_template='/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/{}.pgm',
    #                upper=((default_pos[0] + movex) * size[0], (default_pos[1] + movey) * size[1]),
    #                lower=((default_pos[0] + movex + 1) * size[0], (default_pos[1] + movey + 1) * size[1]),
    #                heap_size=20)

    return


if __name__ == "__main__":
    main()
