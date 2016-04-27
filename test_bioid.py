#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
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
    print upper, lower
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def search(descriptors_db, img_path, upper, lower, heap_size=3):
    print upper, lower
    bound = BoundingBox(map(float, [upper[0], upper[1], upper[0], lower[1], lower[0], lower[1], lower[0], upper[1]]))

    lbp = LocalBinaryPatternsDescriptor()
    hist = HistDescriptor()
    p = np.ndarray((4, 2), buffer=bound.data(), dtype=float)
    img = cv2.imread(img_path, 1)
    orig_img_part = crop_image(p[0][0], p[0][1], p[2][0], p[2][1], img)
    #orig_lbp_desc = lbp.calculate_descriptor(orig_img_part)
    #orig_hist_desc = hist.calculate_descriptor(orig_img_part)

    area = bound.area()
    # finding closest scale. we sup closest scale - scale with closest area size
    scale = min(descriptors_db.cfg.sizes, key=lambda x: abs(area - x[0] * x[1]))
    center = bound.center()
    descriptor_coordinates = (center / scale).astype(np.int)

    orig_desc = descriptors_db.calculate_one_descriptor(img, scale, descriptor_coordinates[0],
                                                        descriptor_coordinates[1])
    heap = Heap(heap_size)
    for ind, descriptor_value, (x_pos, y_pos) in descriptors_db.get_descriptors(scale, descriptor_coordinates,
                                                                                bounding_size=1):
        descriptor_value = np.asarray(descriptor_value)
        dist = abs(np.linalg.norm(orig_desc - descriptor_value))
        ind = ind.split('.')[0]
        heap.push((- dist, ind, (x_pos, y_pos)))
    show_u = tuple(descriptor_coordinates * scale)
    show_l = tuple(descriptor_coordinates * scale + scale)
    show('BioID_0000', show_u, show_l)
    for desc_value, ind, (x_pos, y_pos) in sorted(heap.data()):
        print desc_value, ind, (x_pos, y_pos)
        show(ind, (x_pos * scale[0], y_pos * scale[1]), ((x_pos + 1) * scale[0], (y_pos + 1) * scale[1]))


def main():
    # img = cv2.imread('/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/BioID_0000.pgm', 1)
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
    size = (16, 16)
    global db_path
    db_path = '/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2'

    DBConfig.create_for_path(db_path, '.pgm', descriptor_type='.hist', sizes=[size])

    desc_db = DescriptorsDB(db_location=db_path)
    # desc_db.make_all_descriptors(force=True)
    # print 'staring search'

    default_pos = (10, 10)
    movex = 2
    movey = 1
    search(descriptors_db=desc_db,
           img_path='/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/BioID_0000.pgm',
           upper=((default_pos[0] + movex) * size[0], (default_pos[1] + movey) * size[1]),
           lower=((default_pos[0] + movex + 1) * size[0], (default_pos[1] + movey + 1) * size[1]),
           heap_size=20)

    return


if __name__ == "__main__":
    main()
