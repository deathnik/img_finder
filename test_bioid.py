#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json
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


weights = [0.0169782564225, 0.0221585983632, 0.0593067282694, 0.0545627303568, 0.0549075895622, 0.015538244347,
           0.017070492967, 0.0197276146714, 0.0157261591422, 0.0153504906626, 0.00449816839222, 0.00259302719066,
           0.00296338554002, 0.00274542542805, 0.00332384453378, 0.00405939534904, 0.00366573367647, 0.00448686589904,
           0.00434158841823, 0.00466145110447, 0.00519115011161, 0.00506289582729, 0.00485059343645, 0.00537693366419,
           0.00540762964962, 0.0058485873084, 0.00581975279907, 0.00577739869408, 0.00569606758394, 0.00648683061947,
           0.00576494355153, 0.00614100337206, 0.00694179257182, 0.00690055843222, 0.00649945363931, 0.00518230800785,
           0.00700607166371, 0.00725173394165, 0.00731515878506, 0.00749755357027, 0.00844257548527, 0.00825092925175,
           0.00809471120342, 0.00843233831404, 0.00826047443184, 0.00770729336399, 0.00756560036638, 0.00847302169046,
           0.00850915474747, 0.00983163751109, 0.00907410497489, 0.00903450835608, 0.00928227976846, 0.00843434148317,
           0.00906655691145, 0.00861802428838, 0.00859533263239, 0.0101466662724, 0.00986972521745, 0.00807214589166,
           0.00783873645996, 0.00849279502811, 0.00803303694791, 0.0090796210423, 0.00952004749745, 0.00937179104741,
           0.00874636378951, 0.00923073164294, 0.0107372202481, 0.00933115938445, 0.00813113414481, 0.0067688328794,
           0.00669631458161, 0.00668944980287, 0.00663880575154, 0.0068251694296, 0.00712532661614, 0.00703785927763,
           0.00775159650853, 0.0070441222305, 0.00654853127586, 0.00642739349065, 0.00658302942491, 0.00709639007507,
           0.00609917903026, 0.00600189616256, 0.00669664216406, 0.00513804224913, 0.00581234085949, 0.0052934876224,
           0.00547740116271, 0.00659482149237, 0.00552471213982, 0.00561123369963, 0.00625444520039, 0.00477475141225,
           0.00616531005815, 0.00485194228073, 0.00551586978243, 0.00594302908044, 0.00490983081661, 0.00343866254331,
           0.00434361679932, 0.00351519131779, 0.00352409319206, 0.00335659020353, 0.00277816217353, 0.00289757896802,
           0.00294476684136, 0.00253119296092, 0.00285824620209, 0.00377232880157, 0.00322783778814, 0.0026515643848,
           0.00277553574341, 0.00364616149072, 0.00289159804442, 0.00317434009561, 0.00254861810738, 0.00316538587003,
           0.00274854395587, 0.00265606182453, 0.00193290391668, 0.0023231283608, 0.00155708244647, 0.00186235612981,
           0.00205882929168, 0.00131945012523, 0.00151060432658, 0.00150949841031, 0.00146613524598, 0.00122622314119,
           0.00133821089055, 0.00155121591165, 0.00164685381643, 0.0012374358224, 0.00117084522052, 0.00103379353686,
           0.000953332619716]


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
        dist = abs(sum([w * abs(x - y) for w, (x, y) in zip(weights, zip(orig_desc, descriptor_value))]))
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
        show(ind, (x_pos * scale[0], y_pos * scale[1]), ((x_pos + 1) * scale[0], (y_pos + 1) * scale[1]))
        print dist, ind
        print ''


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


def process_data(descriptor):
    _dir = '/home/deathnik/Documents/magistratura/bioid/points_20'
    for fname in os.listdir(_dir):
        with open(os.path.join(_dir, fname), 'rb') as f:
            data = f.read(-1)
            coords = [map(float, co.split(' ')) for co in data.split('\n')[3:-2]]
            nose_tip = coords[14]
            chin = coords[19]
            eye = coords[0]
            mouse = (coords[2][0] + coords[3][0]) / 2, (coords[2][1] + coords[3][1]) / 2

            img_path = os.path.join('/home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2/',
                                    'BioID_{}.pgm'.format(fname.split('_')[1].split('.')[0]))
            img = cv2.imread(img_path, 1)
            if img is None:
                continue
            statistics = [float(img.min()), float(img.max()), img.mean(), img.std(), img.var()]
            # print type(int(img.min()))
            try:
                hist = calc(descriptor, img_path, nose_tip)
                yield statistics + list(descr), 1
            except:
                print 'failed to cals nose_tip for', img_path

            try:
                descr = calc(descriptor, img_path, chin)
                yield statistics + list(descr), 2
            except:
                print 'failed to cals chin for', img_path
            try:
                descr = calc(descriptor, img_path, mouse)
                yield statistics + list(descr), 3
            except:
                print 'failed to cals mouse for', img_path

            try:
                descr = calc(descriptor, img_path, eye)
                yield statistics + list(descr), 4
            except:
                print 'failed to cals mouse for', img_path


def prepare_viborka(descr, place_to_save):
    with open(place_to_save, 'wb') as f:
        for desc, _class in process_data(HistDescriptor()):
            _str = json.dumps({'desc': list(desc), 'class': _class})
            f.write("{}\n".format(_str))


def main():
    #prepare_viborka(HistDescriptor(), '/home/deathnik/hist_viborka')
    #return
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
    DBConfig.create_for_path(db_path, '.pgm', descriptor_type='.hist', sizes=[size])

    desc_db = DescriptorsDB(db_location=db_path)
    # desc_db.cfg.images = [
    #     'BioID_1080.pgm'
    # ]
    desc_db.make_all_descriptors(force=True)
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
