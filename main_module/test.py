#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict
import logging
from main_module.main import ImageDB

BASE_PATH = '/home/deathnik/src/my/magister/webcrdf-testbed/webcrdf-testbed/data/datadb.segmxr/'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workdir', default='', help='temp dir')
    return parser.parse_args()


def check_hit(db, img_names, x, y, sizes):
    img = img_names[0]
    upper_corner = x * sizes[0] / 8.0, y * sizes[1] / 8.0
    lower_corner = (x + 1) * sizes[0] / 8.0, (y + 1) * sizes[1] / 8.0
    path = '{}{}.png'.format(BASE_PATH, img[5:])

    try:
        res = list(db.do_magic_v2(path, upper_corner, lower_corner))
        res = set([x[-1][:-4] for x in res])
        if any(map(lambda x: x[5:] in res, img_names[1:])):
            print "Got match", sorted(img_names), sorted(res)
        else:
            print "No match", sorted(img_names), sorted(res)
    except:
        logging.exception('Ooops!:')


def prepare_matches(descr_path, match_cell_height, match_cell_width):
    d = defaultdict(list)
    with open(descr_path, 'rb') as f:
        for line in f:
            img, x, y = line.strip('\n').split(',')
            d['{}_{}'.format(int(x) / match_cell_height, int(y) / match_cell_width)].append(img)
    for k, v in d.iteritems():
        if len(v) > 1:
            yield k, v


def main():
    db = ImageDB()
    descr_path = '/home/deathnik/Documents/magistratura/@data_xray/CLNDAT_EN_COORDS.txt'
    sizes = (160, 160)
    matches = list(prepare_matches(descr_path, sizes[0], sizes[1]))
    for coords, img_names in matches:
        x, y = map(int, coords.split('_'))
        check_hit(db, img_names, x, y, sizes)


if __name__ == "__main__":
    main()
