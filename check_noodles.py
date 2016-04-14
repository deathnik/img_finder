#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--workdir', default='', help='temp dir')
    return parser.parse_args()


def main():
    d = defaultdict(list)
    with open('/home/deathnik/Documents/magistratura/@data_xray/CLNDAT_EN_COORDS.txt', 'rb') as f:
        for line in f:
            img, x, y = line.strip('\n').split(',')
            d['{}_{}'.format(int(x) / 80, int(y) / 80)].append(img)
    for k, v in d.iteritems():
        if len(v) > 1:
            print k,v


if __name__ == "__main__":
    main()
