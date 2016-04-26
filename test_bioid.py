#!/usr/bin/env python
# -*- coding: utf-8 -*-
from database import DescriptorsDB
from database.db import DBConfig


def main():
    test_path = '//home/deathnik/Documents/magistratura/bioid/BioID-FaceDatabase-V1.2'
    DBConfig.create_for_path(test_path, '.pgm')
    desc_db = DescriptorsDB(db_location=test_path)
    desc_db.make_all_descriptors()
    return


if __name__ == "__main__":
    main()
