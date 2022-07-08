# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import ObjectMeasureImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    args = vars(ap.parse_args())

    transformer = ObjectMeasureImageTransformation()
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=transformer, video=False)
    else:
        AppController(source=0, image_transformation=transformer)