# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import RotationImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    ap.add_argument('-r', '--rotation_list', nargs="+", type=int, default=[45], help='List of rotation degrees values, separated by blank space. e.g: 10 20 30')
    args = vars(ap.parse_args())

    transformer = RotationImageTransformation(args['rotation_list'])
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=transformer, video=False)
    else:
        AppController(source=0, image_transformation=transformer)