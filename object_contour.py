# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import ObjectContourImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    ap.add_argument('-l', '--lower', type=int, default=30, help='Lower threshold to Canny Contour detector.')
    ap.add_argument('-t', '--higher', type=int, default=155, help='Higher threshold to Canny Contour detector.')
    args = vars(ap.parse_args())

    transformer = ObjectContourImageTransformation()
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=transformer, video=False, transformation_kw=dict(lower_threshold=args['lower'], higher_threshold=args['higher']))
    else:
        AppController(source=0, image_transformation=transformer, transformation_kw=dict(lower_threshold=args['lower'], higher_threshold=args['higher']))