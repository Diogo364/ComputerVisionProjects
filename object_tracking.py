# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import ObjectTrackingImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    ap.add_argument('-b', '--buffer_size', default=64, type=int, help='Size of the buffer to define trace lenght.')
    ap.add_argument('-l', '--hsv_min', nargs=3, default=(115, 33, 65), type=int, help='Minimum HSV to object identification. (Define using imutils/bin/range-detector)')
    ap.add_argument('-t', '--hsv_max', nargs=3, default=(174, 174, 248), type=int, help='Maximum HSV to object identification. (Define using imutils/bin/range-detector)')
    args = vars(ap.parse_args())

    transformer = ObjectTrackingImageTransformation(buffer_size=args['buffer_size'], hsv_min=tuple(args['hsv_min']), hsv_max=tuple(args['hsv_max']))
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=transformer, video=False)
    else:
        AppController(source=0, image_transformation=transformer)