# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import DocumentScannerImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    ap.add_argument('-c', dest='smart_crop', action='store_false', help='Deactivate smart crop function.')
    ap.add_argument('-b', dest='binarization', action='store_false', help='Deactivate binarization function.')
    args = vars(ap.parse_args())

    transformer = DocumentScannerImageTransformation()
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=transformer, video=False, transformation_kw=dict(smart_crop=args['smart_crop'], binarization=args['binarization']))
    else:
        AppController(source=0, image_transformation=transformer, transformation_kw=dict(smart_crop=args['smart_crop'], binarization=args['binarization']))