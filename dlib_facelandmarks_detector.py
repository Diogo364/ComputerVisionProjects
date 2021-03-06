# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import DlibLandmarkDetectorImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-m', '--model', type=str, default='./resource/shape_predictor_68_face_landmarks.dat', help='Path to model weights')
    ap.add_argument('-u', '--num_upsamples', type=int, default=1, help='Confidence for Face detection')
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    args = vars(ap.parse_args())
    
    net = DlibLandmarkDetectorImageTransformation(args['model'])
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=net, video=False, transformation_kw=dict(num_upsamples=args['num_upsamples']))
    else:
        AppController(source=0, image_transformation=net, transformation_kw=dict(num_upsamples=args['num_upsamples']))