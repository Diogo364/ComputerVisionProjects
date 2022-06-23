# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 17:52:36

import argparse
from models import CaffeDetectorImageTransformation
from controller import AppController
    
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--prototxt', type=str, default='./resource/deploy.prototxt', help='Path to prototxt file')
    ap.add_argument('-m', '--model', type=str, default='./resource/opencv_face_detector.caffemodel', help='Path to model weights')
    ap.add_argument('-c', '--confidence', type=float, default=0.5, help='Confidence for Face detection')
    ap.add_argument('-i', '--image_source', default=None, help='Path to image')
    args = vars(ap.parse_args())
    
    net = CaffeDetectorImageTransformation(args['prototxt'], args['model'])
    if args['image_source']:
        AppController(source=args['image_source'], image_transformation=net, video=False, confidence=args['confidence'])
    else:
        AppController(source=0, image_transformation=net, confidence=args['confidence'])