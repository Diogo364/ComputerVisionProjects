# -*- coding: utf-8 -*-
# @Author: Diogo Telheiro do Nascimento
# @Date:   2022-06-18 20:01:57
# @Last Modified by:   Diogo Telheiro do Nascimento
# @Last Modified time: 2022-06-20 15:18:47

import cv2
import numpy as np

def main():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Video', frame)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()