import cv2
import numpy as np
from .interface import ModelInterface, ImageTransformationInterface


class RotationImageTransformer(ImageTransformationInterface):
    def __init__(self, 
                degrees_list=[90]):
        self.degrees_list = degrees_list

    def __call__(self, image, confidence):
        rotated_images = []
        h, w, _ = image.shape
        im_center = (w//2, h//2)
        for rotation_degree in self.degrees_list:
            image_sample = image.copy()

            rotation_matrix = cv2.getRotationMatrix2D(im_center, rotation_degree, 1.0)
            image_sample = cv2.warpAffine(image_sample, rotation_matrix, (w, h))
            text = f"{rotation_degree} degrees rotated"
            cv2.putText(image_sample, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rotated_images.append(image_sample)
        return rotated_images