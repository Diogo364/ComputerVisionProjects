from typing import List, Tuple
import cv2
import numpy as np
import numpy.typing as npt
from .interface import ModelInterface, ImageTransformationInterface


class RotationImageTransformation(ImageTransformationInterface):
    """It implements image transformation interface to rotate images.

    Args:
        degrees_list (list[int], optional): List containing the rotation degree to be applied to the image. Defaults to [90].
    """        
    def __init__(self, degrees_list: List[int]=[90]):
        self.degrees_list = degrees_list

    def __call__(self, image: npt.ArrayLike, padding: bool=False) -> List[npt.ArrayLike]:
        rotated_images = []
        for rotation_degree in self.degrees_list:
            h, w, _ = image.shape
            im_center = (w//2, h//2)
            image_sample = image.copy()

            rotation_matrix = cv2.getRotationMatrix2D(im_center, rotation_degree, 1.0)
            if padding:
                h, w = self.__add_padding(image_sample, rotation_matrix)
            image_sample = cv2.warpAffine(image_sample, rotation_matrix, (w, h))
            text = f"{rotation_degree} degrees rotated"
            cv2.putText(image_sample, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            rotated_images.append(image_sample)
        return rotated_images

    def __add_padding(self, image: npt.ArrayLike, rotation_matrix: npt.ArrayLike) -> Tuple[int]:
        """Transform rotation matrix to apply padding and avoid image cropping (Based on pyimagesearch tutorial https://pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/).

        Args:
            image (npt.ArrayLike): Input Image
            rotation_matrix (npt.ArrayLike): Rotation Matrix generated by cv2.getRotationMatrix2D function.

        Returns:
            tuple[int]: (New height and New Width)
        """        
        cos_value = abs(rotation_matrix[0, 0])
        sin_value = abs(rotation_matrix[0, 1])

        h, w, _ = image.shape
        
        new_h, new_w = self.__calculate_rotated_output_dimensions(h, w, sin_value, cos_value)

        rotation_matrix[0, 2] += (new_w - w) // 2
        rotation_matrix[1, 2] += (new_h - h) // 2

        return new_h, new_w
    
    @staticmethod
    def __calculate_rotated_output_dimensions(h: int, w:int, sin_value:float, cos_value:float) -> Tuple[int]:
        new_h = int((h * cos_value) + (w * sin_value))
        new_w = int((w * cos_value) + (h * sin_value))
        return new_h, new_w