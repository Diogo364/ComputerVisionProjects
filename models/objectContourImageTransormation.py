from typing import List
import cv2
import numpy as np
import numpy.typing as npt
from .interface import ModelInterface, ImageTransformationInterface


class ObjectContourImageTransformation(ImageTransformationInterface):
    """It implements image transformation interface to extract contour from image.
    """        

    def __call__(self, image: npt.ArrayLike, lower_threshold: int=30, higher_threshold: int=155) -> List[npt.ArrayLike]:
        gray_image = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        contour_image = cv2.Canny(gray_image, lower_threshold, higher_threshold)
        processed_contour_image = cv2.bitwise_and(image, image, mask=contour_image)
        return [processed_contour_image]
