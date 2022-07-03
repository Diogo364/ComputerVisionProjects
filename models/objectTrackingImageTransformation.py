from collections import deque
import cv2
import numpy as np
from numpy import typing as npt
from .interface import ImageTransformationInterface


class ObjectTrackingImageTransformation(ImageTransformationInterface):
    """Track object based on HSV color range previous defined.

    Args:
        buffer_size (int, optional): Size of the buffer, Defines trace lenght. Defaults to 64.
        hsv_min (tuple, optional): Minimum HSV identified. Defaults to (115, 33, 65).
        hsv_max (tuple, optional): Maximum HSV identified. Defaults to (174, 174, 248).
    """        
    def __init__(self, buffer_size: int=64, hsv_min=(115, 33, 65), hsv_max=(174, 174, 248)):
        self.__buffer_size = buffer_size
        self.__buffer = deque(maxlen=buffer_size)
        self._hsv_range = {
            'min': hsv_min,
            'max': hsv_max
        }
    
    def __smart_resize(self, image: npt.ArrayLike, size: int=500, height: bool=True) -> npt.ArrayLike:
        """Apply aspect ratio resizing.

        Args:
            image (npt.ArrayLike): Input Image
            size (int, optional): New dimension size. Defaults to 500.
            height (bool, optional): If true, the New dimension size is attributed to Heigh. If False, to Width. Defaults to True.

        Returns:
            npt.ArrayLike: Resized image.
        """        
        h, w, _ = image.shape
        ratio = h/w
        if height:
            new_h = size
            new_w = int(w * (new_h / h))
        else:
            new_w = size
            new_h = int(h * (new_w / w))
        return cv2.resize(image, (new_w, new_h))
        
    def _preprocess_image(self, image: npt.ArrayLike) -> npt.ArrayLike:
        """Apply preprocessing image transformations.

        Args:
            image (npt.ArrayLike): BGR input image

        Returns:
            npt.ArrayLike: Transformed image.
        """        
        processed_image = cv2.GaussianBlur(image, (11, 11), 0)
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
        return processed_image

    def _search_image_for_setted_hsv_range(self, image: npt.ArrayLike) -> npt.ArrayLike:
        """Searches for HSV range regions within HSV image.

        Args:
            image (npt.ArrayLike): HSV image.

        Returns:
            npt.ArrayLike: Mask of identified regions.
        """        
        mask = cv2.inRange(image, self._hsv_range['min'], self._hsv_range['max'])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        return mask

    def _define_object_interest_points(self, image: npt.ArrayLike, cnts: npt.ArrayLike):
        """Identify object in the input image by searching for the minimum circle within the input contours.

        Args:
            image (npt.ArrayLike): BGR image.
            cnts (npt.ArrayLike): Contours Array.
        """        
        larger_cnt = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(larger_cnt)
        M = cv2.moments(larger_cnt)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        if radius > 10:
            cv2.circle(image, (int(x), int(y)), int(radius), (0, 0, 255), 4)
            cv2.circle(image, center, 5, (0, 255, 255), -1)
        self.__buffer.appendleft(center)

    def _draw_buffer_line_trace(self, image: npt.ArrayLike):
        """Uses the buffer to draw the object's center trace line.

        Args:
            image (npt.ArrayLike): BGR image
        """        
        for i in range(1, len(self.__buffer)):
            if self.__buffer[i-1] is None or self.__buffer[i] is None:
                continue
            thickness = int(np.sqrt(self.__buffer_size / float(i + 1)) * 2.5)
            cv2.line(image, self.__buffer[i-1], self.__buffer[i], (255, 0, 0), thickness)

    def __call__(self, image: npt.ArrayLike):
        center = None
        
        resized_image = self.__smart_resize(image, 500, height=False)
        processed_image = self._preprocess_image(resized_image)
        mask = self._search_image_for_setted_hsv_range(processed_image)
        
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(cnts) > 0:
            self._define_object_interest_points(resized_image, cnts)    
            self._draw_buffer_line_trace(resized_image)
            
        return [resized_image]