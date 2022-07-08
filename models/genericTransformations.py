import cv2
import numpy as np
from numpy import typing as npt

class GenericTransformations():
    @staticmethod
    def smart_resize(image: npt.ArrayLike, size: int=500, height: bool=True) -> npt.ArrayLike:
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
        
    @staticmethod
    def sort_contours(pts: npt.ArrayLike, left_right: bool=True) -> npt.ArrayLike:
        """Sort contour list from its minimum position.

        Args:
            pts (npt.ArrayLike): Array of contours.
            left_right (bool, optional): Orders from left to right if True. Orders from top to bottom if False. Defaults to True.

        Returns:
            npt.ArrayLike: Ordered contour list.
        """        
        order_element = 0 if left_right else 1
        return sorted(pts, key=lambda x: x.min(axis=0).flatten()[order_element])

    @staticmethod
    def sort_rectangle_pts(pts: npt.ArrayLike) -> npt.ArrayLike:
        """Order rectangle data points into (top-left, top-right, bottom-right, bottom-left).
            Same implementation as `order_points` in https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

        Args:
            cnts (npt.ArrayLike): Array containing 4 coordinate points.

        Returns:
            npt.ArrayLike: Ordered coordinates (top-left, top-right, bottom-right, bottom-left).
        """        
        rect = np.zeros((4, 2), dtype=np.float32)
        
        s = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        
        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        
        return rect