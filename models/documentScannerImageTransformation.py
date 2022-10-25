from typing import Union, List
import cv2
import numpy as np
import numpy.typing as npt
from .objectContourImageTransormation import ObjectContourImageTransformation
from .interface import ModelInterface, ImageTransformationInterface


class DocumentScannerImageTransformation(ImageTransformationInterface):
    """Computer Vision Document Scanner Image Transformation.
    """    
    
    def __init__(self):
        self._int_points = None
    
    def __find_rectangle_contour(self, cnts: npt.ArrayLike) -> Union[npt.ArrayLike, None]:
        """Simplify contour pts and iterate over them searching for a possible rectangle shape (4 pts).add()

        Args:
            cnts (npt.ArrayLike): Contour points array.

        Returns:
            Union[npt.ArrayLike, None]: (top-left, top-right, bottom-right, bottom-left). coordinates. Or None if no coordinates found
        """        
        for cnt in cnts:
            perimeter = cv2.arcLength(cnt, True)
            approx_poly_coords = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
            if len(approx_poly_coords) == 4:
                approx_poly_coords = approx_poly_coords.reshape(4, 2)
                return self.__sort_rectangle_pts(approx_poly_coords)
        return None
           
    @staticmethod
    def __sort_rectangle_pts(pts: npt.ArrayLike) -> npt.ArrayLike:
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
    
    def __find_document_interest_points(self, image: npt.ArrayLike):
        """Searches for rectangle frame using contour filters.add()

        Args:
            image (npt.ArrayLike): BGR image.
        """        
        canny_color = ObjectContourImageTransformation()(image.copy(), 75, 200)[0]
        canny_gray = cv2.cvtColor(canny_color, cv2.COLOR_BGR2GRAY)

        cnts = cv2.findContours(canny_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts[0], key=cv2.contourArea, reverse=True)

        self._int_points = self.__find_rectangle_contour(cnts)
    
    def _generate_perspective_transformation(self, image: npt.ArrayLike) -> npt.ArrayLike:
        """Apply top-down "birds eye view" perspective transformation.
        It frames nicelly the document.

        Args:
            image (npt.ArrayLike): BGR Image

        Returns:
            npt.ArrayLike: Perspective Transformed BGR Image
        """        
        
        tl, tr, br, bl = self._int_points
        
        width_a = cv2.arcLength(np.asarray([br, bl]), False)
        width_b = cv2.arcLength(np.asarray([tr, tl]), False)
        max_width = int(max(width_a, width_b))
        
        height_a = cv2.arcLength(np.asarray([br, bl]), False)
        height_b = cv2.arcLength(np.asarray([tr, tl]), False)
        max_height = int(max(height_a, height_b))

        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]], dtype = "float32")
        
        M = cv2.getPerspectiveTransform(self._int_points, dst)
        return cv2.warpPerspective(image, M, (max_width, max_height))

    def _apply_document_threshold_binarization(self, image: npt.ArrayLike) -> npt.ArrayLike:
        """Apply Gaussian Adaptive Threshold to improve document contrast.

        Args:
            image (npt.ArrayLike): BGR Image

        Returns:
            npt.ArrayLike: Grayscale Binarized Image
        """        
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)

    def __call__(self, image: npt.ArrayLike, smart_crop=True, binarization=True) -> List[npt.ArrayLike]:
        """Apply Document Scanner effect to image.

        Args:
            image (npt.ArrayLike): BGR image
            smart_crop (bool, optional): Apply border-crop and fix inclination. Defaults to True.
            binarization (bool, optional): Apply binarization contrast. Defaults to True.

        Returns:
            list[npt.ArrayLike]: List of Transformed images.
        """        
        
        self.__find_document_interest_points(image)

        if self._int_points is None:
            text = f"Could not find reference contour"
            cv2.putText(image, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return [image]

        processed_image = image.copy()
        
        if smart_crop:
            processed_image = self._generate_perspective_transformation(processed_image)
        
        if binarization:
            processed_image = self._apply_document_threshold_binarization(processed_image)

        return [processed_image]