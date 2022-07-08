from typing import Union
import cv2
import numpy as np
import numpy.typing as npt
from .interface import ImageTransformationInterface
from .objectContourImageTransormation import ObjectContourImageTransformation
from .genericTransformations import GenericTransformations


class ObjectMeasureImageTransformation(ImageTransformationInterface):
    """It measures all objects in the image based on a known size from the object at the left.
    """        
    def __init__(self):
        self._pixel_ratio = None

    @staticmethod
    def _get_middle(ptA: npt.ArrayLike, ptB: npt.ArrayLike) -> npt.ArrayLike:
        """Find the middle 2D point between ptA and ptB

        Args:
            ptA (npt.ArrayLike): Firt point (1d array)
            ptB (npt.ArrayLike): Second point (1d array)

        Returns:
            npt.ArrayLike: Coordinates of the middle point.
        """        
        return np.asarray([ptA[0] + ptB[0], ptA[1] + ptB[1]]) * 0.5

    @staticmethod
    def __extract_rectangles(contour: npt.ArrayLike):
        """From a given contour, extract the minimum coords that forms a rectangle.

        Args:
            contour (npt.ArrayLike): Input contours.

        Returns:
            npt.ArrayLike: Rectangle coordinates.
        """        
        box = cv2.minAreaRect(contour)
        box = cv2.boxPoints(box)
        return np.asarray(box, dtype='int')

    @staticmethod
    def __draw_at_vertices(image: npt.ArrayLike, box: npt.ArrayLike):
        """Draw a circle at each point in the box.

        Args:
            image (npt.ArrayLike): Input BGR image.
            box (npt.ArrayLike): (x, y) coordinates.
        """        
        for x, y in box:
            cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    
    def _extract_interest_points(self, box: npt.ArrayLike) -> tuple[tuple]:
        """Extract middle point from each rectangle vertice.

        Args:
            box (npt.ArrayLike): Coordinates from rectangle.

        Returns:
            tuple[tuple]: (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)
        """        
        tl, tr, br, bl = GenericTransformations.sort_rectangle_pts(box)

        (tltrX, tltrY) = self._get_middle(tl, tr)
        (blbrX, blbrY) = self._get_middle(bl, br)
        (tlblX, tlblY) = self._get_middle(tl, bl)
        (trbrX, trbrY) = self._get_middle(tr, br)
        return (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY)

    def set_pixel_ratio(self, known_size: float, pts:npt.ArrayLike):
        """Uses the know size and the arc lenght to define the cm/pixel ratio.

        Args:
            known_size (float): Object known size (in cm).
            pts (npt.ArrayLike): Points do calculate arc length.
        """        
        pixel_size = cv2.arcLength(pts, False)
        self._pixel_ratio = known_size/pixel_size
    
    def _measure_object(self, image: npt.ArrayLike, ptA: npt.ArrayLike, ptB: npt.ArrayLike) -> float:
        """Calculate object size based on cm/pixel ratio.

        Args:
            image (npt.ArrayLike): BGR image.
            ptA (npt.ArrayLike): First point to calculate arc length.
            ptB (npt.ArrayLike): Second point to calculate arc length.

        Returns:
            float: Size between ptA and ptB in cm.
        """        
        pixel_size = cv2.arcLength(np.asarray([ptA, ptB]), False)
        return pixel_size * self._pixel_ratio

    
    def object_perspective_measurement(self, image: npt.ArrayLike, pts: npt.ArrayLike, width: bool, known_size: float):
        """High level function to extract interaction points and measure the object.

        Args:
            image (npt.ArrayLike): BGR image.
            pts (npt.ArrayLike): Object contour.
            width (bool): Using width to define cm/pixel ratio. If False use height.
            known_size (float): Size of the known object.
        """        
        (tltrX, tltrY), (blbrX, blbrY), (tlblX, tlblY), (trbrX, trbrY) = self._extract_interest_points(pts)
        cv2.circle(image, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        cv2.circle(image, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)
        cv2.line(image, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(0, 0, 255), 2)
        cv2.line(image, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(0, 0, 255), 2)

        if self._pixel_ratio is None:
            if width:
                self.set_pixel_ratio(known_size, np.asarray([[tlblX, tlblY], [trbrX, trbrY]]))
            else:
                self.set_pixel_ratio(known_size, np.asarray([[tltrX, tltrY], [blbrX, blbrY]]))

        measured_w = self._measure_object(image, (tlblX, tlblY), (trbrX, trbrY))
        measured_h = self._measure_object(image, (tltrX, tltrY), (blbrX, blbrY))

        cv2.putText(image, "{:.1f}cm".format(measured_w), (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(image, "{:.1f}cm".format(measured_h), (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        

    def __call__(self, image: npt.ArrayLike, known_size: float=10.0, width=False) -> list[npt.ArrayLike]:
        image = GenericTransformations.smart_resize(image, size=600, height=False)
        processed_image = cv2.GaussianBlur(image, (7, 7), 1)
        processed_image = cv2.erode(processed_image, None, iterations=1)
        edges_image = ObjectContourImageTransformation()(processed_image)[0]
        
        gray_scaled_edges = cv2.cvtColor(edges_image, cv2.COLOR_BGR2GRAY)
        

        cnts, _ = cv2.findContours(gray_scaled_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sorted_cnts = GenericTransformations.sort_contours(cnts)
        
        for contour in sorted_cnts:
            if cv2.contourArea(contour) < 100:
                continue
        
            box = self.__extract_rectangles(contour)
            cv2.drawContours(image, [box], -1, (255, 130, 0), 2)
            self.__draw_at_vertices(image, box)
            self.object_perspective_measurement(image, box, width, known_size)

        return [image]
