from abc import ABC, abstractmethod
from typing import Union
import cv2
import numpy.typing as npt
from models.interface import ImageTransformationInterface


class AbstractImageController(ABC):
    @abstractmethod
    def __init__(self, source: Union[int, str], image_transformation: ImageTransformationInterface=None, transformation_kw: dict={}):
        pass

    def show_image(self, image: npt.ArrayLike, wait_key: bool=False) -> int:
        cv2.imshow('Frame', image)
        if wait_key:
            cv2.waitKey(0)
            return -1
        return cv2.waitKey(30) & 0xff
    

class VideoController(AbstractImageController):
    def __init__(self, source: int, image_transformation: ImageTransformationInterface=None, transformation_kw: dict={}):
        cap = cv2.VideoCapture(source)
        while True:
            ret, frame = cap.read()

            if image_transformation is not None:
                frame_list = image_transformation(frame, **transformation_kw)
                frame = frame_list[0]
            
            k = self.show_image(frame, wait_key=False)
            if k == 27:
                break
        cv2.destroyAllWindows()
        cap.release()

class ImageController(AbstractImageController):
    def __init__(self, source: str, image_transformation: ImageTransformationInterface=None, transformation_kw: dict={}):
        image = cv2.imread(source)

        if image_transformation is not None:
                image_list = image_transformation(image, **transformation_kw)
        else:
            image_list = [image]
        
        self.show_image_list(image_list, wait_key=True)

    def show_image_list(self, image_list: list[npt.ArrayLike], wait_key: bool):
        for image in image_list:
            k = self.show_image(image, wait_key)
            cv2.destroyAllWindows()