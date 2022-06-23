import cv2
from abc import ABC, abstractmethod


class AbstractImageController(ABC):
    @abstractmethod
    def __init__(self, source, image_transformation=None, transformation_kw={}):
        pass

    def show_image(self, image, wait_key=False):
        cv2.imshow('Frame', image)
        if wait_key:
            cv2.waitKey(0)
            return
        return cv2.waitKey(30) & 0xff
    

class VideoController(AbstractImageController):
    def __init__(self, source, image_transformation=None, transformation_kw={}):
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
    def __init__(self, source, image_transformation=None, transformation_kw={}):
        image = cv2.imread(source)

        if image_transformation is not None:
                image_list = image_transformation(image, **transformation_kw)
        else:
            image_list = [image]
        
        self.show_image_list(image_list, wait_key=True)

    def show_image_list(self, image_list, wait_key):
        for image in image_list:
            k = self.show_image(image, wait_key)
            cv2.destroyAllWindows()