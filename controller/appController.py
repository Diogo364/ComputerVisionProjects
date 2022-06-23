from typing import Union
from models.interface import ImageTransformationInterface
from .imageController import VideoController, ImageController

class AppController:
    def __init__(self, source: Union[str, int], image_transformation: ImageTransformationInterface, video: bool=True, transformation_kw: dict={}):
        if not video:
            ImageController(source, image_transformation, transformation_kw=transformation_kw)
        else:
            VideoController(0, image_transformation, transformation_kw=transformation_kw)