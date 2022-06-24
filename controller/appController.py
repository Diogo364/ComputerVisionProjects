from typing import Union
from models.interface import ImageTransformationInterface
from .imageController import VideoController, ImageController

class AppController:
    """Aplication Controller that chooses between the VideoController and ImageController.

    Args:
        source (int): Path to image or Video Source.
        image_transformation (ImageTransformationInterface, optional): Image Transformation pipeline to be applied.
        video (bool, optional): Boolean to load video from source. Defaults to True.
        transformation_kw (dict, optional): Extra variable arguments to the image_transformation pipeline. Defaults to {}.
    """        
    def __init__(self, source: Union[str, int], image_transformation: ImageTransformationInterface, video: bool=True, transformation_kw: dict={}):
        if not video:
            ImageController(source, image_transformation, transformation_kw=transformation_kw)
        else:
            VideoController(0, image_transformation, transformation_kw=transformation_kw)