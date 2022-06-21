from .imageController import VideoController, ImageController

class AppController:
    def __init__(self, source, image_transformation, video=True, confidence=0.5):
        if not video:
            ImageController(source, image_transformation, transformation_kw=dict(confidence=confidence))
        else:
            VideoController(0, image_transformation, transformation_kw=dict(confidence=confidence))