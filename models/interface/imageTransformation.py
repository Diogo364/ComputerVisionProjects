from abc import ABC, abstractmethod

class ImageTransformationInterface(ABC):
    
    @abstractmethod
    def __call__(self, image, **kwargs):
        pass