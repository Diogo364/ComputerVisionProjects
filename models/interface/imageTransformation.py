from abc import ABC, abstractmethod
import numpy.typing as npt

class ImageTransformationInterface(ABC):
    """Image transformation interface supported throughout this project.
    Use it to create any image transformation pipeline to be applied into default Controllers.
    """        
    @abstractmethod
    def __call__(self, image: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
        pass