from abc import ABC, abstractmethod
import numpy.typing as npt

class ImageTransformationInterface(ABC):
    
    @abstractmethod
    def __call__(self, image: npt.ArrayLike, **kwargs) -> npt.ArrayLike:
        pass