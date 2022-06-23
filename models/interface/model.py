from abc import ABC, abstractmethod

class ModelInterface(ABC):
    """Model Interface with methods supported throughout this code project."""
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def preprocess(self):
        pass
    
    @abstractmethod
    def load_model(self):
        pass