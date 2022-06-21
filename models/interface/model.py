from abc import ABC, abstractmethod

class ModelInterface(ABC):
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def preprocess(self):
        pass
    
    @abstractmethod
    def load_model(self):
        pass