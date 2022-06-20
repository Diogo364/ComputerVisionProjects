from abc import ABC, abstractmethod

class ModelInterface(ABC):
    
    @abstractmethod
    def predict(self):
        pass
    
    @abstractmethod
    def prepocess(self):
        pass
    
    @abstractmethod
    def load_model(self):
        pass

    @property
    @abstractmethod
    def model(self):
        pass
