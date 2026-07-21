from abc import ABC, abstractmethod

class TextClassifierInterface(ABC):
    
    @abstractmethod
    def load_model(self) -> None:
        """Blueprint method to handle downloading or loading weights into memory."""
        pass

    @abstractmethod
    def predict(self, text: str) -> str:
        """Blueprint method that MUST accept a string and MUST return a string label."""
        pass
    