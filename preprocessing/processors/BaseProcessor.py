from abc import ABC, abstractmethod

class BaseProcessor(ABC):

    @abstractmethod
    def run(self):
        pass

