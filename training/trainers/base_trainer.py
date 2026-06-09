# training/trainers/base_trainer.py

from abc import ABC, abstractmethod

class BaseTrainer(ABC):

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def evaluate(self):
        pass