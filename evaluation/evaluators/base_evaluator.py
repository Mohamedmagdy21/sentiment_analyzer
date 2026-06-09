from abc import ABC, abstractmethod


class BaseEvaluator(ABC):

    @abstractmethod
    def evaluate(self, dataset_cfg):
        pass