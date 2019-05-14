import pandas as pd

from abc import ABCMeta, abstractmethod
from typing import List, Dict
from structure import Ensemble, Dataset
from collections import namedtuple


class Experiment(metaclass=ABCMeta):
    def __init__(self):
        self.ensembles: List[Ensemble] = []
        self.results: List[any] = []

    def reset(self):
        self.ensembles = []
        self.results = []

    @abstractmethod
    def run(self, train_data: List[Dataset], val_data: List[Dataset]):
        raise NotImplementedError

    def add_result(self, **kwargs):
        result = {**kwargs}
        self.results.append(result)

    def to_csv(self, filepath: str):
        if len(self.results) == 0:
            raise ValueError(
                'Results are empty. Try "run" before saving to csv.')

        df = pd.DataFrame(self.results)
        df.to_csv(filepath)
