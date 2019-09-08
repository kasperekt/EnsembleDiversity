import numpy as np
import pandas as pd
import multiprocessing as mp

from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple
from data import Dataset
from collections import namedtuple
from structure import Ensemble
from experiments import ExperimentVariant

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class Experiment(metaclass=ABCMeta):
    def __init__(self, variant: str = ExperimentVariant.SHARED):
        self.results: List = []
        self.EnsembleType: callable = None
        self.param_grid: ParameterGrid = None
        self.shared_param_grid: ParameterGrid = self.build_shared_param_grid()
        self.name = 'Experiment'
        self.variant = variant

    def reset(self):
        self.results = []

    @abstractmethod
    def build_shared_param_grid(self):
        return ParameterGrid({
            'n_estimators': [10, 50, 100],
            'max_depth': [5]
        })

    def add_result(self, **kwargs):
        result = {**kwargs}
        self.results.append(result)

    def to_csv(self, filepath: str):
        if len(self.results) == 0:
            raise ValueError(
                'Results are empty. Try "run" before saving to csv.')

        df = pd.DataFrame(self.results)
        df.to_csv(filepath)

    @abstractmethod
    def process(self, train: Dataset, val: Dataset, params: dict):
        raise NotImplementedError

    @abstractmethod
    def collect(self, payload):
        raise NotImplementedError

    def run(self, datasets: List[Tuple[Dataset, Dataset]], repetitions=1):
        if self.EnsembleType is None:
            raise ValueError('Ensemble type is not specified')

        if self.variant == 'individual' and self.param_grid is None:
            raise ValueError('Param grid is not defined')

        print(f'Running {self.name} experiment...\n\n')

        pool = mp.Pool(mp.cpu_count())

        grid = self.param_grid if self.variant == 'individual' else self.shared_param_grid

        for _ in range(repetitions):
            for params in grid:   # pylint: disable=not-an-iterable
                for train, val in datasets:
                    pool.apply_async(self.process, args=(
                        train, val, params), callback=self.collect)

        pool.close()
        pool.join()
