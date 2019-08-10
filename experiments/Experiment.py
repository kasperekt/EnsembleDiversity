import numpy as np
import pandas as pd
import multiprocessing as mp

from abc import ABCMeta, abstractmethod
from typing import List, Dict
from data import Dataset
from collections import namedtuple
from enum import Enum
from structure import Ensemble

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class ExperimentVariant(Enum):
    INDIVIDUAL = 'individual'
    SHARED = 'shared'


class Experiment(metaclass=ABCMeta):
    def __init__(self, variant: str = ExperimentVariant.SHARED, cv: bool = False):
        self.results: List[any] = []
        self.EnsembleType: callable = None
        self.param_grid: ParameterGrid = None
        self.shared_param_grid: ParameterGrid = self.build_shared_param_grid()
        self.name = 'Experiment'
        self.variant = variant
        self.cv = cv

    def reset(self):
        self.results = []

    def build_shared_param_grid(self):
        return ParameterGrid({
            'n_estimators': np.arange(5, 100, 5),
            'max_depth': np.arange(2, 8, 1)
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

    def process(self, train: Dataset, val: Dataset, params: dict):
        '''
        Run individual experiment
        '''
        print(f'[{train.name}] Params: {params}')

        ensemble: Ensemble = self.EnsembleType(  # pylint: disable=not-callable
            params)
        ensemble.fit(train)

        preds = ensemble.predict(val)
        accuracy = accuracy_score(val.y, preds)

        return {
            'name': ensemble.name,
            'dataset_name': train.name,
            'accuracy': accuracy,
            **params,
            # Structural Measures
            'node_diversity': ensemble.node_diversity(),
            'coverage_std': ensemble.coverage_leaves_std(),
            'coverage_minmax': ensemble.coverage_leaves_minmax(),
            'used_attributes_ratio': ensemble.used_attributes_ratio(),
            # Behavioral Measures
            'entropy': ensemble.entropy(val),
            'q': ensemble.q(val),
            'df': ensemble.df(val),
            'kw': ensemble.kw(val),
            'corr': ensemble.corr(val)
        }

    def run(self, datasets: List[Dataset]):
        if self.EnsembleType is None:
            raise ValueError('Ensemble type is not specified')

        if self.variant == 'individual' and self.param_grid is None:
            raise ValueError('Param grid is not defined')

        print(f'Running {self.name} experiment...\n\n')

        pool = mp.Pool(mp.cpu_count())

        def collect(result_dict):
            self.add_result(**result_dict)

        for dataset in datasets:
            grid = self.param_grid if self.variant == 'individual' else self.shared_param_grid

            for params in grid:   # pylint: disable=not-an-iterable
                if self.cv:
                    for train, val in dataset.n_splits(10):
                        pool.apply_async(self.process, args=(
                            train, val, params), callback=collect)
                else:
                    train, val = dataset.split(0.2)
                    pool.apply_async(self.process, args=(
                        train, val, params), callback=collect)

        pool.close()
        pool.join()
