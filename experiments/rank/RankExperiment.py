import numpy as np
import pandas as pd
import multiprocessing as mp

from abc import ABCMeta, abstractmethod
from typing import List, Dict
from data import Dataset
from collections import namedtuple
from structure import Ensemble
from experiments import ExperimentVariant

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class RankExperiment(metaclass=ABCMeta):
    def __init__(self, variant: str = ExperimentVariant.SHARED, cv: bool = False):
        self.results: List[any] = []
        self.EnsembleType: callable = None
        self.param_grid: ParameterGrid = None
        self.shared_param_grid: ParameterGrid = self.build_shared_param_grid()
        self.name = 'Experiment'
        self.variant = variant
        self.cv = cv
        self.repetitions = 10

    def reset(self):
        self.results = []

    def build_shared_param_grid(self):
        return ParameterGrid({
            'n_estimators': [10, 50, 100],
            'max_depth': [6]
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

        results = []

        for idx, tree in enumerate(ensemble.trees):
            preds = tree.predict(val.X, labeled_result=True)
            accuracy = accuracy_score(val.y, preds)

            results.append({
                'name': ensemble.name,
                'tree_id': idx,
                'accuracy': accuracy,
                'num_nodes': tree.num_nodes(),
                'attributes_ratio': tree.attributes_ratio(),
                **params
            })

        return results

    def run(self, datasets: List[Dataset]):
        if self.EnsembleType is None:
            raise ValueError('Ensemble type is not specified')

        if self.variant == 'individual' and self.param_grid is None:
            raise ValueError('Param grid is not defined')

        print(f'Running {self.name} experiment...\n\n')

        def collect(results):
            for result_dict in results:
                self.add_result(**result_dict)

        for i in range(self.repetitions):
            pool = mp.Pool(mp.cpu_count())

            for dataset in datasets:
                grid = self.param_grid if self.variant == 'individual' else self.shared_param_grid

                train_val_sets = []

                if self.cv:
                    train_val_sets = [(train, val)
                                      for train, val in dataset.n_splits(5)]
                else:
                    train_val_sets = [dataset.split(0.2)]

                for params in grid:   # pylint: disable=not-an-iterable
                    for train, val in train_val_sets:
                        pool.apply_async(self.process, args=(
                            train, val, params), callback=collect)

            pool.close()
            pool.join()
