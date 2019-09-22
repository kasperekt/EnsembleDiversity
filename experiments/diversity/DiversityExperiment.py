from abc import ABCMeta, abstractmethod
from typing import List, Dict, Tuple
from data import Dataset
from collections import namedtuple
from structure import Ensemble
from experiments import ExperimentVariant, Experiment

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class DiversityExperiment(Experiment):
    def build_shared_param_grid(self):
        return ParameterGrid({
            'n_estimators': [10, 50, 100],
            'max_depth': [5]
        })

    def collect(self, payload):
        self.add_result(**payload)

    def process(self, repetition: int, train: Dataset, val: Dataset, params: dict):
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
            # Attributes
            'avg_node_count': ensemble.avg_node_count(),
            'avg_attributes_used': ensemble.avg_attributes_used(),
            # Structural Measures
            'node_diversity': ensemble.node_diversity(),
            'coverage_std': ensemble.coverage_leaves_std(),
            'coverage_minmax': ensemble.coverage_leaves_minmax(),
            'coverage_avg': ensemble.coverage_leaves_avg(),
            'used_attributes_ratio': ensemble.used_attributes_ratio(),
            # Behavioral Measures
            'entropy': ensemble.entropy(val),
            'q': ensemble.q(val),
            'df': ensemble.df(val),
            'kw': ensemble.kw(val),
            'corr': ensemble.corr(val)
        }
