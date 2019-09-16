from typing import List, Dict, Tuple
from data import Dataset
from structure import Ensemble
from experiments import ExperimentVariant, Experiment

from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score


class RankExperiment(Experiment):
    def build_shared_param_grid(self):
        return ParameterGrid({
            'n_estimators': [10, 40, 80],
            'max_depth': [6]
        })

    def process(self, repetition: int, train: Dataset, val: Dataset, params: dict):
        '''
        Run individual experiment
        '''
        print(f'[{train.name}] Params: {params}')

        ensemble: Ensemble = self.EnsembleType(  # pylint: disable=not-callable
            params)
        ensemble.fit(train)

        ensemble_preds = ensemble.predict(val)
        ensemble_acc = accuracy_score(val.y, ensemble_preds)

        results = []

        for idx, tree in enumerate(ensemble.trees):
            preds = tree.predict(val.X, labeled_result=True)
            accuracy = accuracy_score(val.y, preds)

            results.append({
                'repetition': repetition,
                'name': ensemble.name,
                'dataset': train.name,
                'tree_id': idx,
                'ensemble_accuracy': ensemble_acc,
                'accuracy': accuracy,
                'num_nodes': tree.num_nodes(),
                'attributes_count': tree.attributes_count(),
                'attributes_ratio': tree.attributes_ratio(),
                **params
            })

        return results

    def collect(self, payload):
        for result_dict in payload:
            self.add_result(**result_dict)
