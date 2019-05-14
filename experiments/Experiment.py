import pandas as pd

from typing import List, Dict
from structure import Ensemble, Dataset
from collections import namedtuple

Result = namedtuple('Results', [
    'name', 'dataset_name', 'accuracy', 'node_diversity'])


class Experiment(object):
    def __init__(self, datasets: List[Dataset]):
        self.ensembles: List[Ensemble] = []
        self.results: List[Result] = []

    def reset(self):
        self.ensembles = []
        self.results = []

    def add_result(self, **kwargs):
        result = Result(**kwargs)
        self.results.append(result)

    def to_csv(self, filepath: str):
        if len(self.results) == 0:
            raise ValueError(
                'Results are empty. Try "run" before saving to csv.')

        df = pd.DataFrame(self.results)
        df.to_csv(filepath)
