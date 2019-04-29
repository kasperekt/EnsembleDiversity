import numpy as np
import scipy.stats

from structure.Tree import Tree
from typing import List


def predict_majority(trees: List[Tree], data) -> np.ndarray:
    preds = np.array([tree.predict(data) for tree in trees])
    preds = np.rollaxis(preds, axis=1)

    best_preds = []

    for pred in preds:
        best, _ = scipy.stats.mode(pred)
        best_preds.append(best[0])

    return np.array(best_preds)


def majority_voting(preds: np.ndarray) -> np.ndarray:
    if len(preds.shape) != 2:
        raise ValueError('Preds should be 2-dimensional')

    best_preds = []

    for pred in preds:
        best, _ = scipy.stats.mode(pred)
        best_preds.append(best[0])

    return np.array(best_preds)
