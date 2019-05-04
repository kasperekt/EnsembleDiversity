import numpy as np
import scipy.stats


def majority_voting(preds: np.ndarray) -> np.ndarray:
    if len(preds.shape) != 2:
        raise ValueError('Preds should be 2-dimensional')

    best_preds = []

    for pred in preds:
        best, _ = scipy.stats.mode(pred)
        best_preds.append(best[0])

    return np.array(best_preds)
