import numpy as np
from typing import List
from math import ceil

'''
This module works only for binary problems
'''


def confusion_matrix(true_pred: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> np.ndarray:
    cm = np.zeros((2, 2), dtype=np.int)

    for t, a, b in zip(true_pred, pred_a, pred_b):
        if t == a and t == b:
            # True Positive
            cm[0, 0] += 1
        elif t == a and t != b:
            # False Positive
            cm[0, 1] += 1
        elif t != a and t == b:
            # False Negative
            cm[1, 0] += 1
        else:
            # True Negative
            cm[1, 1] += 1

    return cm


def bin_q(true_pred: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    cm = confusion_matrix(true_pred, pred_a, pred_b)

    denominator = (cm[0, 0] * cm[1, 1] + cm[1, 0] * cm[0, 1])

    if denominator == 0:
        return 0

    return (cm[0, 0] * cm[1, 1] - cm[1, 0] * cm[0, 1]) / denominator


def bin_corr(true_pred: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    cm = confusion_matrix(true_pred, pred_a, pred_b)

    denominator = np.sqrt(
        (cm[0, 0] + cm[0, 1]) *
        (cm[1, 0] + cm[1, 1]) *
        (cm[0, 0] + cm[1, 0]) *
        (cm[0, 1] + cm[1, 1])
    )

    if denominator == 0:
        return 0

    return (cm[0, 0] * cm[1, 1] - cm[1, 0] * cm[0, 1]) / denominator


def bin_df(true_pred: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    cm = confusion_matrix(true_pred, pred_a, pred_b)
    return cm[1, 1] / np.sum(cm)


def bin_entropy(true_pred: np.ndarray, preds: List[np.ndarray]) -> float:
    num_clf = len(preds)
    num_examples = len(true_pred)

    res = np.sum([(true_pred == pred).astype(np.int)
                  for pred in preds], axis=0)
    res_min = np.minimum(res, num_clf - res)

    if num_examples == 1 or (num_clf - ceil(num_clf / 2)) == 0:
        return 0

    return 1 / num_examples * (1 / (num_clf - ceil(num_clf / 2))) * np.sum(res_min)


def bin_kw(true_pred: np.ndarray, preds: List[np.ndarray]) -> float:
    num_clf = len(preds)
    num_examples = len(true_pred)

    res = np.sum([(true_pred == pred).astype(np.int)
                  for pred in preds], axis=0)

    return (1 / (num_clf * num_examples ** 2)) * np.sum(res * (num_clf - res))
