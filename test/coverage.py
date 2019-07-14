import numpy as np

from data import Dataset
from structure import RandomForestEnsemble, TreeCoverage


def prepare() -> (RandomForestEnsemble, Dataset):
    data = Dataset.from_openml('splice')
    rf = RandomForestEnsemble({'n_estimators': 5, 'max_depth': 4})
    rf.fit(data)
    return rf, data


def main():
    clf, data = prepare()
    trees_coverage = clf.get_coverage()

    tree_coverage = trees_coverage[0]
    overall_size = 0
    for key, value in tree_coverage.leaves():
        # Tylko dla liści
        if 'Leaf' in key:
            overall_size += len(value)

    assert(len(data.X) == overall_size)
    print(tree_coverage)

    print('It works!')


if __name__ == '__main__':
    main()
