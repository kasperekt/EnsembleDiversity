import os

from experiments import LGBExperiment
from structure import Tree, Dataset, AdaboostEnsemble, RandomForestEnsemble, LGBEnsemble
from config import OUT_DIR
from typing import List
from sklearn.datasets import load_iris, load_breast_cancer


def draw_trees(structures: List[Tree]):
    for i, tree_structure in enumerate(structures):
        tree_path = os.path.join(
            'plots', f'tree_{tree_structure.clf_type.lower()}_{i + 1}.png')
        tree_structure.draw(tree_path)


def draw():
    iris_data = Dataset.from_sklearn("iris", load_iris())

    general_params = {
        'n_estimators': 10,
        'max_depth': 3
    }

    lgb_ensemble = LGBEnsemble(general_params)
    lgb_ensemble.fit(iris_data)

    adaboost_ensemble = AdaboostEnsemble(general_params)
    adaboost_ensemble.fit(iris_data)

    rf_ensemble = RandomForestEnsemble(general_params)
    rf_ensemble.fit(iris_data)

    draw_trees(lgb_ensemble.trees)
    draw_trees(adaboost_ensemble.trees)
    draw_trees(rf_ensemble.trees)


def run_experiment():
    iris_train, iris_val = Dataset.from_sklearn("iris", load_iris()).split()
    cancer_train, cancer_val = Dataset.from_sklearn("cancer",
                                                    load_breast_cancer()).split()

    train_datasets = [iris_train, cancer_train]
    val_datasets = [iris_val, cancer_val]

    exp = LGBExperiment(train_datasets)
    exp.run(train_datasets, val_datasets)
    exp.to_csv(os.path.join(OUT_DIR, 'lgb-ensemble.csv'))


if __name__ == '__main__':
    run_experiment()
