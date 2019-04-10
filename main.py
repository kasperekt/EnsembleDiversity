import os
import lightgbm as lgb
from sklearn.datasets import load_iris
from parser.lightgbm import parse_lightgbm_json


def main():
    X, y = load_iris(True)

    clf = lgb.LGBMClassifier(n_estimators=5, objective='multiclass')
    clf.fit(X, y)

    lgb_json = clf.booster_.dump_model()
    tree_structures = parse_lightgbm_json(lgb_json)
    for i, tree_structure in enumerate(tree_structures):
        tree_path = os.path.join('plots', f'tree_{i + 1}.png')
        tree_structure.draw(tree_path)


if __name__ == '__main__':
    main()
