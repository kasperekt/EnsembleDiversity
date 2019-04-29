import lightgbm as lgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from parser.lightgbm import parse_lightgbm
from parser.sklearn import parse_sklearn
from structure.Dataset import Dataset


def get_lgb_trees(dataset: Dataset, **params):
    final_params = {
        'n_estimators': 100,
        'predict_leaf_index': True,
        **params
    }

    clf = lgb.LGBMClassifier(**final_params, objective='multiclass')
    clf.fit(dataset.X, dataset.y)

    return parse_lightgbm(clf, dataset)


def get_lgb_rf_trees(dataset: Dataset, **params):
    final_params = {
        'n_estimators': 100,
        'predict_leaf_index': True,
        'bagging_freq': 0,
        'bagging_fraction': 1.0,
        **params,
        'boosting_type': 'rf',
        'objective': 'multiclass'
    }

    clf = lgb.LGBMClassifier(**final_params)
    clf.fit(dataset.X, dataset.y)

    return parse_lightgbm(clf, dataset)


def get_adaboost_trees(dataset: Dataset, max_depth=5, **params):
    final_params = {
        'n_estimators': 100,
        **params
    }

    tree = DecisionTreeClassifier(max_depth=max_depth)
    clf = AdaBoostClassifier(**final_params, base_estimator=tree)
    clf.fit(dataset.X, dataset.y)

    return parse_sklearn(clf, dataset, "AdaBoost")


def get_rf_trees(dataset: Dataset, **params):
    final_params = {
        'max_depth': 5,
        'n_estimators': 100,
        **params
    }

    clf = RandomForestClassifier(**final_params)
    clf.fit(dataset.X, dataset.y)

    return parse_sklearn(clf, dataset, "RandomForest")
