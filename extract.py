import lightgbm as lgb

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

from structure import Dataset, SklearnEnsemble, LGBEnsemble


def get_lgb_trees(dataset: Dataset, **params):
    final_params = {
        'n_estimators': 100,
        'predict_leaf_index': True,
        **params
    }

    clf = lgb.LGBMClassifier(**final_params, objective='multiclass')
    ensemble = LGBEnsemble(clf, "LGBTrees")
    ensemble.fit(dataset)

    return ensemble


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
    ensemble = LGBEnsemble(clf, "LGBTrees")
    ensemble.fit(dataset)

    return ensemble


def get_adaboost_trees(dataset: Dataset, max_depth=5, **params):
    final_params = {
        'n_estimators': 100,
        **params
    }

    tree = DecisionTreeClassifier(max_depth=max_depth)
    clf = AdaBoostClassifier(**final_params, base_estimator=tree)
    ensemble = SklearnEnsemble(clf, "AdaBoost")
    ensemble.fit(dataset)

    return ensemble


def get_rf_trees(dataset: Dataset, **params):
    final_params = {
        'max_depth': 5,
        'n_estimators': 100,
        **params
    }

    clf = RandomForestClassifier(**final_params)
    ensemble = SklearnEnsemble(clf, "RF")
    ensemble.fit(dataset)

    return ensemble
