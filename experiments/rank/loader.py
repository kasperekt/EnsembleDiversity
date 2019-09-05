from .AdaboostExperiment import AdaboostExperiment
from .RandomForestExperiment import RandomForestExperiment
from .BaggingExperiment import BaggingExperiment
from .LGBExperiment import LGBExperiment
from .CatboostExperiment import CatboostExperiment
from .XGBoostExperiment import XGBoostExperiment


def load_all_experiments(variant, cv):
    return [
        AdaboostExperiment(variant, cv),
        RandomForestExperiment(variant, cv),
        BaggingExperiment(variant, cv),
        LGBExperiment(variant, cv),
        CatboostExperiment(variant, cv),
        XGBoostExperiment(variant, cv),
    ]
