from .AdaboostExperiment import AdaboostExperiment
from .RandomForestExperiment import RandomForestExperiment
from .BaggingExperiment import BaggingExperiment
from .LGBExperiment import LGBExperiment
from .CatboostExperiment import CatboostExperiment
from .XGBoostExperiment import XGBoostExperiment


def load_all_experiments(variant):
    return [
        AdaboostExperiment(variant),
        RandomForestExperiment(variant),
        BaggingExperiment(variant),
        LGBExperiment(variant),
        CatboostExperiment(variant),
        XGBoostExperiment(variant),
    ]
