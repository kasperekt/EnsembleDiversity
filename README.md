# Ensemble diversity

## Running experiments

```
usage: main.py [-h] [--variant {individual,shared}]
               [--experiment {rank,diversity}] [--cv CV] [--reps REPS]

optional arguments:
  -h, --help            show this help message and exit
  --variant {individual,shared}
  --experiment {rank,diversity}
  --cv CV
  --reps REPS
```

What is variant? It is experiment strategy representing which hyperparameters are going to be used. **individual** variant allows each ensemble to apply its own hyperparameters, whereas **shared** strategy applies the same hyperparameters for every classifer. To change parameters for **individual** you need to change `param_grid` attribute in experiment variant, i.e. set `self.param_grid` inside **AdaBoostExperiment** initializer. To modify **shared** variant parameters, change `build_shared_param_grid` function in experiment class.

Example command:

```
python main.py --variant shared --experiment diversity --cv 5 --reps 1
```

## Visualizations

Example command:

```
python visualize.py
```

Creating scatterplots is done using `visualize.py` script.

## Validation

```
usage: validate_structure.py [-h] [--verbose] [-c CHECK] [-A]

optional arguments:
  -h, --help            show this help message and exit
  --verbose
  -c CHECK, --check CHECK
  -A, --all
```

Example command:

```
python validate_structure.py -c lgb -c bag
```

In order to ensure that all ensemble structures are correctly implemented, you can run `validate_structure.py` script. It accepts parameters like -c, --check to select certain ensembles by the key (i.e. `ada`, `bag`, `rf`, `lgb`, `cb` and `xgb`).
To check all ensembles, just pass `-A` flag.
