from .Dataset import Dataset


MULTI_CLASS_SETS = [('iris', 'active'), ('aids', 'active'),
                    ('spectrometer', 'active')]

BINARY_CLASS_SETS = [('SPECT', 'active'), ('oil_spill', 'active'),
                     ('splice', 'active'), ('segment', '2'),
                     ('ionosphere', 'active'), ('sonar', 'active'),
                     ('pyrim', '2'), ('pollution', '2'),
                     ('boston', '2'), ('glass', '2'),
                     ('lymph', '2'), ('vehicle', '2'),
                     ('flags', '2'), ('wine', '2'),
                     ('cmc', '2'), ('isolet', '2'),
                     ('sensory', '2')]

SETS = MULTI_CLASS_SETS + BINARY_CLASS_SETS


def load_dataset(set_id, set_version='active'):
    if (set_id, set_version) not in SETS:
        raise ValueError(f'{set_id} is not valid dataset')

    return Dataset.from_openml(set_id, version=set_version)


def load_all_datasets(test_size=0.0, sets=BINARY_CLASS_SETS):
    datasets = [load_dataset(set_id, set_version)
                for set_id, set_version in sets]

    if test_size > 0:
        train_sets = []
        val_sets = []

        for dataset in datasets:
            train_set, val_set = dataset.split(test_size)
            train_sets.append(train_set)
            val_sets.append(val_set)

        return train_sets, val_sets

    return datasets, None
