import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from . import Dataset


class DatasetEncoder(object):
    def __init__(self, column_transformer=None, feature_names=None):
        self.column_transformer = column_transformer
        self.feature_names = feature_names

    def transform(self, dataset: Dataset):
        if self.column_transformer is None:
            return dataset

        return Dataset(self.column_transformer.transform(dataset.X), dataset.y,
                       self.feature_names, dataset.target_names,
                       name=dataset.name, categories=dataset.categories)

    @staticmethod
    def create_one_hot(dataset: Dataset):
        categories = list(dataset.categories.keys())

        if len(categories) == 0:
            return DatasetEncoder()

        all_idx = set(range(0, dataset.X.shape[1]))
        cat_features_idx = [dataset.feature_names.index(
            cat) for cat in categories]
        num_features_idx = list(all_idx.difference(cat_features_idx))

        ct = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(categories='auto', sparse=False), cat_features_idx),
            ('num', 'passthrough', num_features_idx)
        ])

        ct.fit(dataset.X)

        # New feature names
        categories = ct.transformers_[0][1].categories_
        cat_dict = {idx: cat for idx, cat in zip(
            cat_features_idx, categories)}

        new_feature_names = []
        for idx in all_idx:
            name = dataset.feature_names[idx]

            if idx in cat_features_idx:
                names = [f'{name}_{int(cat_idx)}' for cat_idx in cat_dict[idx]]
                new_feature_names += names
            else:
                new_feature_names.append(name)

        return DatasetEncoder(ct, new_feature_names)
