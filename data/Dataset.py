import numpy as np
import sklearn

from typing import Tuple
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_breast_cancer, fetch_openml


class Dataset(object):
    def __init__(self, X, y, feature_names, target_names, name=None, categories={}, ct=None):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names
        self.name = name
        self.categories = categories
        self.ct = ct

        self._is_oh_encoded = False

    def copy(self):
        return Dataset(self.X.copy(), self.y.copy(),
                       self.feature_names[:], self.target_names[:],
                       name=self.name, categories=self.categories)

    @staticmethod
    def from_sklearn(name, dataset):
        return Dataset(dataset.data, dataset.target,
                       dataset.feature_names, dataset.target_names, name=name)

    @staticmethod
    def from_openml(name):
        dataset = fetch_openml(name)

        feature_names = dataset.feature_names if 'feature_names' in dataset else None

        if 'target_names' in dataset:
            target_names = dataset.target_names
            target = dataset.target
        else:
            le = LabelEncoder()
            le.fit(dataset.target)

            target = le.transform(dataset.target)
            target_names = le.classes_

        return Dataset(dataset.data, target,
                       feature_names, target_names,
                       name=name, categories=dataset.categories)

    @staticmethod
    def create_iris():
        return Dataset.from_sklearn('iris', load_iris())

    @staticmethod
    def create_cancer():
        return Dataset.from_sklearn('cancer', load_breast_cancer())

    def oh_encoded(self):
        if self._is_oh_encoded or len(self.categories) < 1:
            return self

        categories = list(self.categories.keys())

        all_idx = set(range(0, self.X.shape[1]))
        cat_features_idx = [self.feature_names.index(
            cat) for cat in categories]
        num_features_idx = list(all_idx.difference(cat_features_idx))

        ct = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(categories='auto', sparse=False), cat_features_idx),
            ('num', 'passthrough', num_features_idx)
        ])

        ct.fit(self.X)

        # New feature names
        categories = ct.transformers_[0][1].categories_
        cat_dict = {idx: cat for idx, cat in zip(
            cat_features_idx, categories)}

        new_feature_names = []
        for idx in all_idx:
            name = self.feature_names[idx]

            if idx in cat_features_idx:
                names = [f'{name}_{int(cat_idx)}' for cat_idx in cat_dict[idx]]
                new_feature_names += names
            else:
                new_feature_names.append(name)

        new_dataset = Dataset(ct.transform(self.X), self.y,
                              new_feature_names, self.target_names,
                              name=self.name, categories=self.categories,
                              ct=ct)
        new_dataset._is_oh_encoded = True

        return new_dataset

    def size(self):
        return len(self.X)

    def num_classes(self):
        return len(self.target_names)

    def split(self, test_size=0.1):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=test_size)

        train_dataset = Dataset(
            X_train, y_train, self.feature_names, self.target_names, name=self.name)
        val_dataset = Dataset(
            X_val, y_val, self.feature_names, self.target_names, name=self.name)

        return train_dataset, val_dataset
