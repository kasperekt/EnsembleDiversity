from sklearn.model_selection import train_test_split


class Dataset(object):
    def __init__(self, X, y, feature_names, target_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names

    def size(self):
        return len(self.X)

    def split(self, test_size=0.1):
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=test_size)

        train_dataset = Dataset(X_train, y_train, self.feature_names, self.target_names)
        val_dataset = Dataset(X_val, y_val, self.feature_names, self.target_names)

        return train_dataset, val_dataset
