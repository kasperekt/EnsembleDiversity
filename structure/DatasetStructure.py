class DatasetStructure(object):
    def __init__(self, X, y, feature_names, target_names):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.target_names = target_names

    def size(self):
        return len(self.X)
