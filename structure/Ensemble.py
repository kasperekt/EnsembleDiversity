class Ensemble(object):
    def __init__(self, clf, name="Ensemble"):
        self.trees = []
        self.clf = clf
        self.name = name

    def __iter__(self):
        for tree in self.trees:
            yield tree
