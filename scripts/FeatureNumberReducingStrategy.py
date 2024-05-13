from abc import ABC, abstractmethod

from sklearn.decomposition import TruncatedSVD


class FeatureNumberReducingStrategy(ABC):
    @abstractmethod
    def execute(self, X_train, X_test):
        pass


class TruncatedSVDStrategy(FeatureNumberReducingStrategy):
    def __init__(self, n_components=3000):
        self.vectorizer = TruncatedSVD(n_components=n_components)

    def execute(self, X_train, X_test):
        X_train = self.vectorizer.fit_transform(X_train)
        X_test = self.vectorizer.transform(X_test)
        return X_train, X_test