from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import TfidfVectorizer


class VectorizationStrategy(ABC):
    @abstractmethod
    def execute(self, X_train, X_test):
        pass


class TfidfPreprocessingStrategy(VectorizationStrategy):
    def __init__(self, ngram_range=(1, 1), max_features=5000):
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)

    def execute(self, X_train, X_test):
        X_train =  self.vectorizer.fit_transform(X_train).toarray()
        X_test = self.vectorizer.transform(X_test).toarray()
        return X_train, X_test