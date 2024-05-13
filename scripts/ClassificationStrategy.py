from abc import ABC, abstractmethod

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb


class ClassificationStrategy(ABC):
    """
    Загальний інтерфейс стратегій.
    """
    @abstractmethod
    def train_and_predict(self, X_train, y_train, X_test):
        pass

    @abstractmethod
    def get_model(self):
        pass


class LogisticRegressionStrategy(ClassificationStrategy):
    """
    Конкретна стратегії, що реалізує інтерфейс ClassificationStrategy.
    Классифікация за допомогою LogisticRegression.
    """
    def __init__(self):
        self.model = LogisticRegression()

    def train_and_predict(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def get_model(self):
        return self.model


class SVCStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = LinearSVC()

    def train_and_predict(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def get_model(self):
        return self.model


class GaussianNBStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = GaussianNB()

    def train_and_predict(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def get_model(self):
        return self.model


class RandomForestStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = RandomForestClassifier()

    def train_and_predict(self, X_train, y_train, X_test):
        self.model.fit(X_train, y_train)
        return self.model.predict(X_test)

    def get_model(self):
        return self.model


class LightGBMStrategy(ClassificationStrategy):
    def __init__(self):
        self.model = None

    def set_model(self, model):
        self.model = model

    def train_and_predict(self, X_train, y_train, X_test, params=None):
        if not params:
            params = {}
        train_data = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(params, train_data)
        self.set_model(model)
        return self.model.predict(X_test)

    def get_model(self):
        return self.model