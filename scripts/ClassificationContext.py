import pickle

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from scripts.VectorizationStrategy import VectorizationStrategy
from scripts.ClassificationStrategy import ClassificationStrategy
from scripts.FeatureNumberReducingStrategy import FeatureNumberReducingStrategy
from scripts.preprocessing import base_text_preprocessing


class ClassificationContext:
    """
    Контекст, що працює зі стратегіями.
    """
    def __init__(self,
                 preprocessing: VectorizationStrategy,
                 strategy: ClassificationStrategy,
                 feature_reducing: FeatureNumberReducingStrategy = None):
        self._preprocessing = preprocessing
        self._strategy = strategy
        self._feature_reducing = feature_reducing

    def set_strategy(self, strategy: ClassificationStrategy):
        self._strategy = strategy

    def set_preprocessing(self, preprocessing: VectorizationStrategy):
        self._preprocessing = preprocessing

    def set_feature_reducing(self, feature_reducing):
        self._feature_reducing = feature_reducing

    def execute_strategy(self, X_train, y_train, X_test):

        X_train = base_text_preprocessing(X_train.iloc[:, 0])
        X_test = base_text_preprocessing(X_test.iloc[:, 0])

        X_train, X_test = self._preprocessing.execute(X_train, X_test)
        if self._feature_reducing:
            X_train, X_test = self._feature_reducing.execute(X_train, X_test)
        return self._strategy.train_and_predict(X_train, y_train, X_test)

    def cross_validate(self, X, y, cv=5):
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_val = X[train_index], X[test_index]
            y_train, y_val = y[train_index], y[test_index]

            y_pred = self.execute_strategy(X_train, y_train, X_val)
            scores.append(roc_auc_score(y_val, y_pred))
        mean_score = sum(scores) / len(scores)
        return mean_score

    def serialize_model(self, filename=None):
        if not filename:
            filename = ' '.join(
                [str(self._preprocessing), str(self._strategy), str(self._feature_reducing)]) + '.pkl'
        with open('../models/' + filename, "wb") as f:
            pickle.dump(self._strategy.get_model(), f)
