from inspect import signature
from typing import Callable, Optional

import pandas as pd

from wrappers import dataframe_input


class BinaryClassifierEvaluator:
    class MetricType:
        SCORE = "SCORE"
        PREDICTION = "PREDICTION"
        CONFUSION_MATRIX = "CONFUSION_MATRIX"

    def __init__(self, estimator):
        self.estimator = estimator

    def predict_scores(self, X):
        return self.estimator.predict_proba(X)[:, 1]

    def predict(self, X, threshold: float):
        return self.predict_scores(X) > threshold

    def confusion_matrix_parameters(self, X, y, threshold: Optional[float] = None):
        if threshold:
            return self._confusion_matrix_from_predictions(y, self.predict(X, threshold))
        else:
            return self._confusion_matrices_for_all_thresholds(y, self.predict_scores(X))

    @staticmethod
    def _confusion_matrix_from_predictions(target, predicted_target):
        return pd.Series(dict(
            TP=(predicted_target & target).sum(),
            FP=(predicted_target & ~target).sum(),
            TN=(~predicted_target & ~target).sum(),
            FN=(~predicted_target & target).sum()
        ))

    @staticmethod
    def _confusion_matrices_for_all_thresholds(target, target_scores):
        num_samples = len(target)
        num_positive = target.sum()
        num_negative = num_samples - num_positive

        df = pd.DataFrame({"target": target, "score": target_scores})
        df = df.sort_values("score", ascending=False)

        df["TP"] = df.target.cumsum()
        df["FN"] = num_positive - df["TP"]
        df["FP"] = (~df.target).cumsum()
        df["TN"] = num_negative - df["FP"]

        return df.sort_values("score", ascending=True)

    def calculate_score_metrics(self, X: pd.DataFrame, y: pd.Series, **metrics: Callable[[pd.Series, pd.Series], float]):
        y_scores = self.predict_scores(X)
        return {
            metric_name: metric_function(y_scores, y)
            for metric_name, metric_function in metrics.items()
        }

    def calculate_prediction_metrics(self, X, y, threshold: float, **metrics: Callable[[pd.Series, pd.Series], float]):
        y_predicted = self.predict(X, threshold)
        return {
            metric_name: metric_function(y_predicted, y)
            for metric_name, metric_function in metrics.items()
        }

    def calculate_confusion_matrix_metrics(self, X, y, threshold: Optional[float] = None, **metrics: Callable):
        if threshold:
            return {
                metric_name: dataframe_input(metric_function)(self.confusion_matrix_parameters(X, y, threshold))
                for metric_name, metric_function in metrics.items()
            }
        else:
            confusion_matrix_parameters = self.confusion_matrix_parameters(X, y)
            metric_results = pd.DataFrame({
                metric_name: dataframe_input(metric_function)(confusion_matrix_parameters)
                for metric_name, metric_function in metrics.items()
            })

            return pd.concat([confusion_matrix_parameters, metric_results], axis="columns")
