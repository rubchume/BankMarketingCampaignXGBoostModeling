from inspect import signature
from typing import Any, Callable, Dict

import numpy as np
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from dataclasses import dataclass

from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go

from eda_tools.statistical_tools import estimate_probability_density_function


@dataclass
class EstimatorScores:
    estimator: Any = None
    training_scores: Dict[str, float] = None
    cross_validation_mean_scores: Dict[str, float] = None
    test_scores: Dict[str, float] = None


TRAINING = "Training"
CROSS_VALIDATION = "Cross-validation"
TEST = "Test"


class ModelEvaluationLaboratory:
    def __init__(self, metrics: Dict[str, Callable[[np.array, np.array], float]]):
        self.scorers = {
            metric_name: self._make_scorer_from_metric(metric_function)
            for metric_name, metric_function in metrics.items()
        }

        self.estimators_scores = defaultdict(EstimatorScores)

    @staticmethod
    def _make_scorer_from_metric(metric: Callable[[np.array, np.array], float]):
        if "y_pred" in signature(metric).parameters:
            return make_scorer(metric)

        if "y_score" in signature(metric).parameters:
            return make_scorer(metric, needs_threshold=True)

    def get_estimator(self, estimator_name):
        return self.estimators_scores[estimator_name].estimator

    def register_estimator(self, estimator_name, estimator):
        self._register_scores(estimator_name, estimator=estimator)

    def evaluate_estimator(self, estimator_name, X_train=None, y_train=None, X_test=None, y_test=None, cv=8):
        estimator = self.get_estimator(estimator_name)

        if X_train is not None and y_train is not None:
            self._register_scores(
                estimator_name,
                training_scores=self._calculate_scores(estimator, X_train, y_train),
                cross_validation_mean_scores=self._calculate_cross_validation_mean_scores(estimator, X_train, y_train,
                                                                                          cv)
            )

        if X_test is not None and y_test is not None:
            self._register_scores(
                estimator_name,
                test_scores=self._calculate_scores(estimator, X_test, y_test)
            )

    def plot_estimator_binary_detection_measures(self, estimator_name, X, y, metrics=None):
        def plot_roc():
            num_positives = detection_measures.target.sum()
            recall = detection_measures.TP / num_positives
            false_positive_rate = detection_measures.FP / (num_samples - num_positives)

            return go.Figure(
                data=go.Scatter(x=false_positive_rate, y=recall),
                layout=go.Layout(
                    title="ROC",
                    xaxis=dict(title="False Positive Rate (1 - Specificity)", range=[0, 1]),
                    yaxis=dict(title="Recall", range=[0, 1], scaleanchor="x", scaleratio=1),
                    height=600,
                    width=600
                )
            )

        def plot_metrics():
            quantile = np.arange(num_samples) / num_samples

            negative_pdf = estimate_probability_density_function(
                quantile[detection_measures.target == False],
                bandwidth=0.025
            )
            positive_pdf = estimate_probability_density_function(
                quantile[detection_measures.target == True],
                bandwidth=0.025
            )

            data = [
                go.Scatter(x=quantile, y=negative_pdf(quantile), name="Negative"),
                go.Scatter(x=quantile, y=positive_pdf(quantile), name="Positive"),
            ]

            if metrics is not None:
                data += [
                    go.Scatter(x=quantile, y=detection_measures[metric], name=metric, yaxis="y2")
                    for metric in metrics.keys()
                ]

            return go.Figure(
                data=data,
                layout=go.Layout(
                    xaxis=dict(title="Sample quantile (ordered by predicted score)", range=[0, 1]),
                    yaxis=dict(title="Class Probability density"),
                    yaxis2=dict(title="Metric value", overlaying="y", side="right"),
                )
            )

        def plot_confusion_matrix():
            threshold_index = detection_measures.score[detection_measures.score > 0.5].idxmin()
            detection_measures_at_threshold = detection_measures.loc[threshold_index]
            confusion_matrix = np.array([
                [detection_measures_at_threshold.TN, detection_measures_at_threshold.FP],
                [detection_measures_at_threshold.FN, detection_measures_at_threshold.TP]
            ])
            return ConfusionMatrixDisplay(confusion_matrix)

        detection_measures = self.get_estimator_binary_detection_measures(estimator_name, X, y, metrics)

        num_samples = len(detection_measures)

        return plot_metrics(), plot_roc(), plot_confusion_matrix()

    def get_estimator_binary_detection_measures(self, estimator_name, X, y, metrics=None):
        estimator = self.get_estimator(estimator_name)
        y_score = estimator.predict_proba(X)[:, 1]
        binary_detection_measures = self._get_binary_detection_measures(y, y_score)

        if metrics is None:
            return binary_detection_measures

        derivated_metrics = pd.DataFrame({
            name: metric(**{
                parameter: binary_detection_measures[parameter]
                for parameter in signature(metric).parameters.keys()
            }) for name, metric in metrics.items()
        })

        return pd.concat([binary_detection_measures, derivated_metrics], axis="columns")

    @staticmethod
    def _get_binary_detection_measures(binary_target: pd.Series, score: pd.Series):
        num_samples = len(binary_target)
        num_positive = binary_target.sum()
        num_negative = num_samples - num_positive

        df = pd.DataFrame({"target": binary_target, "score": score})
        df = df.sort_values("score", ascending=False)

        true_positive = df.target.cumsum()
        false_positive = (~df.target).cumsum()
        true_negative = num_negative - false_positive
        false_negative = num_positive - true_positive

        df["TP"] = true_positive
        df["FP"] = false_positive
        df["TN"] = true_negative
        df["FN"] = false_negative

        return df.sort_values("score", ascending=True)

    def get_estimator_scores_dataframe(self):
        df = pd.DataFrame(
            columns=pd.MultiIndex.from_product([[TRAINING, CROSS_VALIDATION, TEST], self.scorers.keys()])
        ).rename_axis(index="Estimator")

        for estimator_name, estimator_scores in self.estimators_scores.items():
            if (scores := estimator_scores.training_scores) is not None:
                df.loc[estimator_name, (TRAINING, list(scores.keys()))] = scores.values()

            if (scores := estimator_scores.cross_validation_mean_scores) is not None:
                df.loc[estimator_name, (CROSS_VALIDATION, list(scores.keys()))] = scores.values()

            if (scores := estimator_scores.test_scores) is not None:
                df.loc[estimator_name, (TEST, list(scores.keys()))] = scores.values()

        return df

    def _register_scores(self, estimator_name, **kwargs):
        for attr, value in kwargs.items():
            setattr(self.estimators_scores[estimator_name], attr, value)

    def _calculate_scores(self, estimator, X, y):
        return {
            name: calculate_metric(estimator, X, y)
            for name, calculate_metric in self.scorers.items()
        }

    def _calculate_cross_validation_mean_scores(self, estimator, X, y, cv=8):
        cross_validation_results = cross_validate(
            estimator,
            X,
            y,
            scoring=self.scorers,
            cv=cv
        )

        return {
            metric: cross_validation_results[f'test_{metric}'].mean()
            for metric in self.scorers.keys()
        }
