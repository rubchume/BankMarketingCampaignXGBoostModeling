from inspect import signature
from typing import Any, Callable, Dict, List

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, fbeta_score, make_scorer, mutual_info_score, \
    ConfusionMatrixDisplay, precision_score, recall_score, roc_curve
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, cross_val_score, cross_validate, \
    LeaveOneOut
from dataclasses import dataclass, make_dataclass, fields, asdict

from collections import defaultdict

import pandas as pd


def calculate_scores(classifier, X, y):
    return Scores(**{
        name: calculate_metric(classifier, X, y)
        for name, calculate_metric in metrics.items()
    })


def print_scores(classifier, X, y, cross_validation=8):
    def get_training_scores(X, y):
        return calculate_scores(classifier, X, y)

    def get_cross_validation_mean_scores(X, y):
        cross_validation_results = cross_validate(
            classifier,
            X,
            y,
            scoring=metrics,
            cv=cross_validation
        )

        return Scores(**{
            metric: cross_validation_results[f'test_{metric}'].mean()
            for metric in metrics.keys()
        })

    training_scores = get_training_scores(X, y)
    print("Training")
    print(training_scores)

    cross_validation_mean_scores = get_cross_validation_mean_scores(X, y)
    print("Cross-validation")
    print(cross_validation_mean_scores)

    return training_scores, cross_validation_mean_scores


def register_scores(estimator_name, **kwargs):
    for attr, value in kwargs.items():
        setattr(estimators_scores[estimator_name], attr, value)


def dataframe_from_estimator_scores():
    df = pd.DataFrame(
        columns=pd.MultiIndex.from_product([[TRAINING, CROSS_VALIDATION, TEST], metrics.keys()])).rename_axis(
        index="Estimator")

    for estimator_name, estimator_scores in estimators_scores.items():
        if (scores := estimator_scores.training_scores) is not None:
            df.loc[estimator_name, (TRAINING, scores.keys())] = scores.values()

        if (scores := estimator_scores.cross_validation_mean_scores) is not None:
            df.loc[estimator_name, (CROSS_VALIDATION, scores.keys())] = scores.values()

        if (scores := estimator_scores.test_scores) is not None:
            df.loc[estimator_name, (TEST, scores.keys())] = scores.values()

    return df


def specificity_score(y_true: pd.Series, y_pred: pd.Series):
    true_negatives = sum((y_true == False) & (y_pred == False))
    false_positives = sum((y_true == False) & (y_pred == True))
    return true_negatives / (true_negatives + false_positives)


metrics = dict(
    Accuracy=make_scorer(accuracy_score),
    F1=make_scorer(f1_score),
    ROC_AUC=make_scorer(roc_auc_score, needs_threshold=True),
    Precision=make_scorer(precision_score),
    Recall_Sensibility=make_scorer(recall_score),
    Specificity=make_scorer(specificity_score),
    beta_score=make_scorer(fbeta_score, beta=2)
)

Scores = make_dataclass(
    "Scores",
    fields=[(metric, float) for metric in metrics.keys()],
    namespace={
        'keys': lambda self: list(asdict(self).keys()),
        "values": lambda self: list(asdict(self).values()),
    }
)


@dataclass
class EstimatorScores:
    estimator: Any = None
    training_scores: Scores = None
    cross_validation_mean_scores: Scores = None
    test_scores: Scores = None


TRAINING = "Training"
CROSS_VALIDATION = "Cross-validation"
TEST = "Test"

estimators_scores = defaultdict(EstimatorScores)


class ModelEvaluationLaboratory:
    def __init__(self, metrics: Dict[str, Callable[[np.array, np.array], float]]):
        self.scorers = {
            metric_name: self._make_scorer_from_metric(metric_function)
            for metric_name, metric_function in metrics.items()
        }

        self.estimators_scores = defaultdict(EstimatorScores)

    @staticmethod
    def _make_scorer_from_metric(metric: Callable[[np.array, np.array], float]):
        if "y_pred" in signature(metric):
            return make_scorer(metric)

        if "y_score" in signature(metric):
            return make_scorer(metric, needs_threshold=True)

    def calculate_scores(self, estimator, X, y):
        return Scores(**{
            name: calculate_metric(estimator, X, y)
            for name, calculate_metric in self.scorers.items()
        })
