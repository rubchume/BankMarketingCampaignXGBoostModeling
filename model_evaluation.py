from inspect import signature
from typing import Any, Callable, Dict

import numpy as np
from sklearn.inspection import partial_dependence
from sklearn.metrics import make_scorer, ConfusionMatrixDisplay
from sklearn.model_selection import cross_validate
from dataclasses import dataclass

from collections import defaultdict

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

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
    def __init__(self, *estimators, metrics: Dict[str, Callable[[np.array, np.array], float]]):
        self.scorers = {
            metric_name: self._make_scorer_from_metric(metric_function)
            for metric_name, metric_function in metrics.items()
        }

        self.estimators_scores = defaultdict(EstimatorScores)


