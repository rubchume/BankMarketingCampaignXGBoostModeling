from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity
from statsmodels.stats.proportion import proportion_confint

from .probability_distribution import ProbabilityDistribution


class Normalization(Enum):
    COUNT = "COUNT"
    PROBABILITY = "PROBABILITY"


def get_distribution(variables: List[pd.Series], normalization=Normalization.PROBABILITY, conditioned_on=None):
    num_samples = len(variables[0])
    all_variables = (conditioned_on + variables) if conditioned_on is not None else variables

    joint_distribution_count = pd.Series([0] * num_samples).groupby(by=all_variables).count()

    if conditioned_on is None:
        if normalization == Normalization.COUNT:
            return joint_distribution_count
        else:
            return joint_distribution_count / joint_distribution_count.sum()

    return joint_distribution_count.groupby(level=list(range(len(conditioned_on)))).transform(lambda s: s / s.sum())


def get_confidence_intervals_for_binomial_variable(
        distribution: ProbabilityDistribution, binary_variable, significance_level=0.05
):
    successes = distribution.given(**{binary_variable: True}).get_distribution()

    all_variables_except_binary = [variable for variable in distribution.variables if variable != binary_variable]
    totals = distribution.get_marginal_count(all_variables_except_binary)

    confidence_intervals = pd.DataFrame({"successes": successes, "totals": totals}).apply(
        lambda s: pd.Series(proportion_confint(s["successes"], s["totals"], significance_level)),
        axis="columns"
    ).set_axis(["Low", "High"], axis="columns")

    return confidence_intervals


def clip_outliers(x: pd.Series, max_normalized_deviation=2):
    x_clipped = x.copy()
    mu = x.mean()
    std = x.std()
    x_normalized = (x - mu) / std

    high_outliers = x_normalized > max_normalized_deviation
    x_clipped[high_outliers] = std * max_normalized_deviation + mu

    low_outliers = x_normalized < (-max_normalized_deviation)
    x_clipped[low_outliers] = -std * max_normalized_deviation + mu

    return x_clipped


class DiscretizationMagnitude(Enum):
    VALUE = "VALUE"
    QUANTILE = "QUANTILE"


def discretize_variable(variable: pd.Series, number_of_bins=10, keep_equal_size_in: DiscretizationMagnitude = DiscretizationMagnitude.VALUE):
    if keep_equal_size_in == DiscretizationMagnitude.VALUE:
        return pd.cut(variable, bins=number_of_bins, duplicates="drop")
    else:
        return pd.qcut(variable, q=number_of_bins, duplicates="drop")


def estimate_probability_density_function(variable: pd.Series, bandwidth=1):
    def get_probability_density(values: pd.Series):
        return np.exp(kde.score_samples(np.array(values).reshape(-1, 1)))

    kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(np.array(variable).reshape(-1, 1))
    return get_probability_density
