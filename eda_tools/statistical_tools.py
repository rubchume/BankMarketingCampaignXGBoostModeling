import pandas as pd
from statsmodels.stats.proportion import proportion_confint

from .probability_distribution import ProbabilityDistribution


def get_confidence_intervals_for_binomial_variable(distribution: ProbabilityDistribution, binary_variable, significance_level=0.05):
    successes = distribution.given(**{binary_variable: True}).distribution

    all_variables_except_binary = [variable for variable in distribution.variables if variable != binary_variable]
    totals = distribution.get_marginal_count(all_variables_except_binary)

    confidence_intervals = pd.DataFrame({"successes": successes, "totals": totals}).apply(
        lambda s: pd.Series(proportion_confint(s["successes"], s["totals"], significance_level)),
        axis="columns"
    ).set_axis(["Low", "High"], axis="columns")

    return confidence_intervals