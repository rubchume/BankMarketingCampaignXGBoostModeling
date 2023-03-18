from unittest import TestCase

import pandas as pd
from pandas._testing import assert_series_equal

from eda_tools.probability_distribution import ProbabilityDistribution


class ProbabilityDistributionTests(TestCase):
    def test_expected_value_with_one_variable(self):
        # Given
        probability_distribution = ProbabilityDistribution.from_variables_samples(
            pd.Series([1, 1, 3, 4, 6], name="X")
        )
        # When
        expected_values = probability_distribution.get_expected_value("X")
        # Then
        self.assertEqual(expected_values, 3)

    def test_expected_value_with_two_variables(self):
        # Given
        probability_distribution = ProbabilityDistribution.from_variables_samples(
            pd.Series([1, 1, 3, 4, 5], name="X"),
            pd.Series([1, 1, 2, 1, 2], name="Y")
        )
        # When
        expected_values = probability_distribution.get_expected_value("X")
        # Then
        assert_series_equal(
            expected_values,
            pd.Series([2.0, 4], index=[1, 2], name="X").rename_axis("Y")
        )
