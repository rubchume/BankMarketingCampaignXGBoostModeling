from copy import deepcopy
from typing import List, Optional, Union

import pandas as pd


class ProbabilityDistribution:
    def __init__(self, samples: Optional[pd.DataFrame] = None, distribution: Optional[pd.Series] = None, **kwargs):
        if distribution is None:
            self.samples = samples
            self.distribution = self._get_count_from_samples()
        else:
            self.distribution = distribution

        self.categories_mapping = {}

        for key, value in kwargs.items():
            setattr(self, key, value)
        
    @property
    def variables(self) -> List[str]:
        return list(self.samples.columns)

    @property
    def current_variables(self) -> List[str]:
        return list(self.distribution.index.names)
    
    @classmethod
    def from_variables_samples(cls, *variables_samples: pd.Series):
        """Variables must have the same indices"""
        return cls(pd.concat(variables_samples, axis="columns"))

    def get_distribution(self):
        distribution = self.distribution.reset_index()
        distribution_variables = self.distribution.index.names
        for variable, intervals in self.categories_mapping.items():
            if variable in distribution_variables:
                distribution[variable] = intervals.cat.categories[distribution[variable]]

        return distribution.set_index(distribution_variables).iloc[:, 0]

    def select_variables(self, variables: List[str]):
        return self._create_copy(distribution=self.get_marginal_count(variables))

    def given(self, condition=None, **conditions):
        """Slice"""
        if condition is not None:
            conditions = condition

        sliced = self.distribution
        for variable, value in conditions.items():
            if isinstance((values := value), list):
                sliced = sliced.loc[sliced.index.get_level_values(variable).isin(values)]
            else:
                sliced = sliced.xs(value, level=variable, axis="index")

        return self._create_copy(distribution=sliced)

    def conditioned_on(self, level):
        return self._create_copy(distribution=self.distribution.groupby(level=level).transform(lambda s: s / s.sum()))

    def sort_by(self, level, ascending=True):
        return self._create_copy(distribution=self.distribution.sort_index(axis="index", level=level, ascending=ascending))

    def _get_count_from_samples(self, variables: List[str] = None) -> pd.Series:
        return self.samples.groupby(by=variables or self.variables).size()

    def _create_copy(self, **kwargs):
        copy = deepcopy(self)
        for key, value in kwargs.items():
            setattr(copy, key, value)

        return copy

    def apply_function_to_variable(self, variable, function):
        df = self.distribution.reset_index()
        df[variable] = function(df[variable])

        if df[variable].dtype == "category":
            self.categories_mapping[variable] = df[variable]
            df[variable] = df[variable].cat.codes

        self.distribution = df.set_index(self.current_variables).iloc[:, 0]

    def get_marginal_count(self, variables: Union[str, List[str]]) -> pd.Series:
        return self.distribution.groupby(level=variables).sum()
    
    def get_marginal_probability(self, variables: Union[str, List[str]]) -> pd.Series:
        return self.get_marginal_count(variables).transform(lambda s: s / s.sum())
