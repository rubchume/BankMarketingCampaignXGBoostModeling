from typing import List, Optional, Union

import pandas as pd


class ProbabilityDistribution:
    def __init__(self, samples: Optional[pd.DataFrame] = None, count: Optional[pd.Series] = None):
        if count is None:
            self.samples = samples
            self.distribution = self.get_count_from_samples()
        else:
            self.distribution = count
        
    @property
    def variables(self) -> List[str]:
        return list(self.samples.columns)
    
    @classmethod
    def from_variables_samples(cls, *variables_samples: List[pd.Series]):
        """Variables must have the same indices"""
        return cls(pd.concat(variables_samples, axis="columns"))
        
    def get_count_from_samples(self, variables: List[str] = None) -> pd.Series:
        return self.samples.groupby(by=variables or self.variables).size()
    
    def get_marginal_count(self, variables: Union[str, List[str]]) -> pd.Series:
        return self.distribution.groupby(level=variables).sum()
    
    def get_marginal_probability(self, variables: Union[str, List[str]]) -> pd.Series:
        return self.get_marginal_count(variables).transform(lambda s: s / s.sum())
    
    def select_variables(self, variables: List[str]):
        return type(self)(count=self.get_marginal_probability(variables))
    
    def given(self, **conditions):
        """Slice"""
        sliced = self.distribution
        for variable, value in conditions.items():
            if isinstance((values := value), list):
                sliced = sliced.loc[sliced.index.get_level_values(variable).isin(values)]
            else:
                sliced = sliced.xs(value, level=variable, axis="index")
        
        return type(self)(count=sliced)
    
    def conditioned_on(self, conditioned_on):
        return type(self)(count=self.distribution.groupby(level=conditioned_on).transform(lambda s: s / s.sum()))