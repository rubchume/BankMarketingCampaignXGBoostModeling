from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .probability_distribution import ProbabilityDistribution
from .statistical_tools import get_confidence_intervals_for_binomial_variable


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


def feature_histogram_plus_target_positive_rate(
        df,
        feature,
        binary_target,
        clip_outliers_max_deviation=None,
        number_of_bins=8,

):
    def add_interval_index_properties(df: pd.DataFrame):
        df["left"] = df.index.map(lambda interval: interval.left)
        df["right"] = df.index.map(lambda interval: interval.right)
        df["midpoint"] = df.index.map(lambda interval: interval.mid)
        df["width"] = df.index.map(lambda interval: interval.length).astype(float)
        return df
    
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

    if clip_outliers_max_deviation is not None:
        feature_series = clip_outliers(df[feature], clip_outliers_max_deviation)
    else:
        feature_series = df[feature]

    df_grouped = df.groupby(pd.qcut(feature_series, q=number_of_bins, duplicates="drop")).aggregate(
        freq=pd.NamedAgg(binary_target, "count"),
        positive_rate=pd.NamedAgg(binary_target, "mean")
    )
    df_grouped = add_interval_index_properties(df_grouped)
    df_grouped["probability_density"] = df_grouped.freq / df_grouped.freq.sum() / df_grouped.width

    return go.Figure(
        data=[
            go.Bar(x=df_grouped.midpoint, y=df_grouped.probability_density, width=df_grouped.width, yaxis="y", showlegend=False, text=df_grouped.freq),
            go.Scatter(x=df_grouped.midpoint, y=df_grouped.positive_rate, yaxis="y2", showlegend=False, hovertemplate="%{y:.1%}"),
        ],
        layout=go.Layout(
            xaxis_title=feature,
            yaxis=dict(
                title=f"{feature} probability density",
                titlefont=dict(
                    color=px.colors.qualitative.Plotly[0]
                ),
                side="left",
                tickfont_color=px.colors.qualitative.Plotly[0]
            ),
            yaxis2=dict(
                title=f"{binary_target} positive rate",
                range=[0, df_grouped.positive_rate.max()],
                anchor="x",
                overlaying="y",
                side="right",
                titlefont=dict(
                    color=px.colors.qualitative.Plotly[1]
                ),
                tickfont_color=px.colors.qualitative.Plotly[1],
                tickmode="sync",
                tickformat = '.0%'
            ),
        )
    )


def plot_categorical_variable_vs_binary_target(
    categorical_feature: pd.Series,
    binary_target: pd.Series,
    feature_name: str = None,
    feature_count_name="Count",
    target_average_name="Target average"
):
    df = pd.concat([categorical_feature, binary_target], axis="columns", ignore_index=True).set_axis(("feature", "label"), axis="columns")
    df_grouped = df.groupby("feature").aggregate(freq=pd.NamedAgg("label", "count"), avg=pd.NamedAgg("label", "mean")).sort_values("avg", ascending=False)
    
    return go.Figure(
        data = [
            go.Bar(
                x=df_grouped.index,
                y=df_grouped.freq,
                texttemplate="%{y}",
                showlegend=False,
                yaxis="y"
            ),
            go.Scatter(
                x=df_grouped.index,
                y=df_grouped.avg,
                showlegend=False,
                yaxis="y2"
            )
        ],
        layout=go.Layout(
            xaxis_title=feature_name or categorical_feature.name,
            yaxis=dict(
                title=feature_count_name,
                titlefont_color=px.colors.qualitative.Plotly[0],
                side="left"
            ),
            yaxis2=dict(
                title=target_average_name,
                anchor="x",
                overlaying="y",
                side="right",
                titlefont_color=px.colors.qualitative.Plotly[1],
                tickmode="sync",
                tickformat = '.0%',
                range=[0, df_grouped.avg.max() * 1.1]
            ),
        )
    )


def plot_numerical_variable_vs_binary_target(
    numerical_feature: pd.Series,
    binary_target: pd.Series,
    feature_name: str = None,
    feature_count_name="Count",
    target_average_name="Target average",
    confidence_intervals_significance_level=None
):  
    distribution = ProbabilityDistribution.from_variables_samples(numerical_feature.rename("feature"), binary_target.rename("target"))
    
    freq = distribution.get_marginal_count("feature")
    average = distribution.conditioned_on("feature").given(target=True).distribution
    df_grouped = pd.DataFrame({"freq": freq, "avg": average})
    
    if confidence_intervals_significance_level is not None:
        confidence_intervals = get_confidence_intervals_for_binomial_variable(distribution, "target", confidence_intervals_significance_level)
        error_y = dict(
            type='data',
            symmetric=False,
            array=confidence_intervals.High - average,
            arrayminus=average - confidence_intervals.Low
        )
    else:
        error_y = None
    
    return go.Figure(
        data = [
            go.Bar(
                x=df_grouped.index,
                y=df_grouped.freq,
                texttemplate="%{y}",
                showlegend=False,
                yaxis="y"
            ),
            go.Scatter(
                x=df_grouped.index,
                y=df_grouped.avg,
                showlegend=False,
                yaxis="y2",
                error_y=error_y
            )
        ],
        layout=go.Layout(
            xaxis_title=feature_name or numerical_feature.name,
            yaxis=dict(
                title=feature_count_name,
                titlefont_color=px.colors.qualitative.Plotly[0],
                side="left"
            ),
            yaxis2=dict(
                title=target_average_name,
                anchor="x",
                overlaying="y",
                side="right",
                titlefont_color=px.colors.qualitative.Plotly[1],
                tickmode="sync",
                tickformat = '.0%',
                range=[0, df_grouped.avg.max() * 1.1]
            ),
        )
    )


def marimekko(
    df: pd.DataFrame,
    x_feature: str,
    y_feature: str,
    binary_target: str
):
    def sort_grouped_dataframe_by_positive_rate(grouped_dataframe):
        x_marginal = grouped_dataframe.groupby(level=x_feature).freq.sum()
        x_positives = grouped_dataframe.groupby(level=x_feature).freq_positive.sum()
        x_positive_rate = (x_positives / x_marginal).sort_values(ascending=False)
        x_values = x_positive_rate.index

        grouped_dataframe = grouped_dataframe.reindex(x_values, level=x_feature)
        
        y_marginal = grouped_dataframe.groupby(level=y_feature).freq.sum()
        y_positives = grouped_dataframe.groupby(level=y_feature).freq_positive.sum()
        y_positive_rate = (y_positives / y_marginal).sort_values(ascending=False)
        y_values = y_positive_rate.index
        
        grouped_dataframe = grouped_dataframe.reindex(y_values, level=y_feature)
        
        return grouped_dataframe
        
    df_grouped = (
        df.groupby(by=[x_feature, y_feature])
        .aggregate(
            freq=pd.NamedAgg(binary_target, "count"),
            freq_positive=pd.NamedAgg(binary_target, "sum"),
            positive_rate=pd.NamedAgg(binary_target, lambda s: s.mean() * 100)
        )
    )

    df_grouped[f"{y_feature}_percentage_conditioned_to_{x_feature}"] = df_grouped.groupby(level=x_feature).freq.transform(lambda s: s / s.sum()) * 100
    
    df_grouped = sort_grouped_dataframe_by_positive_rate(df_grouped)
    
    x_marginal = df_grouped.groupby(level=x_feature, sort=False).freq.sum()
    
    y_values = df_grouped.index.get_level_values(y_feature).unique()
    
    bar_widths = x_marginal
    bar_rigths = bar_widths.cumsum()
    bar_lefts = bar_rigths - bar_widths
    bar_middles = bar_rigths - bar_widths / 2
    
    cmin = df_grouped.positive_rate.min()
    cmax = df_grouped.positive_rate.max()
    
    pattern_shapes = [None, "/", "\\", ".", "x", "+", "-", "|"]
    
    return go.Figure(
        data=[
            go.Bar(
                x=bar_lefts,
                y=df_y[f"{y_feature}_percentage_conditioned_to_{x_feature}"],
                width=bar_widths,
                offset=0,
                name=y_value,
                marker=dict(
                    pattern_shape=pattern_shapes[i],
                    color=df_y["positive_rate"],
                    colorscale="Viridis",
                    cmin=cmin,
                    cmax=cmax,
                ),
                customdata=df_y[["freq", "positive_rate"]],
                texttemplate="%{customdata[0]}<br>Rate: %{customdata[1]:.1f}%",
                hovertemplate="Count: %{customdata[0]}<br>Positive rate: %{customdata[1]:.2f}%",
            ) for i, (y_value, df_y) in enumerate(df_grouped.groupby(level=y_feature, sort=False))
        ],
        layout=go.Layout(
            barmode="stack",
            xaxis=dict(
                title=x_feature,
                tickvals=bar_middles,
                ticktext=x_marginal.index
            ),
            yaxis=dict(
                title=y_feature,
                showticklabels=False
            ),
            height=600
        )
    )


def barchart_100_percent(x_feature, y_feature, binary_target):
    conditional_distribution = get_distribution([y_feature], conditioned_on=[x_feature])
    positive_rate = get_distribution([binary_target], conditioned_on=[x_feature, y_feature]).loc[(slice(None), slice(None), True)] * 100
    df_grouped = pd.concat([conditional_distribution, positive_rate], axis="columns").set_axis(["y_given_x", "positive_rate"], axis="columns")

    x_order = get_distribution([binary_target], conditioned_on=[x_feature]).loc[(slice(None), True)].sort_values(ascending=False).index
    y_order = get_distribution([binary_target], conditioned_on=[y_feature]).loc[(slice(None), True)].sort_values(ascending=False).index
    df_grouped = df_grouped.reindex(index=x_order, level=x_feature.name).reindex(index=y_order, level=y_feature.name)
    
    pattern_shapes = [None, "/", "\\", ".", "x", "+", "-", "|"]

    return go.Figure(
            data=[
                go.Bar(
                    x=df_given_x.index.get_level_values(x_feature.name),
                    y=df_given_x.y_given_x,
                    name=y_value,
                    marker=dict(
                        pattern_shape=pattern_shapes[i % len(pattern_shapes)],
                        color=df_given_x.positive_rate,
                        colorscale="Viridis",
                        cmin=positive_rate.min(),
                        cmax=positive_rate.max(),
                    ),
                    customdata=df_given_x.positive_rate,
                    texttemplate="Rate: %{customdata:.1f}%",
                    hovertemplate="Positive rate: %{customdata:.2f}%",
                ) for i, (y_value, df_given_x) in enumerate(df_grouped.groupby(level=y_feature.name, sort=False))
            ],
            layout=go.Layout(
                barmode="stack",
                xaxis_title=x_feature.name,
                yaxis_title=y_feature.name,
                height=600
            )
        )
