import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from eda_tools.probability_distribution import ProbabilityDistribution
from eda_tools.statistical_tools import get_confidence_intervals_for_binomial_variable


def bar(
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


def bar_true_target_vs_predicted(
        categorical_feature: pd.Series,
        binary_target: pd.Series,
        predicted_target: pd.Series,
        feature_name: str = None,
        feature_count_name="Count",
        target_average_name="Target average"
):
    df = pd.concat([categorical_feature, binary_target, predicted_target], axis="columns", ignore_index=True).set_axis(
        ("feature", "label", "predicted"), axis="columns")
    df_grouped = df.groupby("feature").aggregate(
        freq=pd.NamedAgg("label", "count"),
        avg=pd.NamedAgg("label", "mean"),
        avg_predicted=pd.NamedAgg("predicted", "mean")
    ).sort_values("avg", ascending=False)

    return go.Figure(
        data=[
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
            ),
            go.Scatter(
                x=df_grouped.index,
                y=df_grouped.avg_predicted,
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
                tickformat='.0%',
                range=[0, df_grouped.avg.max() * 1.1]
            ),
        )
    )


def bar_numerical(
        numerical_feature: pd.Series,
        binary_target: pd.Series,
        feature_name: str = None,
        feature_count_name="Count",
        target_average_name="Target average",
        confidence_intervals_significance_level=None
):
    distribution = ProbabilityDistribution.from_variables_samples(numerical_feature.rename("feature"),
                                                                  binary_target.rename("target"))

    freq = distribution.get_marginal_count("feature")
    average = distribution.conditioned_on("feature").given(target=True).distribution
    df_grouped = pd.DataFrame({"freq": freq, "avg": average})

    if confidence_intervals_significance_level is not None:
        confidence_intervals = get_confidence_intervals_for_binomial_variable(distribution, "target",
                                                                              confidence_intervals_significance_level)
        error_y = dict(
            type='data',
            symmetric=False,
            array=confidence_intervals.High - average,
            arrayminus=average - confidence_intervals.Low
        )
    else:
        error_y = None

    return go.Figure(
        data=[
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
                tickformat='.0%',
                range=[0, df_grouped.avg.max() * 1.1]
            ),
        )
    )
