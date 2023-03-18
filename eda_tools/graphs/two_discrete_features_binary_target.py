import pandas as pd
import plotly.graph_objects as go

from eda_tools.probability_distribution import ProbabilityDistribution
from eda_tools.statistical_tools import get_distribution


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

    df_grouped[f"{y_feature}_percentage_conditioned_to_{x_feature}"] = df_grouped.groupby(
        level=x_feature).freq.transform(lambda s: s / s.sum()) * 100

    df_grouped = sort_grouped_dataframe_by_positive_rate(df_grouped)

    x_marginal = df_grouped.groupby(level=x_feature, sort=False).freq.sum()

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
    dist = ProbabilityDistribution.from_variables_samples(x_feature.rename("x"), y_feature.rename("y"), binary_target.rename("target"))
    conditional_distribution = dist.select_variables(["x", "y"]).conditioned_on("x").get_distribution()
    positive_rate = dist.conditioned_on(["x", "y"]).given(target=True).get_distribution() * 100
    df_grouped = pd.concat([conditional_distribution, positive_rate], axis="columns").set_axis(
        ["y_given_x", "positive_rate"], axis="columns")

    x_order = dist.select_variables(["x", "target"]).conditioned_on("x").given(target=True).get_distribution().sort_values(ascending=False).index
    y_order = dist.select_variables(["y", "target"]).conditioned_on("y").given(target=True).get_distribution().sort_values(ascending=False).index
    df_grouped = df_grouped.reindex(index=x_order, level="x").reindex(index=y_order, level="y")

    pattern_shapes = [None, "/", "\\", ".", "x", "+", "-", "|"]

    return go.Figure(
        data=[
            go.Bar(
                x=df_given_x.index.get_level_values("x"),
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
            ) for i, (y_value, df_given_x) in enumerate(df_grouped.groupby(level="y", sort=False))
        ],
        layout=go.Layout(
            barmode="stack",
            xaxis_title=x_feature.name,
            yaxis_title=y_feature.name,
            height=600
        )
    )
