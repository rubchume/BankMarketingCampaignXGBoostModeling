from enum import Enum
from typing import Callable, Optional, Literal, List

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import partial_dependence
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, log_loss
from sklearn.pipeline import Pipeline

from eda_tools.graphs.discrete_feature_binary_target import bar_true_target_vs_predicted
from eda_tools.statistical_tools import estimate_probability_density_function
from wrappers import dataframe_input


class BinaryClassifierEvaluator:
    class MetricType(Enum):
        SCORE = "SCORE"
        PREDICTION = "PREDICTION"
        CONFUSION_MATRIX = "CONFUSION_MATRIX"

    def __init__(self, estimator, X, y):
        self.estimator = estimator
        self.X = X
        self.y = y

    def get_predict_scores(self):
        return pd.Series(self.estimator.predict_proba(self.X)[:, 1], index=self.X.index)

    def get_predictions(self, threshold: Optional[float] = 0.5):
        return self.get_predict_scores() >= threshold

    def get_confusion_matrix_measures(self, threshold: Optional[float] = None):
        if threshold:
            return self._confusion_matrix_from_predictions(self.y, self.get_predictions(threshold))
        else:
            return self._confusion_matrices_for_all_thresholds(self.y, self.get_predict_scores())

    def calculate_score_metrics(self, **metrics: Callable[[pd.Series, pd.Series], float]):
        y_scores = self.get_predict_scores()
        return {
            metric_name: metric_function(self.y, y_scores)
            for metric_name, metric_function in metrics.items()
        }

    def calculate_prediction_metrics(self, threshold: Optional[float] = 0.5, **metrics: Callable[[pd.Series, pd.Series], float]):
        y_predicted = self.get_predictions(threshold)
        return {
            metric_name: metric_function(self.y, y_predicted)
            for metric_name, metric_function in metrics.items()
        }

    def calculate_confusion_matrix_metrics(self, threshold: Optional[float] = None, **metrics: Callable):
        num_samples = self.X.shape[0]
        num_positive_samples = self.y.sum()
        num_negative_samples = num_samples - num_positive_samples

        if threshold:
            return pd.Series({
                metric_name: dataframe_input(metric_function)(
                    self.get_confusion_matrix_measures(threshold),
                    P=num_positive_samples,
                    N=num_negative_samples,
                )
                for metric_name, metric_function in metrics.items()
            })
        else:
            confusion_matrix_parameters = self.get_confusion_matrix_measures()

            metric_results = pd.DataFrame({
                metric_name: dataframe_input(metric_function)(
                    confusion_matrix_parameters,
                    P=num_positive_samples,
                    N=num_negative_samples,
                )
                for metric_name, metric_function in metrics.items()
            })

            return pd.concat([confusion_matrix_parameters, metric_results], axis="columns")

    def calculate_optimum_for_confusion_matrix_metric(self, metric: Callable[[pd.Series, pd.Series], float]):
        confusion_matrix_metrics = self.calculate_confusion_matrix_metrics(metric=metric)
        return (
            confusion_matrix_metrics.metric.max(),
            confusion_matrix_metrics.score[confusion_matrix_metrics.metric.idxmax()]
        )

    def calculate_optimum_cross_entropy_for_features(self, features: List[str] = None, pipeline_step=None, feature_values=None):
        if feature_values is None:
            if features is None:
                target_prob = self.y.mean()
                return -(target_prob * np.log(target_prob) + (1 - target_prob) * np.log(1 - target_prob))
            else:
                if pipeline_step is not None:
                    feature_values = self._get_feature_after_pipeline_step(features, pipeline_step)
                else:
                    feature_values = self.X[features]

        feature_distribution = feature_values.value_counts() / len(feature_values)
        target_conditional_prob = self.y.groupby(by=feature_values).mean()

        return - sum(feature_distribution * (
            target_conditional_prob * np.log(target_conditional_prob)
            + (1 - target_conditional_prob) * np.log(1 - target_conditional_prob)
        ))

    def visualize_score_to_confusion_matrix_metrics(
            self,
            x_scaling: Literal["threshold", "quantile"] = "quantile",
            threshold: float = 0.5,
            plot_probability_densities=False,
            **metrics: Callable
    ):
        def probability_density_traces():
            pdf_given_negative = estimate_probability_density_function(
                x[~confusion_matrix_metrics.target],
                bandwidth=0.025
            )(x)

            pdf_given_positive = estimate_probability_density_function(
                x[confusion_matrix_metrics.target],
                bandwidth=0.025
            )(x)

            return [
                go.Scatter(x=x, y=pdf_given_negative, name="Negative", yaxis="y2"),
                go.Scatter(x=x, y=pdf_given_positive, name="Positive", yaxis="y2"),
            ]

        confusion_matrix_metrics = self.calculate_confusion_matrix_metrics(**metrics)

        if x_scaling == "threshold":
            x = confusion_matrix_metrics.score
            x_axis_name = "Threshold"
        else:
            num_samples = len(confusion_matrix_metrics)
            x = pd.Series(range(num_samples), index=confusion_matrix_metrics.index) / num_samples
            x_axis_name = "Quantile"

        metric_lines = [
            go.Scatter(x=x, y=confusion_matrix_metrics[metric], name=metric, yaxis="y")
            for metric in metrics.keys()
        ]

        fig = go.Figure(
            data=metric_lines + probability_density_traces() if plot_probability_densities else metric_lines,
            layout=go.Layout(
                xaxis=dict(title=x_axis_name, range=[0, 1]),
                yaxis=dict(title="Metric value"),
                yaxis2=dict(title="Class Probability density", overlaying="y", side="right"),
                hovermode="x unified",
            )
        )

        threshold_index = confusion_matrix_metrics.score.ge(threshold).idxmax()
        threshold_x_position = x[threshold_index]
        fig.add_vline(x=threshold_x_position, line_width=3, line_dash="dash", line_color="green")
        fig.add_annotation(
            x=threshold_x_position,
            y=0,
            text=f"Threshold: {threshold:.2e}",
            showarrow=True,
            arrowhead=1
        )

        return fig

    def visualize_confusion_matrix_metric_to_metric(
            self,
            threshold: Optional[float] = 0.5,
            title: Optional[str] = None,
            **metrics: Callable,
    ):
        confusion_matrix_metrics = self.calculate_confusion_matrix_metrics(**metrics)

        threshold_index = confusion_matrix_metrics.score.ge(threshold).idxmax()

        metric_x_name, metric_y_name = list(metrics.keys())

        return go.Figure(
            data=[
                go.Scatter(
                    x=confusion_matrix_metrics[metric_x_name],
                    y=confusion_matrix_metrics[metric_y_name],
                    mode="lines",
                    name=title,
                ),
                go.Scatter(
                    x=[confusion_matrix_metrics[metric_x_name][threshold_index]],
                    y=[confusion_matrix_metrics[metric_y_name][threshold_index]],
                    marker=dict(
                        color=px.colors.qualitative.Plotly[0]
                    ),
                    name="Threshold",
                )
            ],
            layout=go.Layout(
                title=title,
                xaxis=dict(title=metric_x_name),
                yaxis=dict(title=metric_y_name, scaleanchor="x", scaleratio=1),
            )
        )

    def visualize_confusion_matrix(self, threshold=0.5):
        y_pred = self.get_predictions(threshold)
        cm = confusion_matrix(self.y, y_pred)
        return ConfusionMatrixDisplay(cm)

    def visualize_features_importance(self):
        features_importance = self._get_features_importance()
        return px.bar(features_importance.reset_index(), y="feature", x="importance", orientation='h')

    def visualize_partial_dependence_plot(self, feature_name, pipeline_step=None):
        if pipeline_step is not None:
            pipeline_step_index = list(self.estimator.named_steps.keys()).index(pipeline_step)
            partial_pipeline = self.estimator[:pipeline_step_index + 1]
            features = partial_pipeline.transform(self.X)
            complementary_pipeline = self.estimator[pipeline_step_index + 1:]
        else:
            features = self.X
            complementary_pipeline = self.estimator

        partial_dependence_values = partial_dependence(complementary_pipeline, features, feature_name)
        df = pd.DataFrame({feature_name: partial_dependence_values["values"][0], "average": partial_dependence_values["average"][0]})
        return px.line(df, x=feature_name, y="average").update_layout(title="Partial dependence plot for " + feature_name)

    def visualize_predicted_vs_observed(self, feature=None, pipeline_step=None, feature_values=None, threshold=0.5):
        if feature_values is None:
            if pipeline_step is not None:
                feature_values = self._get_feature_after_pipeline_step(feature, pipeline_step)
            else:
                feature_values = self.X[feature]

        return bar_true_target_vs_predicted(
            feature_values,
            self.y,
            self.get_predictions(threshold)
        ).update_layout(title=f"Predicted vs observed for {feature}", yaxis2_range=[0, None])

    def visualize_predicted_proba_vs_observed(self, feature=None, pipeline_step=None, feature_values=None):
        def get_cross_entropy():
            return log_loss(self.y, self.get_predict_scores())

        if feature_values is None:
            if pipeline_step is not None:
                feature_values = self._get_feature_after_pipeline_step(feature, pipeline_step)
            else:
                feature_values = self.X[feature]

        cross_entropy = get_cross_entropy()
        optimum_cross_entropy = self.calculate_optimum_cross_entropy_for_features(feature_values=feature_values)

        return bar_true_target_vs_predicted(
            feature_values,
            self.y,
            self.get_predict_scores()
        ).update_layout(
            title=f"Average predicted probability vs observed for {feature}. Cross-entropy: {cross_entropy:.2f} (Optimum: {optimum_cross_entropy:.2f})",
            yaxis2_range=[0, None]
        )

    def _get_feature_after_pipeline_step(self, features, pipeline_step):
        if not isinstance(self.estimator, Pipeline):
            raise ValueError("Estimator is not a pipeline")

        if pipeline_step not in self.estimator.named_steps:
            raise ValueError(f"Pipeline does not contain step {pipeline_step}")

        step_index = list(self.estimator.named_steps.keys()).index(pipeline_step)
        previous_steps_pipeline = self.estimator[:step_index + 1]

        return previous_steps_pipeline.transform(self.X)[features].set_axis(self.X.index, axis="index")

    def _get_features_importance(self):
        if isinstance(self.estimator, Pipeline):
            estimator = self.estimator.named_steps["classifier"]

        return pd.Series(
            estimator.feature_importances_,
            index=estimator.feature_names_in_,
            name="importance"
        ).rename_axis(index="feature").sort_values(ascending=True)

    @staticmethod
    def _confusion_matrix_from_predictions(target, predicted_target):
        return pd.Series(dict(
            TP=(predicted_target & target).sum(),
            FP=(predicted_target & ~target).sum(),
            TN=(~predicted_target & ~target).sum(),
            FN=(~predicted_target & target).sum()
        ))

    @staticmethod
    def _confusion_matrices_for_all_thresholds(target, target_scores):
        num_samples = len(target)
        num_positive = target.sum()
        num_negative = num_samples - num_positive

        df = pd.DataFrame({"target": target, "score": target_scores})
        df = df.sort_values("score", ascending=False)

        df["TP"] = df.target.cumsum()
        df["FN"] = num_positive - df["TP"]
        df["FP"] = (~df.target).cumsum()
        df["TN"] = num_negative - df["FP"]

        return df.iloc[::-1]
