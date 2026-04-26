from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


def _bounded_perturbation(value: float, lower: float, upper: float, factor: float) -> float:
    return float(min(max(value * factor, lower), upper))


def compute_sensitivity_table(pipeline, input_frame: pd.DataFrame, feature_ranges: dict[str, list[float]]) -> pd.DataFrame:
    baseline = float(pipeline.predict(input_frame)[0])
    rows = []
    for feature, (lower, upper) in feature_ranges.items():
        minus_frame = input_frame.copy()
        plus_frame = input_frame.copy()
        minus_frame.at[0, feature] = _bounded_perturbation(float(input_frame.at[0, feature]), lower, upper, 0.9)
        plus_frame.at[0, feature] = _bounded_perturbation(float(input_frame.at[0, feature]), lower, upper, 1.1)
        minus_prediction = float(pipeline.predict(minus_frame)[0])
        plus_prediction = float(pipeline.predict(plus_frame)[0])
        rows.append(
            {
                "Feature": feature,
                "Baseline EUI": baseline,
                "Minus 10% EUI": minus_prediction,
                "Plus 10% EUI": plus_prediction,
                "Sensitivity Span": plus_prediction - minus_prediction,
            }
        )
    return pd.DataFrame(rows).sort_values("Sensitivity Span", ascending=False)


def build_sensitivity_figure(pipeline, input_frame: pd.DataFrame, feature_ranges: dict[str, list[float]]):
    sensitivity_df = compute_sensitivity_table(pipeline, input_frame, feature_ranges)
    figure = px.bar(
        sensitivity_df,
        x="Feature",
        y="Sensitivity Span",
        color="Sensitivity Span",
        color_continuous_scale="Sunset",
        title="Local Sensitivity Analysis for EUI (±10% perturbation)",
    )
    figure.update_layout(yaxis_title="Prediction span", xaxis_title="Feature")
    return figure
