from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.preprocessing import FEATURE_ORDER


MODEL_FILENAMES = {
    "eui": "ebm_eui.pkl",
    "cooling": "ebm_cooling.pkl",
    "heating": "ebm_heating.pkl",
}


def load_model_bundle(models_dir: Path) -> dict[str, dict[str, object]]:
    bundles: dict[str, dict[str, object]] = {}
    for key, filename in MODEL_FILENAMES.items():
        pipeline = joblib.load(models_dir / filename)
        bundles[key] = {
            "pipeline": pipeline,
            "scaler": pipeline.named_steps["scaler"],
            "ebm": pipeline.named_steps["ebm"],
        }
    return bundles


def _aggregate_feature_importance(ebm, feature_names: list[str]) -> pd.DataFrame:
    importances = np.zeros(len(feature_names), dtype=float)
    for term_features, importance in zip(ebm.term_features_, ebm.term_importances()):
        share = float(importance) / len(term_features)
        for feature_index in term_features:
            importances[feature_index] += share
    return pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values("Importance", ascending=False)


def build_global_importance_figure(ebm, feature_names: list[str]) -> go.Figure:
    importance_df = _aggregate_feature_importance(ebm, feature_names)
    return px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Global Feature Importance",
        color="Importance",
        color_continuous_scale="Tealgrn",
    )


def build_pdp_figure(pipeline, input_frame: pd.DataFrame, feature_name: str, feature_ranges: dict[str, list[float]], points: int = 60) -> go.Figure:
    lower, upper = feature_ranges[feature_name]
    grid = np.linspace(lower, upper, points)
    scenario = pd.concat([input_frame] * points, ignore_index=True)
    scenario[feature_name] = grid
    predictions = pipeline.predict(scenario)
    figure = go.Figure()
    figure.add_trace(go.Scatter(x=grid, y=predictions, mode="lines", line=dict(color="#0f766e", width=3)))
    figure.update_layout(title=f"Partial Dependence Plot: {feature_name}", xaxis_title=feature_name, yaxis_title="Predicted response")
    return figure


def build_shape_function_figure(ebm, feature_names: list[str], feature_name: str) -> go.Figure:
    feature_index = feature_names.index(feature_name)
    global_explanation = ebm.explain_global(name="global")
    term_data = global_explanation.data(feature_index)
    x_values = term_data.get("names", [])
    y_values = term_data.get("scores", [])

    if len(x_values) == len(y_values) + 1:
        x_values = [(x_values[i] + x_values[i + 1]) / 2.0 for i in range(len(y_values))]
    else:
        x_values = x_values[: len(y_values)]

    figure = go.Figure()
    figure.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers", line=dict(color="#2563eb", width=3)))
    figure.update_layout(title=f"EBM Shape Function: {feature_name}", xaxis_title=feature_name, yaxis_title="Contribution")
    return figure


def build_local_explanation_figure(pipeline, ebm, input_frame: pd.DataFrame) -> go.Figure:
    scaled_input = pipeline.named_steps["scaler"].transform(input_frame)
    local_explanation = ebm.explain_local(scaled_input, name="local")
    data = local_explanation.data(0)
    names = data.get("names", [])
    scores = data.get("scores", [])
    figure = px.bar(
        x=scores,
        y=names,
        orientation="h",
        title="Local Explanation for Current Design",
        color=scores,
        color_continuous_scale="RdBu",
    )
    figure.update_layout(xaxis_title="Contribution", yaxis_title="Term")
    return figure


def explain_prediction_text(pipeline, ebm, input_frame: pd.DataFrame) -> str:
    scaled_input = pipeline.named_steps["scaler"].transform(input_frame)
    local_explanation = ebm.explain_local(scaled_input, name="local")
    data = local_explanation.data(0)
    pairs = list(zip(data.get("names", []), data.get("scores", [])))
    pairs = sorted(pairs, key=lambda item: abs(item[1]), reverse=True)[:3]
    statements = []
    for term, score in pairs:
        direction = "increases" if score >= 0 else "decreases"
        statements.append(f"{term} {direction} the prediction by {abs(score):.2f} units.")
    return " ".join(statements)
