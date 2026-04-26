from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.explainability import (
    build_global_importance_figure,
    build_local_explanation_figure,
    build_pdp_figure,
    build_shape_function_figure,
    explain_prediction_text,
    load_model_bundle,
)
from utils.preprocessing import (
    FEATURE_ORDER,
    build_feature_frame,
    get_project_paths,
    load_feature_ranges,
    setup_logging,
)
from utils.sensitivity import build_sensitivity_figure, compute_sensitivity_table


st.set_page_config(
    page_title="Building Energy Performance Predictor",
    page_icon="🏢",
    layout="wide",
)

TARGET_OPTIONS = {
    "Total Energy Use Intensity (EUI)": "eui",
    "Cooling EUI": "cooling",
    "Heating EUI": "heating",
}

SCOPE_NOTICE = """Project Scope Notice
This machine learning model is valid only within the simulation framework used to generate the training dataset.

Assumptions:

• Building type: Three-story office building

• Climate zone: ASHRAE 3B (hot-dry)

• Operation schedules: Derived from Table P-3-5 of the Iranian National Building Regulations (Mabhas 19), office occupancy schedules

• HVAC System: Ideal Air Loads system with no economizer

• Domestic hot water system: Gas tankless heater, efficiency 0.85

• DHW reference temperature: 20°C

• DHW system loss coefficient: 1.3

All simulations, codes, and the Streamlit application were developed by:

Alireza Oroomiei

Dr. Morteza Rahbar

Dr. Mohamadali Khanmohamadi

School of Architecture, Iran University of Science and Technology"""


def render_sidebar(feature_ranges: dict[str, list[float]]) -> dict[str, float]:
    st.sidebar.header("Design Parameters")
    st.sidebar.caption("Use the sliders to define a valid office-building design candidate.")

    values: dict[str, float] = {}
    for feature in FEATURE_ORDER:
        lower, upper = map(float, feature_ranges[feature])
        default = round((lower + upper) / 2.0, 3)
        step = 0.01 if upper - lower < 1.0 else 0.05
        values[feature] = st.sidebar.slider(
            feature,
            min_value=lower,
            max_value=upper,
            value=default,
            step=step,
        )
    return values


def log_prediction(log_path: Path, inputs: dict[str, float], outputs: dict[str, float]) -> None:
    logger = logging.getLogger("mine_app")
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    logger.info("inputs=%s | predictions=%s", inputs, outputs)


def main() -> None:
    paths = get_project_paths(Path(__file__))
    setup_logging(paths["logs"])
    feature_ranges = load_feature_ranges(paths["ranges_json"])
    user_inputs = render_sidebar(feature_ranges)

    st.title("Building Energy Performance Predictor")
    st.markdown(
        "Predict total, cooling, and heating energy use intensity using trained Explainable Boosting Machine (EBM) regressors."
    )

    bundles = load_model_bundle(paths["models_dir"])
    input_frame = build_feature_frame(user_inputs)

    predict_clicked = st.button("Predict", type="primary")
    if predict_clicked:
        predictions = {
            "EUI": float(bundles["eui"]["pipeline"].predict(input_frame)[0]),
            "Cooling EUI": float(bundles["cooling"]["pipeline"].predict(input_frame)[0]),
            "Heating EUI": float(bundles["heating"]["pipeline"].predict(input_frame)[0]),
        }
        log_prediction(paths["logs"], user_inputs, predictions)

        metric_cols = st.columns(3)
        metric_cols[0].metric("EUI (kWh/m²·year)", f"{predictions['EUI']:.2f}")
        metric_cols[1].metric("Cooling EUI", f"{predictions['Cooling EUI']:.2f}")
        metric_cols[2].metric("Heating EUI", f"{predictions['Heating EUI']:.2f}")

        selection = st.selectbox(
            "Choose a target for the explainability dashboard",
            options=list(TARGET_OPTIONS.keys()),
            index=0,
        )
        target_key = TARGET_OPTIONS[selection]
        bundle = bundles[target_key]

        st.markdown("## Explainability Dashboard")
        tabs = st.tabs(
            [
                "Global Feature Importance",
                "Partial Dependence",
                "EBM Shape Functions",
                "Local Explanation",
                "Sensitivity Mini-Analysis",
            ]
        )

        with tabs[0]:
            st.plotly_chart(
                build_global_importance_figure(bundle["ebm"], FEATURE_ORDER),
                use_container_width=True,
            )

        with tabs[1]:
            feature_choice = st.selectbox(
                "Select a feature for PDP",
                options=FEATURE_ORDER,
                key="pdp_feature",
            )
            st.plotly_chart(
                build_pdp_figure(bundle["pipeline"], input_frame, feature_choice, feature_ranges),
                use_container_width=True,
            )

        with tabs[2]:
            shape_feature = st.selectbox(
                "Select a feature for the EBM shape function",
                options=FEATURE_ORDER,
                key="shape_feature",
            )
            st.plotly_chart(
                build_shape_function_figure(bundle["ebm"], FEATURE_ORDER, shape_feature),
                use_container_width=True,
            )

        with tabs[3]:
            st.plotly_chart(
                build_local_explanation_figure(bundle["pipeline"], bundle["ebm"], input_frame),
                use_container_width=True,
            )
            st.info(explain_prediction_text(bundle["pipeline"], bundle["ebm"], input_frame))

        with tabs[4]:
            st.plotly_chart(
                build_sensitivity_figure(bundle["pipeline"], input_frame, feature_ranges),
                use_container_width=True,
            )
            st.dataframe(
                compute_sensitivity_table(bundle["pipeline"], input_frame, feature_ranges),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown("---")
    st.text(SCOPE_NOTICE)


if __name__ == "__main__":
    main()
