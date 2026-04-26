from __future__ import annotations

import json
from pathlib import Path

import logging
import pandas as pd


FEATURE_ORDER = [
    "Aspect Ratio",
    "Orientation",
    "WWR",
    "Glass U-factor",
    "SHGC",
    "Lighting Power Density",
    "EX-Wall Insulation Thickness",
]


def get_project_paths(app_file: Path) -> dict[str, Path]:
    base_dir = app_file.resolve().parent
    return {
        "base_dir": base_dir,
        "models_dir": base_dir / "models",
        "ranges_json": Path(app_file).parent / "data" / "feature_ranges.json",
        "logs": base_dir / "logs" / "app_usage.log",
        "shap_dir": base_dir / "shap_plots",
    }


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    logging.basicConfig(level=logging.INFO)


def load_feature_ranges(ranges_path: Path) -> dict[str, list[float]]:
    with ranges_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_feature_frame(user_inputs: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([[user_inputs[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)
