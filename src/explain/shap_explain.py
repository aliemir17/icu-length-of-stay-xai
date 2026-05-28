"""SHAP-based global summary for the multi-model Pipeline.

Tree-only at the moment — uses `shap.TreeExplainer`. Multi-method XAI
(LIME + permutation) and non-tree estimators come in Phase 2.
"""
from __future__ import annotations

import argparse

import joblib
import matplotlib.pyplot as plt
import shap

from src.config import FIGURES_DIR, MODELS_DIR
from src.data.load import load_icu_stays
from src.features.build import split_xy


def explain(
    source: str = "mimic4",
    model_name: str = "xgb",
    task: str = "regression",
    n_sample: int = 500,
) -> None:
    model_path = MODELS_DIR / f"{model_name}_{task}_{source}.joblib"
    pipe = joblib.load(model_path)
    df = load_icu_stays(source)
    X, _, _, _ = split_xy(df)
    X_sample = X.sample(n=min(n_sample, len(X)), random_state=42)

    preprocess = pipe.named_steps["preprocess"]
    estimator = pipe.named_steps["model"]
    X_trans = preprocess.transform(X_sample)
    feature_names = preprocess.get_feature_names_out()

    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer(X_trans)
    shap_values.feature_names = list(feature_names)

    shap.summary_plot(shap_values, X_trans, feature_names=feature_names, show=False)
    out = FIGURES_DIR / f"shap_summary_{model_name}_{task}_{source}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved SHAP summary -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mimic4", choices=["synthetic", "mimic4"])
    parser.add_argument(
        "--model",
        default="xgb",
        choices=["xgb", "rf", "lgbm"],
        help="Tree models only for now; linear support arrives in Phase 2.",
    )
    parser.add_argument(
        "--task",
        default="regression",
        choices=["regression", "regression_raw", "classification"],
    )
    args = parser.parse_args()
    explain(args.source, args.model, args.task)