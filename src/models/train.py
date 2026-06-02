"""Multi-model trainer for ICU LoS prediction.

Trains three task heads — `regression` (log1p target), `regression_raw`
(no transform), and `classification` (7-day threshold) — across four
algorithms (Linear/Ridge, RF, XGBoost, LightGBM). Each (task, model)
combo gets 5-fold CV on the 80 % training split and a single final fit
on the full train set, evaluated on the untouched 20 % test set.

Outputs:
- `models/<model>_<task>_<source>.joblib` — fitted Pipeline
- `reports/tables/comparison_<task>_<source>.md` — markdown summary
- `reports/tables/results_<source>.json` — raw results
- `reports/figures/model_comparison_<task>_<source>.png` — bar chart

Usage:
    python -m src.models.train --source mimic4 --task all --model all
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_SEED, REPORTS_DIR, TARGET
from src.data.load import load_icu_stays
from src.features.build import build_preprocessor, split_xy

# Long-stay thresholds for binary classification (days)
# - 7.0  : clinical convention (weekly planning unit, Hempel et al.)
# - 4.06 : cohort mean LoS — statistically motivated, more balanced classes
LONG_STAY_DAYS = 7.0
LONG_STAY_DAYS_MEAN = 4.063534  # actual cohort mean (full MIMIC-IV 3.1)

TABLES_DIR = REPORTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ("regression", "regression_raw", "classification", "classification_mean")
MODELS = ("linear", "rf", "xgb", "lgbm")

# Default placeholder; the actual scale_pos_weight is now computed dynamically
# from the training y at fit time (because the two classification thresholds
# yield very different class ratios). See _compute_spw().
_SCALE_POS_WEIGHT = 6.81


# ---------------------------------------------------------------------------
# Model + pipeline factory
# ---------------------------------------------------------------------------

def make_model(name: str, task: str):
    """Return an unfitted sklearn-compatible estimator for the (model, task)."""
    is_clf = task in ("classification", "classification_mean")

    if name == "linear":
        if is_clf:
            return LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=RANDOM_SEED,
            )
        # RidgeCV auto-picks alpha per fold via generalized CV; fixed
        # alpha=1 produced runaway predictions on some folds, blowing up the
        # log1p->expm1 inversion. The grid covers under- to over-regularized.
        return RidgeCV(alphas=(0.1, 1.0, 10.0, 100.0, 1000.0))

    if name == "rf":
        common = dict(
            n_estimators=100, max_depth=18, max_features=0.3,
            min_samples_leaf=8, min_samples_split=4,
            n_jobs=-1, random_state=RANDOM_SEED,
        )
        if is_clf:
            return RandomForestClassifier(class_weight="balanced", **common)
        return RandomForestRegressor(**common)

    if name == "xgb":
        if is_clf:
            return XGBClassifier(
                n_estimators=950, max_depth=9, learning_rate=0.0144,
                min_child_weight=36, subsample=0.681, colsample_bytree=0.881,
                reg_alpha=0.0195, reg_lambda=6.96,
                scale_pos_weight=_SCALE_POS_WEIGHT, eval_metric="logloss",
                random_state=RANDOM_SEED, n_jobs=-1,
            )
        return XGBRegressor(
            n_estimators=950, max_depth=6, learning_rate=0.0285,
            min_child_weight=1, subsample=0.783, colsample_bytree=0.713,
            reg_alpha=6.904, reg_lambda=0.00112,
            random_state=RANDOM_SEED, n_jobs=-1,
        )

    if name == "lgbm":
        if is_clf:
            return LGBMClassifier(
                n_estimators=800, max_depth=10, learning_rate=0.0165,
                num_leaves=65, min_child_samples=68,
                subsample=0.919, colsample_bytree=0.728,
                reg_alpha=0.00111, reg_lambda=9.449,
                class_weight="balanced",
                random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
            )
        return LGBMRegressor(
            n_estimators=900, max_depth=9, learning_rate=0.0224,
            num_leaves=76, min_child_samples=89,
            subsample=0.779, colsample_bytree=0.563,
            reg_alpha=0.0692, reg_lambda=0.00445,
            random_state=RANDOM_SEED, n_jobs=-1, verbose=-1,
        )

    raise ValueError(f"Unknown model name: {name!r}")


def make_pipeline(
    model_name: str, task: str, numeric_cols: list[str], categorical_cols: list[str]
) -> Pipeline:
    """Build a Pipeline with named steps 'preprocess' and 'model'.

    The 'model' step name is intentionally generic — SHAP / permutation /
    LIME code can locate the estimator without branching on model type.
    """
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", make_model(model_name, task)),
        ]
    )


# ---------------------------------------------------------------------------
# Target preparation and inversion
# ---------------------------------------------------------------------------

def prepare_target(y_days: pd.Series, task: str) -> pd.Series:
    """Apply the task-specific target transform on the raw LoS-in-days series."""
    if task == "regression":
        return np.log1p(y_days)
    if task == "regression_raw":
        return y_days.copy()
    if task == "classification":
        return (y_days >= LONG_STAY_DAYS).astype(int)
    if task == "classification_mean":
        return (y_days >= LONG_STAY_DAYS_MEAN).astype(int)
    raise ValueError(f"Unknown task: {task!r}")


# Cap on log-space regression predictions before expm1 inversion.
# log1p(500) ~ 6.2 ; observed LoS maxes at ~160 days, so this never affects
# realistic predictions but prevents float64 overflow when a poorly
# regularized linear model emits extreme values in a CV fold.
_LOG1P_PRED_CAP = float(np.log1p(500))


def to_days(y_transformed, task: str):
    """Invert the target transform back to days, for metric reporting."""
    if task == "regression":
        return np.expm1(np.clip(y_transformed, a_min=None, a_max=_LOG1P_PRED_CAP))
    if task == "regression_raw":
        return y_transformed
    raise ValueError(f"to_days not defined for task {task!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true_days, y_pred_days) -> dict:
    """All values in original-day units."""
    return {
        "mae": float(mean_absolute_error(y_true_days, y_pred_days)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_days, y_pred_days))),
        "r2": float(r2_score(y_true_days, y_pred_days)),
        "mape": float(mean_absolute_percentage_error(y_true_days, y_pred_days)),
    }


def classification_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "auc_pr": float(average_precision_score(y_true, y_proba)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def _is_classification(task: str) -> bool:
    return task in ("classification", "classification_mean")


def score_on(pipe, X, y_true, task: str) -> dict:
    """Predict + score, taking care of the regression target inversion."""
    if _is_classification(task):
        y_pred = pipe.predict(X)
        y_proba = pipe.predict_proba(X)[:, 1]
        return classification_metrics(y_true, y_pred, y_proba)
    y_pred_days = to_days(pipe.predict(X), task)
    y_true_days = to_days(y_true, task) if task == "regression" else np.asarray(y_true)
    return regression_metrics(y_true_days, y_pred_days)


# ---------------------------------------------------------------------------
# CV + train + eval one combo
# ---------------------------------------------------------------------------

def cv_score(pipe_factory, X, y, task: str, n_splits: int = 5) -> list[dict]:
    """5-fold CV with task-appropriate splitter. Re-instantiates the pipeline per fold."""
    if _is_classification(task):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_y = y
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_y = None

    folds = []
    for fold, (tr, va) in enumerate(splitter.split(X, split_y)):
        pipe = pipe_factory()
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        y_tr, y_va = y.iloc[tr], y.iloc[va]
        pipe.fit(X_tr, y_tr)
        m = score_on(pipe, X_va, y_va.values, task)
        m["fold"] = fold
        folds.append(m)
    return folds


@dataclass
class RunResult:
    source: str
    task: str
    model: str
    cv_mean: dict
    cv_std: dict
    test: dict
    train_size: int
    test_size: int
    feature_count: int


def train_one(source: str, task: str, model_name: str) -> RunResult:
    df = load_icu_stays(source)
    X, y_raw, numeric_cols, categorical_cols = split_xy(df)
    y = prepare_target(y_raw, task)

    stratify = y if _is_classification(task) else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify,
    )

    # For XGB classifier, override scale_pos_weight from the actual training
    # class ratio (the two classification tasks have very different ratios:
    # 87/13 at 7-day vs ~60/40 at 4.06-day mean threshold).
    def _factory():
        pipe = make_pipeline(model_name, task, numeric_cols, categorical_cols)
        if model_name == "xgb" and _is_classification(task):
            n_pos = max(int(y_tr.sum()), 1)
            n_neg = len(y_tr) - n_pos
            pipe.named_steps["model"].set_params(scale_pos_weight=n_neg / n_pos)
        return pipe

    folds = cv_score(_factory, X_tr, y_tr, task)

    metric_keys = [k for k in folds[0] if k != "fold"]
    cv_mean = {k: float(np.mean([f[k] for f in folds])) for k in metric_keys}
    cv_std = {k: float(np.std([f[k] for f in folds])) for k in metric_keys}

    # Final refit on full train, evaluate on held-out test
    pipe = _factory()
    pipe.fit(X_tr, y_tr)
    test_metrics = score_on(pipe, X_te, y_te.values, task)

    out = MODELS_DIR / f"{model_name}_{task}_{source}.joblib"
    joblib.dump(pipe, out)

    headline_metric = next(iter(test_metrics))
    print(
        f"  {model_name:<6} / {task:<15}  "
        f"cv[{headline_metric}]={cv_mean[headline_metric]:.3f}+-{cv_std[headline_metric]:.3f}  "
        f"test[{headline_metric}]={test_metrics[headline_metric]:.3f}  "
        f"-> {out.name}"
    )

    return RunResult(
        source=source,
        task=task,
        model=model_name,
        cv_mean=cv_mean,
        cv_std=cv_std,
        test=test_metrics,
        train_size=len(X_tr),
        test_size=len(X_te),
        feature_count=X.shape[1],
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_tables(results: list[RunResult], source: str) -> None:
    by_task: dict[str, list[RunResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)

    for task, rows in by_task.items():
        rows = sorted(rows, key=lambda r: r.model)
        metric_keys = list(rows[0].test.keys())

        lines = [
            f"# Model comparison — {task}  ({source})",
            "",
            f"`N_train = {rows[0].train_size}`, "
            f"`N_test = {rows[0].test_size}`, "
            f"`features = {rows[0].feature_count}`. "
            "CV = 5-fold on the training split; test = single held-out 20 % split. "
            "Random seed = 42.",
            "",
        ]
        header = ["Model"]
        header += [f"{k} (CV mean ± std)" for k in metric_keys]
        header += [f"{k} (test)" for k in metric_keys]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for r in rows:
            cells = [r.model]
            cells += [f"{r.cv_mean[k]:.3f} ± {r.cv_std[k]:.3f}" for k in metric_keys]
            cells += [f"{r.test[k]:.3f}" for k in metric_keys]
            lines.append("| " + " | ".join(cells) + " |")

        out = TABLES_DIR / f"comparison_{task}_{source}.md"
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"  wrote {out.relative_to(REPORTS_DIR.parent)}")

    json_out = TABLES_DIR / f"results_{source}.json"
    json_out.write_text(
        json.dumps([asdict(r) for r in results], indent=2), encoding="utf-8"
    )
    print(f"  wrote {json_out.relative_to(REPORTS_DIR.parent)}")


def plot_comparison(results: list[RunResult], source: str) -> None:
    by_task: dict[str, list[RunResult]] = {}
    for r in results:
        by_task.setdefault(r.task, []).append(r)

    for task, rows in by_task.items():
        rows = sorted(rows, key=lambda r: r.model)
        models = [r.model for r in rows]
        metric_keys = list(rows[0].cv_mean.keys())

        n_models = len(models)
        n_metrics = len(metric_keys)
        x = np.arange(n_models)
        width = 0.8 / n_metrics

        fig, ax = plt.subplots(figsize=(max(7, 1.3 * n_models * n_metrics), 4.5))
        for i, metric in enumerate(metric_keys):
            values = [r.cv_mean[metric] for r in rows]
            errors = [r.cv_std[metric] for r in rows]
            ax.bar(x + i * width, values, width, yerr=errors, capsize=3, label=metric)

        ax.set_xticks(x + width * (n_metrics - 1) / 2)
        ax.set_xticklabels(models)
        ax.set_ylabel("Score")
        ax.set_title(f"Model comparison — {task}  ({source}, 5-fold CV mean ± std)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        out = FIGURES_DIR / f"model_comparison_{task}_{source}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out.relative_to(REPORTS_DIR.parent)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mimic4", choices=["synthetic", "mimic4"])
    parser.add_argument("--task", default="all", choices=(*TASKS, "all"))
    parser.add_argument("--model", default="all", choices=(*MODELS, "all"))
    args = parser.parse_args()

    tasks = list(TASKS) if args.task == "all" else [args.task]
    models = list(MODELS) if args.model == "all" else [args.model]

    df = load_icu_stays(args.source)
    n_stays = len(df)
    print(
        f"Source: {args.source}  |  Stays: {n_stays}  |  "
        f"Tasks: {tasks}  |  Models: {models}"
    )

    results: list[RunResult] = []
    for task in tasks:
        print(f"\n>> Task: {task}")
        for m in models:
            results.append(train_one(args.source, task, m))

    print("\nWriting tables and figures...")
    write_tables(results, args.source)
    plot_comparison(results, args.source)
    print("Done.")


if __name__ == "__main__":
    main()