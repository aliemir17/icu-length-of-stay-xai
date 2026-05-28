"""Optuna hyperparameter tuning for tree models.

Runs Bayesian optimization over the search space for each (model, task)
pair. Best params are printed as a JSON dict at the end of each study so
they can be pasted into `make_model` in `train.py`.

Usage:
    python -m src.models.tune --source mimic4 --task regression --model xgb --n-trials 50
    python -m src.models.tune --source mimic4 --task all --model all --n-trials 50
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import optuna
from sklearn.metrics import average_precision_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.config import RANDOM_SEED
from src.data.load import load_icu_stays
from src.features.build import build_preprocessor, split_xy
from src.models.train import LONG_STAY_DAYS, _LOG1P_PRED_CAP, prepare_target

TUNE_TASKS = ("regression", "classification")
TUNE_MODELS = ("xgb", "lgbm", "rf")


def _suggest_xgb(trial: optuna.Trial, is_clf: bool) -> dict:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 50),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    if is_clf:
        from src.models.train import _SCALE_POS_WEIGHT
        params["scale_pos_weight"] = _SCALE_POS_WEIGHT
        params["eval_metric"] = "logloss"
    return params


def _suggest_lgbm(trial: optuna.Trial, is_clf: bool) -> dict:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 1000, step=50),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        num_leaves=trial.suggest_int("num_leaves", 15, 127),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1,
    )
    if is_clf:
        params["class_weight"] = "balanced"
    return params


def _suggest_rf(trial: optuna.Trial, is_clf: bool) -> dict:
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
        max_depth=trial.suggest_int("max_depth", 5, 20),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 20),
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    if is_clf:
        params["class_weight"] = "balanced"
    return params


_SUGGESTERS = {"xgb": _suggest_xgb, "lgbm": _suggest_lgbm, "rf": _suggest_rf}


def _make_estimator(model_name: str, task: str, params: dict):
    is_clf = task == "classification"
    if model_name == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier(**params) if is_clf else XGBRegressor(**params)
    if model_name == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMClassifier(**params) if is_clf else LGBMRegressor(**params)
    if model_name == "rf":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return RandomForestClassifier(**params) if is_clf else RandomForestRegressor(**params)
    raise ValueError(model_name)


def _cv_score(pipe_factory, X, y, task: str, n_splits: int = 5) -> float:
    if task == "classification":
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_y = y
    else:
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        split_y = None

    scores = []
    for tr, va in splitter.split(X, split_y):
        pipe = pipe_factory()
        pipe.fit(X.iloc[tr], y.iloc[tr])
        if task == "classification":
            y_proba = pipe.predict_proba(X.iloc[va])[:, 1]
            scores.append(average_precision_score(y.iloc[va], y_proba))
        else:
            y_pred = pipe.predict(X.iloc[va])
            y_pred_days = np.expm1(np.clip(y_pred, a_min=None, a_max=_LOG1P_PRED_CAP))
            y_true_days = np.expm1(y.iloc[va].values)
            scores.append(mean_absolute_error(y_true_days, y_pred_days))
    return float(np.mean(scores))


def tune_one(
    source: str,
    task: str,
    model_name: str,
    n_trials: int = 50,
) -> dict:
    df = load_icu_stays(source)
    X, y_raw, numeric_cols, categorical_cols = split_xy(df)
    y = prepare_target(y_raw, task)

    stratify = y if task == "classification" else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=stratify,
    )

    suggest = _SUGGESTERS[model_name]
    is_clf = task == "classification"
    direction = "maximize" if is_clf else "minimize"

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, is_clf)
        factory = lambda: Pipeline([
            ("preprocess", build_preprocessor(numeric_cols, categorical_cols)),
            ("model", _make_estimator(model_name, task, params)),
        ])
        return _cv_score(factory, X_tr, y_tr, task)

    study = optuna.create_study(
        direction=direction,
        study_name=f"{model_name}_{task}_{source}",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial
    metric_name = "AUC-PR" if is_clf else "MAE"
    print(f"\n{'='*60}")
    print(f"  {model_name} / {task}  —  best CV {metric_name}: {best.value:.4f}")
    print(f"  trial #{best.number}, params:")
    print(f"  {json.dumps(best.params, indent=2)}")
    print(f"{'='*60}\n")

    return {"model": model_name, "task": task, "best_cv": best.value, "params": best.params}


def main() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="mimic4", choices=["synthetic", "mimic4"])
    parser.add_argument("--task", default="all", choices=[*TUNE_TASKS, "all"])
    parser.add_argument("--model", default="all", choices=[*TUNE_MODELS, "all"])
    parser.add_argument("--n-trials", type=int, default=50)
    args = parser.parse_args()

    tasks = list(TUNE_TASKS) if args.task == "all" else [args.task]
    models = list(TUNE_MODELS) if args.model == "all" else [args.model]

    print(f"Optuna tuning: source={args.source}, tasks={tasks}, models={models}, "
          f"n_trials={args.n_trials}")

    all_results = []
    for task in tasks:
        for m in models:
            res = tune_one(args.source, task, m, args.n_trials)
            all_results.append(res)

    print("\n" + "=" * 60)
    print("SUMMARY — best params per (model, task):")
    print("=" * 60)
    for r in all_results:
        metric = "AUC-PR" if r["task"] == "classification" else "MAE"
        print(f"\n  {r['model']} / {r['task']}  best CV {metric}: {r['best_cv']:.4f}")
        for k, v in r["params"].items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()