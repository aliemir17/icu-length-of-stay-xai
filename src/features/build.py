"""Feature engineering: convert raw ICU stays into model-ready X, y.

Feature lists are derived from the input dataframe so the same code works on
synthetic data and on MIMIC-IV without per-source branches.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LAB_ITEMIDS, MISSINGNESS_DROP_THRESHOLD, TARGET, VITAL_ITEMIDS

# Columns to never feed to the model. Includes IDs, timestamps, the target,
# and outcome variables that would leak (hospital mortality, ICU mortality).
EXCLUDE_COLS = {
    "subject_id", "hadm_id", "stay_id", "icustay_id",
    "intime", "outtime",
    "hospital_expire_flag", "died_in_icu",
    "has_sepsis",   # Phase 4 subgroup flag; not a feature (would leak)
    TARGET,
}


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Return X, y, plus the resolved numeric and categorical column lists.

    Fully-NaN columns are dropped: median imputation can't fill a column
    that has zero observed values, the resulting NaNs propagate silently
    through StandardScaler, and linear models explode on them while tree
    models silently tolerate them. Dropping is the consistent fix.

    When `MISSINGNESS_DROP_THRESHOLD` (config) is set, columns whose share
    of missing values *strictly* exceeds the threshold are also dropped.
    """
    feature_df = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    fully_na = feature_df.columns[feature_df.isna().all()].tolist()
    if fully_na:
        feature_df = feature_df.drop(columns=fully_na)
    if MISSINGNESS_DROP_THRESHOLD is not None:
        miss = feature_df.isna().mean()
        high_miss = miss[miss > MISSINGNESS_DROP_THRESHOLD].index.tolist()
        if high_miss:
            feature_df = feature_df.drop(columns=high_miss)

    # --- Feature A: missing-indicator flags ---
    # For numeric columns with >5 % NaN, add a binary "<col>_missing"
    # feature. Trees can split on "was this lab drawn?" directly — the
    # *presence* of a measurement is itself a clinical signal (e.g.
    # lactate drawn → sepsis workup → likely longer stay).
    num_before = feature_df.select_dtypes(include="number")
    miss_rates = num_before.isna().mean()
    for col in miss_rates[miss_rates > 0.05].index:
        feature_df[f"{col}_missing"] = feature_df[col].isna().astype(int)

    # --- Feature B: measurement-intensity counts ---
    # How many vital / lab concepts were measured at all in the first 24 h.
    # A concept counts as "measured" when its _mean column is non-null.
    vital_concepts = list(VITAL_ITEMIDS.keys())
    lab_concepts = list(LAB_ITEMIDS.keys())
    feature_df["n_vitals_measured"] = sum(
        feature_df[f"{c}_mean"].notna().astype(int)
        for c in vital_concepts if f"{c}_mean" in feature_df.columns
    )
    feature_df["n_labs_measured"] = sum(
        feature_df[f"{c}_mean"].notna().astype(int)
        for c in lab_concepts if f"{c}_mean" in feature_df.columns
    )

    numeric_cols = feature_df.select_dtypes(include="number").columns.tolist()
    categorical_cols = feature_df.select_dtypes(exclude="number").columns.tolist()
    X = feature_df[numeric_cols + categorical_cols].copy()
    y = df[TARGET].copy()
    return X, y, numeric_cols, categorical_cols


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ]
    )
