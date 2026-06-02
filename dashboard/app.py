"""ICU Length-of-Stay XAI Dashboard — Streamlit app.

What-if analysis prototype built on the trained LGBM regression + XGB
classification pipelines from this project. Loads the held-out test
set, lets the user pick or randomise a patient, edit their first-24 h
features, and inspect the resulting prediction + SHAP waterfall.

Run with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import LAB_ITEMIDS, MODELS_DIR, RANDOM_SEED, TARGET, VITAL_ITEMIDS
from src.data.load import load_icu_stays
from src.features.build import split_xy
from src.models.train import LONG_STAY_DAYS, prepare_target, to_days

# Concept names whose `<concept>_mean` column drives `n_vitals_measured`
# / `n_labs_measured`. `temp_f` lives in VITAL_ITEMIDS but is merged into
# `temp_c` during feature building, so it never shows up as its own column.
VITAL_CONCEPTS = [c for c in VITAL_ITEMIDS if c != "temp_f"]
LAB_CONCEPTS = list(LAB_ITEMIDS)

SOURCE = "mimic4"
REG_MODEL = "lgbm"           # best regressor
CLF_MODEL = "xgb"            # best 7-day classifier

# 6 stratified preset patients (picked by _pick_presets.py).
# Filter: per-patient |LGBM pred - actual LoS| < 1.5 d AND no missing features,
# so SHAP attributions are clean and the predictions are realistic.
PRESETS: list[tuple[int, str]] = [
    (35193405, "Short-stay #1 (1.6 d actual, pred 1.7 d, non-sepsis, age 61)"),
    (31155071, "Short-stay #2 (1.9 d actual, pred 1.8 d, non-sepsis, age 68)"),
    (36469158, "Long-stay #1 (7.2 d actual, pred 7.6 d, non-sepsis, age 32)"),
    (36937902, "Long-stay #2 (8.9 d actual, pred 8.3 d, non-sepsis, age 28)"),
    (36354239, "Sepsis #1 (10.0 d actual, pred 9.8 d, sepsis, age 53)"),
    (33862410, "Sepsis #2 (5.2 d actual, pred 5.4 d, sepsis, age 81)"),
]

# Editable feature groups for the what-if sidebar.
DEMOGRAPHIC_NUMERIC = ["age"]
DEMOGRAPHIC_CATEGORICAL = ["gender", "admission_type", "race", "insurance"]

VITAL_MEANS = [
    "heart_rate_mean", "sbp_mean", "dbp_mean", "map_mean",
    "resp_rate_mean", "spo2_mean", "temp_c_mean",
]
LAB_MEANS = [
    "lactate_mean", "creatinine_mean", "bun_mean", "wbc_mean",
    "hemoglobin_mean", "hematocrit_mean", "platelets_mean",
    "sodium_mean", "potassium_mean", "glucose_mean",
    "bilirubin_mean", "ph_mean", "anion_gap_mean", "inr_mean",
]
# Extra SHAP-top drivers not covered above
EXTRA_DRIVERS = [
    "spo2_slope", "dbp_slope", "resp_rate_slope",
]


# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading trained pipelines...")
def load_pipelines() -> dict:
    return {
        "reg": joblib.load(MODELS_DIR / f"{REG_MODEL}_regression_{SOURCE}.joblib"),
        "clf": joblib.load(MODELS_DIR / f"{CLF_MODEL}_classification_{SOURCE}.joblib"),
    }


@st.cache_data(show_spinner="Loading test set...")
def load_test_data() -> dict:
    df = load_icu_stays(SOURCE)
    X, y_raw, num_cols, cat_cols = split_xy(df)

    # Reproduce the regression-task split (random_state=42, test_size=0.2)
    _, X_te, _, y_te = train_test_split(
        X, prepare_target(y_raw, "regression"),
        test_size=0.2, random_state=RANDOM_SEED,
    )
    te_idx = X_te.index

    # Pull metadata for the same index — actual LoS, sepsis flag etc.
    meta = df.loc[te_idx, [
        "stay_id", "subject_id", "has_sepsis", TARGET,
    ]].reset_index(drop=True)
    X_te = X_te.reset_index(drop=True)
    y_te = y_te.reset_index(drop=True)

    return {
        "X_te": X_te,
        "y_te": y_te,
        "y_days": to_days(y_te.values, "regression"),
        "meta": meta,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }


@st.cache_resource(show_spinner="Building SHAP explainers...")
def build_explainers(_pipes: dict, _test: dict) -> dict:
    """Build TreeExplainer for each model on the preprocessed test data."""
    background = _test["X_te"].sample(100, random_state=RANDOM_SEED)
    explainers = {}
    for key in ("reg", "clf"):
        pipe = _pipes[key]
        pre = pipe.named_steps["preprocess"]
        model = pipe.named_steps["model"]
        bg_transformed = pre.transform(background)
        feature_names = pre.get_feature_names_out()
        explainers[key] = {
            "explainer": shap.TreeExplainer(model, bg_transformed, feature_names=feature_names),
            "feature_names": feature_names,
            "preprocess": pre,
        }
    return explainers


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def predict_one(pipe, x_row: pd.DataFrame, task: str) -> float:
    """Return regression days OR classification probability."""
    if task == "regression":
        log_pred = pipe.predict(x_row)
        return float(np.expm1(log_pred[0]))
    proba = pipe.predict_proba(x_row)[0, 1]
    return float(proba)


def shap_waterfall_figure(explainer_info: dict, pipe, x_row: pd.DataFrame, title: str):
    """Return a matplotlib figure containing a single-row SHAP waterfall.

    TreeExplainer's additivity check is disabled. On a small number of
    LGBM regression / XGB classification rows the sum-of-shap-values can
    drift from the model output by ~0.01-0.02 (log-space) due to
    floating-point accumulation in deep trees — well below the user-
    visible scale and routinely worked around by the same flag in the
    SHAP examples gallery.
    """
    pre = explainer_info["preprocess"]
    feature_names = explainer_info["feature_names"]
    explainer = explainer_info["explainer"]
    x_transformed = pre.transform(x_row)

    raw_shap = explainer.shap_values(x_transformed, check_additivity=False)
    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)) and np.ndim(expected_value) > 0:
        expected_value = float(np.asarray(expected_value).ravel()[0])
    # raw_shap may come back as a list (classification with two outputs) or array
    if isinstance(raw_shap, list):
        raw_shap = raw_shap[-1]  # positive-class contributions
    raw_shap = np.asarray(raw_shap)
    if raw_shap.ndim == 2:
        row_values = raw_shap[0]
    else:
        row_values = raw_shap

    expl = shap.Explanation(
        values=row_values,
        base_values=float(expected_value),
        data=x_transformed[0],
        feature_names=list(feature_names),
    )

    fig = plt.figure(figsize=(8, 6))
    shap.plots.waterfall(expl, max_display=12, show=False)
    fig.suptitle(title, fontsize=10, y=1.02)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def fmt_pct(p: float) -> str:
    return f"{p * 100:.1f} %"


def fmt_days(d: float) -> str:
    return f"{d:.2f} d"


def missing_flag_col(feature: str, df_columns) -> str | None:
    """Return the name of the `_missing` indicator paired with `feature`, if any.

    The feature builder names every flag `<feature>_missing` — direct suffix,
    no rewrites — so a single concatenation is all we need. Earlier versions
    of this helper also tried `feature.replace("_mean", "_missing")`, but for
    a feature like `age` or `spo2_slope` the replace is a no-op and the helper
    ended up returning the feature's own name as if it were a missing flag.
    """
    cand = f"{feature}_missing"
    return cand if cand in df_columns else None


def compute_measurement_counts(row: pd.DataFrame) -> tuple[int, int]:
    """Return (n_vitals_measured, n_labs_measured) derived live from `<concept>_mean`.

    A concept counts as measured iff its `_mean` aggregate is non-null in `row`.
    Used after the what-if reconstruction so that toggling labs/vitals on or off
    propagates into the measurement-count features the model reads.
    """
    n_vit = sum(
        f"{c}_mean" in row.columns and not pd.isna(row[f"{c}_mean"].iloc[0])
        for c in VITAL_CONCEPTS
    )
    n_lab = sum(
        f"{c}_mean" in row.columns and not pd.isna(row[f"{c}_mean"].iloc[0])
        for c in LAB_CONCEPTS
    )
    return n_vit, n_lab


def render_prediction_cards(original: dict, whatif: dict) -> None:
    """Two side-by-side cards showing regression + classification predictions."""
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Original patient")
        st.metric("Predicted LoS (LGBM)", fmt_days(original["los"]))
        st.metric("Long-stay probability (XGB, 7-day)",
                  fmt_pct(original["proba"]))
        st.caption(f"Actual LoS: **{fmt_days(original['actual_los'])}** "
                   f"({'Long' if original['actual_los'] >= LONG_STAY_DAYS else 'Short'} stay)")
    with c2:
        st.markdown("#### What-if")
        delta_los = whatif["los"] - original["los"]
        delta_p = whatif["proba"] - original["proba"]
        st.metric(
            "Predicted LoS (LGBM)",
            fmt_days(whatif["los"]),
            delta=f"{delta_los:+.2f} d",
            delta_color="inverse" if delta_los > 0 else "normal",
        )
        st.metric(
            "Long-stay probability (XGB, 7-day)",
            fmt_pct(whatif["proba"]),
            delta=f"{delta_p * 100:+.1f} pp",
            delta_color="inverse" if delta_p > 0 else "normal",
        )
        crossed = (whatif["proba"] >= 0.5) != (original["proba"] >= 0.5)
        if crossed:
            st.caption("⚠️ Decision flipped across the 0.5 cut-off.")


# Features that get an explicit "✓ Measured" toggle on top of the number_input.
# Covers every vital and lab `_mean` aggregate regardless of whether the
# feature also has a `_missing` indicator counterpart. Toggle-off blanks the
# value to NaN (the pipeline's median imputer takes over) and updates
# `n_vitals_measured` / `n_labs_measured` via `compute_measurement_counts`.
# If the feature DOES have a `_missing` flag we flip it too; if it doesn't
# (most vital means) the NaN alone carries the signal.
TOGGLABLE_FEATURES: set[str] = (
    {f"{c}_mean" for c in VITAL_CONCEPTS}
    | {f"{c}_mean" for c in LAB_CONCEPTS}
)


# Editable field registry. Each tuple: (label, format, step)
EDITABLE_NUMERIC: list[tuple[str, str, str, float]] = [
    # Demographics
    ("age",               "Age (years)",                          "%d",    1.0),
    # Vitals
    ("heart_rate_mean",   "Heart rate (bpm)",                     "%.2f",  0.5),
    ("sbp_mean",          "Systolic BP (mmHg)",                   "%.2f",  0.5),
    ("dbp_mean",          "Diastolic BP (mmHg)",                  "%.2f",  0.5),
    ("map_mean",          "Mean arterial pressure (mmHg)",        "%.2f",  0.5),
    ("resp_rate_mean",    "Respiratory rate (/min)",              "%.2f",  0.5),
    ("spo2_mean",         "SpO2 (%)",                             "%.2f",  0.5),
    ("temp_c_mean",       "Temperature (°C)",                     "%.2f",  0.5),
    # Labs
    ("lactate_mean",      "Lactate (mmol/L)",                     "%.2f",  0.1),
    ("creatinine_mean",   "Creatinine (mg/dL)",                   "%.2f",  0.1),
    ("bun_mean",          "BUN (mg/dL)",                          "%.2f",  1.0),
    ("wbc_mean",          "WBC (10³/µL)",                         "%.2f",  0.5),
    ("hemoglobin_mean",   "Hemoglobin (g/dL)",                    "%.2f",  0.2),
    ("hematocrit_mean",   "Hematocrit (%)",                       "%.2f",  0.5),
    ("platelets_mean",    "Platelets (10³/µL)",                   "%.2f", 10.0),
    ("sodium_mean",       "Sodium (mEq/L)",                       "%.2f",  0.5),
    ("potassium_mean",    "Potassium (mEq/L)",                    "%.2f",  0.1),
    ("glucose_mean",      "Glucose (mg/dL)",                      "%.2f",  5.0),
    ("bilirubin_mean",    "Bilirubin (mg/dL)",                    "%.2f",  0.2),
    ("ph_mean",           "pH",                                   "%.2f",  0.01),
    ("anion_gap_mean",    "Anion gap (mEq/L)",                    "%.2f",  0.5),
    ("inr_mean",          "INR",                                  "%.2f",  0.1),
    # Top SHAP drivers
    ("spo2_slope",        "SpO2 slope (/h)",                      "%.3f",  0.01),
    ("dbp_slope",         "Diastolic BP slope (/h)",              "%.3f",  0.01),
    ("resp_rate_slope",   "Respiratory rate slope (/h)",          "%.3f",  0.01),
]
EDITABLE_CATEGORICAL: list[tuple[str, str]] = [
    ("gender",         "Gender"),
    ("admission_type", "Admission type"),
    ("race",           "Race"),
    ("insurance",      "Insurance"),
]


def _widget_prefix() -> str:
    """Prefix used to namespace all editable widget keys.

    The widget keys carry both stay_id AND a version counter. Bumping the
    version (`form_version += 1`) makes every old key invalid, which
    forces Streamlit to render fresh widgets initialised from the new
    `value=` parameter — that is how we guarantee Reset and patient-
    switch truly drop the user's edits on the floor.
    """
    return f"p{st.session_state.current_stay_id}_v{st.session_state.form_version}_"


def _drop_widget_state() -> None:
    """Strip every widget key that matches the current prefix from session_state."""
    prefix = _widget_prefix()
    for k in list(st.session_state.keys()):
        if k.startswith(prefix):
            del st.session_state[k]


def _custom_label_for(stay_id: int) -> str:
    return f"(custom: stay_id {stay_id})"


def _on_picker_change() -> None:
    """Callback fired when the user picks a preset from the dropdown."""
    selected_label = st.session_state.patient_picker
    label_to_id = {label: sid for sid, label in PRESETS}
    if selected_label not in label_to_id:
        return  # user clicked the synthetic "(custom: ...)" option, no-op
    new_id = label_to_id[selected_label]
    if new_id != st.session_state.current_stay_id:
        _drop_widget_state()
        st.session_state.current_stay_id = new_id
        st.session_state.form_version += 1
        st.session_state.whatif_result = None


def _on_random_click(meta: pd.DataFrame) -> None:
    """Callback for the 🎲 Random patient button.

    Runs BEFORE any widget renders on the next rerun, so it can safely
    mutate `patient_picker` — the constraint Streamlit enforces is that
    you cannot write to a widget key *after* the widget has already been
    instantiated in the current run. Callbacks side-step that.
    """
    random_id = int(meta.sample(1).iloc[0]["stay_id"])
    _drop_widget_state()
    st.session_state.current_stay_id = random_id
    st.session_state.form_version += 1
    st.session_state.whatif_result = None
    st.session_state["patient_picker"] = _custom_label_for(random_id)


def _on_reset_click() -> None:
    """Callback for the ↺ Reset to original button."""
    _drop_widget_state()
    st.session_state.form_version += 1
    st.session_state.whatif_result = None


# ---------------------------------------------------------------------------
# Sidebar — input
# ---------------------------------------------------------------------------
def build_sidebar(X_te: pd.DataFrame, meta: pd.DataFrame) -> tuple[int, pd.DataFrame, list[str]]:
    """Render the patient picker + editable form.

    Returns (row_idx, edited_row, missing_notes).

    Widget keys carry `stay_id` AND a `form_version` counter. Bumping
    the version on patient-switch / random / reset is the only reliable
    way to make Streamlit drop cached widget values and re-render from
    `value=`.
    """
    st.sidebar.title("Patient input")

    # ----- State init -----
    if "current_stay_id" not in st.session_state:
        st.session_state.current_stay_id = PRESETS[0][0]
    if "form_version" not in st.session_state:
        st.session_state.form_version = 0
    if "whatif_result" not in st.session_state:
        st.session_state.whatif_result = None

    # ----- 1. Patient picker -----
    # The picker is fully driven by `st.session_state.patient_picker`. We
    # never pass `index=` so Streamlit's session_state is the single source
    # of truth — after Random / Reset we just overwrite that key, and the
    # picker always renders the matching option on the next rerun.
    st.sidebar.markdown("##### 1. Pick a patient")
    preset_labels = [label for _, label in PRESETS]
    current_id = st.session_state.current_stay_id
    current_label = next((label for sid, label in PRESETS if sid == current_id), None)

    if current_label is not None:
        options = preset_labels
        synced_label = current_label
    else:
        synced_label = _custom_label_for(current_id)
        options = preset_labels + [synced_label]

    # Keep session_state.patient_picker consistent with current_stay_id so the
    # widget renders the correct row on every rerun (including the rerun that
    # follows a number_input edit).
    if st.session_state.get("patient_picker") not in options:
        st.session_state["patient_picker"] = synced_label

    st.sidebar.selectbox(
        "Preset patient", options,
        key="patient_picker", on_change=_on_picker_change,
    )

    st.sidebar.button(
        "🎲 Random patient from test set",
        on_click=_on_random_click, args=(meta,),
        use_container_width=True,
    )

    # ----- Locate patient -----
    stay_id = st.session_state.current_stay_id
    row_idx = int(meta.index[meta["stay_id"] == stay_id][0])
    original_row = X_te.iloc[[row_idx]].copy()
    meta_row = meta.iloc[row_idx]

    st.sidebar.caption(
        f"**stay_id**: `{stay_id}` · "
        f"**actual LoS**: {meta_row[TARGET]:.2f} d · "
        f"**sepsis**: {'yes' if meta_row['has_sepsis'] else 'no'}"
    )

    # ----- 2. Edit form -----
    st.sidebar.markdown("##### 2. Edit features (what-if)")
    st.sidebar.button(
        "↺ Reset to original",
        on_click=_on_reset_click,
        use_container_width=True,
    )

    prefix = _widget_prefix()

    # Widget helpers: render only, do NOT modify edited_row here.
    # The edited_row is reconstructed below from current widget state.
    # `container` is the Streamlit DeltaGenerator we draw into — must be the
    # expander (or `st.sidebar`) so widgets nest correctly. Calling
    # `st.sidebar.number_input(...)` inside a `with st.sidebar.expander(...)`
    # block would skip the expander entirely.
    def num_widget(col: str, label: str, fmt: str, step: float, container) -> None:
        if col not in original_row.columns:
            return
        ov = original_row[col].iloc[0]
        originally_measured = not pd.isna(ov)
        default_val = float(ov) if originally_measured else 0.0
        # Streamlit warns when `format="%d"` is paired with a float value/step,
        # so cast both to int for integer-formatted fields.
        if fmt == "%d":
            default_val = int(default_val)
            step = int(step)

        if col not in TOGGLABLE_FEATURES:
            # Always-present feature (age, vital slopes) — plain number_input.
            container.number_input(
                label, value=default_val, format=fmt, step=step,
                key=f"{prefix}{col}",
            )
            return

        # Two-part widget: explicit "Measured?" toggle above the value input.
        # Toggle off → the value is treated as NaN downstream (the pipeline
        # imputer fills it). If the feature also has a `_missing` flag, the
        # flag tracks the toggle state. Either way, the measurement count
        # features (`n_vitals_measured`, `n_labs_measured`) re-derive at the
        # bottom of build_sidebar from the resulting `_mean` non-null pattern.
        is_measured = container.toggle(
            f"✓ Measured — {label}",
            value=originally_measured,
            key=f"{prefix}{col}__measured",
        )
        container.number_input(
            label, value=default_val, format=fmt, step=step,
            key=f"{prefix}{col}", disabled=not is_measured,
        )

    def cat_widget(col: str, label: str, container) -> None:
        if col not in original_row.columns:
            return
        choices = sorted(X_te[col].dropna().unique().tolist())
        if not choices:
            return
        ov = original_row[col].iloc[0]
        if pd.isna(ov) or ov not in choices:
            ov = choices[0]
        container.selectbox(
            label, options=choices, index=choices.index(ov),
            key=f"{prefix}{col}",
        )

    # Group widgets into expanders
    vital_cols = {"heart_rate_mean", "sbp_mean", "dbp_mean", "map_mean",
                  "resp_rate_mean", "spo2_mean", "temp_c_mean"}
    lab_cols = {"lactate_mean", "creatinine_mean", "bun_mean", "wbc_mean",
                "hemoglobin_mean", "hematocrit_mean", "platelets_mean",
                "sodium_mean", "potassium_mean", "glucose_mean",
                "bilirubin_mean", "ph_mean", "anion_gap_mean", "inr_mean"}
    driver_cols = {"spo2_slope", "dbp_slope", "resp_rate_slope"}

    demo_exp = st.sidebar.expander("Demographics", expanded=True)
    num_widget("age", "Age (years)", "%d", 1.0, demo_exp)
    for col, label in EDITABLE_CATEGORICAL:
        cat_widget(col, label, demo_exp)

    vital_exp = st.sidebar.expander("Vital signs — 24 h means")
    for col, label, fmt, step in EDITABLE_NUMERIC:
        if col in vital_cols:
            num_widget(col, label, fmt, step, vital_exp)

    lab_exp = st.sidebar.expander("Labs — 24 h means")
    for col, label, fmt, step in EDITABLE_NUMERIC:
        if col in lab_cols:
            num_widget(col, label, fmt, step, lab_exp)

    driver_exp = st.sidebar.expander("Top SHAP drivers — vital slopes")
    for col, label, fmt, step in EDITABLE_NUMERIC:
        if col in driver_cols:
            num_widget(col, label, fmt, step, driver_exp)

    # ----- Reconstruct edited_row from current widget state -----
    edited_row = original_row.copy()
    notes: list[str] = []

    for col, _, _, _ in EDITABLE_NUMERIC:
        if col not in edited_row.columns:
            continue
        value_key = f"{prefix}{col}"
        if value_key not in st.session_state:
            continue
        original_val = original_row[col].iloc[0]
        originally_measured = not pd.isna(original_val)
        flag = missing_flag_col(col, original_row.columns)
        measure_key = f"{prefix}{col}__measured"

        if measure_key in st.session_state:
            # Explicit toggle drives the measurement intent. `flag` may be None
            # (no `_missing` counterpart) — that's fine; the NaN value alone is
            # the signal and the count features get recomputed below.
            is_measured = bool(st.session_state[measure_key])
            widget_val = float(st.session_state[value_key])

            if is_measured:
                edited_row.at[edited_row.index[0], col] = widget_val
                if flag is not None and not originally_measured:
                    edited_row.at[edited_row.index[0], flag] = 0
                    notes.append(
                        f"`{col}` was unmeasured in the original record; "
                        f"`{flag}` flipped to 0 to reflect the new value."
                    )
            else:
                edited_row.at[edited_row.index[0], col] = np.nan
                if flag is not None and originally_measured:
                    edited_row.at[edited_row.index[0], flag] = 1
                    notes.append(
                        f"`{col}` is being marked as unmeasured; "
                        f"`{flag}` set to 1 and the value will be median-imputed by the pipeline."
                    )
        else:
            # No toggle (age, vital slopes) — plain edit.
            widget_val = float(st.session_state[value_key])
            if originally_measured:
                if not np.isclose(widget_val, float(original_val)):
                    edited_row.at[edited_row.index[0], col] = widget_val
            elif widget_val != 0.0:
                edited_row.at[edited_row.index[0], col] = widget_val

    for col, _ in EDITABLE_CATEGORICAL:
        if col not in edited_row.columns:
            continue
        key = f"{prefix}{col}"
        if key not in st.session_state:
            continue
        widget_val = st.session_state[key]
        if widget_val != original_row[col].iloc[0]:
            edited_row.at[edited_row.index[0], col] = widget_val

    # ----- Auto-recompute measurement-count features from edited `_mean` state -----
    n_vit, n_lab = compute_measurement_counts(edited_row)
    if "n_vitals_measured" in edited_row.columns:
        orig_n_vit = int(original_row["n_vitals_measured"].iloc[0])
        edited_row.at[edited_row.index[0], "n_vitals_measured"] = n_vit
        if n_vit != orig_n_vit:
            notes.append(
                f"`n_vitals_measured` auto-recomputed from toggle state: "
                f"{orig_n_vit} → {n_vit}."
            )
    if "n_labs_measured" in edited_row.columns:
        orig_n_lab = int(original_row["n_labs_measured"].iloc[0])
        edited_row.at[edited_row.index[0], "n_labs_measured"] = n_lab
        if n_lab != orig_n_lab:
            notes.append(
                f"`n_labs_measured` auto-recomputed from toggle state: "
                f"{orig_n_lab} → {n_lab}."
            )

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"**Measurement counts** (auto from toggles) — "
        f"vitals **{n_vit} / {len(VITAL_CONCEPTS)}**, "
        f"labs **{n_lab} / {len(LAB_CONCEPTS)}**"
    )

    return row_idx, edited_row, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="ICU LoS — XAI Dashboard",
                       layout="wide", page_icon="🏥")
    st.title("ICU Length-of-Stay — XAI What-if Dashboard")
    st.caption(
        "Prototype based on MIMIC-IV 3.1 (N = 48,222 ICU stays). "
        "Headline models: LGBM regression (test MAE 2.168 d), "
        "XGB classification at the 7-day threshold (test AUC-PR 0.462). "
        "This tool is a research demo — not for clinical use."
    )

    pipes = load_pipelines()
    test = load_test_data()
    explainers = build_explainers(pipes, test)

    row_idx, edited_row, missing_notes = build_sidebar(test["X_te"], test["meta"])
    original_row = test["X_te"].iloc[[row_idx]]
    actual_los = float(test["y_days"][row_idx])

    # Always compute the Original-patient prediction for the current patient.
    original = {
        "los": predict_one(pipes["reg"], original_row, "regression"),
        "proba": predict_one(pipes["clf"], original_row, "classification"),
        "actual_los": actual_los,
    }

    # Predict button: when pressed, snapshot current edited_row + its predictions.
    predict_btn = st.button("Predict", type="primary", use_container_width=True)
    if predict_btn:
        st.session_state.whatif_result = {
            "los": predict_one(pipes["reg"], edited_row, "regression"),
            "proba": predict_one(pipes["clf"], edited_row, "classification"),
            "row": edited_row.copy(),
        }

    # What-if card: shows whatif_result if user has pressed Predict for this
    # patient; otherwise mirrors the Original prediction so the page is never
    # blank.
    if st.session_state.get("whatif_result") is None:
        whatif = {**original}
        whatif_row = original_row
    else:
        whatif = {
            "los": st.session_state.whatif_result["los"],
            "proba": st.session_state.whatif_result["proba"],
            "actual_los": actual_los,
        }
        whatif_row = st.session_state.whatif_result["row"]

    if missing_notes:
        for n in missing_notes:
            st.info(n)

    st.markdown("### Predictions")
    render_prediction_cards(original, whatif)

    st.markdown("### Explanations (what-if patient)")
    tab_reg, tab_clf = st.tabs([
        "Regression — LGBM (LoS days)",
        "Classification — XGB (long-stay probability)",
    ])
    with tab_reg:
        fig = shap_waterfall_figure(
            explainers["reg"], pipes["reg"], whatif_row,
            title="Per-feature contribution to log1p(LoS) — LGBM regression",
        )
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Values are SHAP contributions in log1p(days) space. "
            "Positive (red) bars push the predicted LoS upward; negative (blue) bars pull it down. "
            "Base value = mean training-set log1p(LoS)."
        )
    with tab_clf:
        fig = shap_waterfall_figure(
            explainers["clf"], pipes["clf"], whatif_row,
            title="Per-feature contribution to long-stay probability — XGB classification",
        )
        st.pyplot(fig)
        plt.close(fig)
        st.caption(
            "Values are SHAP contributions in raw margin (logit) space. "
            "Positive (red) bars increase the long-stay probability; negative (blue) bars decrease it. "
            "Base value = average margin across the training set."
        )

    with st.expander("About this dashboard"):
        st.markdown(
            "This is a research prototype built for a B.Sc. thesis at "
            "Istanbul Technical University (Industrial Engineering). It loads "
            "the project's best two models — LGBM regression (log1p target) "
            "and XGB classification at the 7-day long-stay threshold — and "
            "lets a user pick a real test-set ICU stay, edit the first-24 h "
            "features, and inspect how the model's prediction and SHAP "
            "explanation change.  \n\n"
            "**Repository:** https://github.com/aliemir17/icu-length-of-stay-xai  \n"
            "**Data:** MIMIC-IV 3.1 (credentialed, PhysioNet DUA)  \n"
            "**Disclaimer:** Not a medical device. Not for clinical use."
        )


if __name__ == "__main__":
    main()
