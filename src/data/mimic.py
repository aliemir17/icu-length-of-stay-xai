"""MIMIC-IV ICU LoS feature extraction.

Wired to the credentialed MIMIC-IV 3.1 release at the path in `config.MIMIC_IV_DIR`.
The schema is stable across the demo and credentialed releases, so this module is
data-version agnostic.

Output: one row per cohort ICU stay, with first-24h vital + lab aggregates and
patient/admission metadata, plus the `los_days` target.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    AGG_FUNCS,
    EARLY_WINDOW_HOURS,
    LAB_ITEMIDS,
    MIMIC_IV_HOSP_DIR,
    MIMIC_IV_ICU_DIR,
    MIN_AGE,
    MIN_LOS_DAYS,
    TARGET,
    VITAL_ITEMIDS,
)


# ---------------------------------------------------------------------------
# Cohort
# ---------------------------------------------------------------------------

def load_base_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three small tables that define the cohort."""
    icustays = pd.read_csv(
        MIMIC_IV_ICU_DIR / "icustays.csv.gz",
        parse_dates=["intime", "outtime"],
    )
    patients = pd.read_csv(MIMIC_IV_HOSP_DIR / "patients.csv.gz")
    admissions = pd.read_csv(
        MIMIC_IV_HOSP_DIR / "admissions.csv.gz",
        parse_dates=["admittime", "dischtime", "deathtime"],
    )
    return icustays, patients, admissions


def build_cohort() -> pd.DataFrame:
    """Apply the cohort filters from README.md (first stay, age >= 18, LoS >= 1 day,
    exclude ICU deaths). Returns one row per `stay_id` with metadata."""
    icustays, patients, admissions = load_base_tables()

    # Approximate age at ICU admission: anchor_age + (intime.year - anchor_year)
    pat = patients[["subject_id", "gender", "anchor_age", "anchor_year"]].copy()
    cohort = icustays.merge(pat, on="subject_id", how="left")
    cohort["age"] = cohort["anchor_age"] + (cohort["intime"].dt.year - cohort["anchor_year"])

    # Bring in admission metadata + ICU death derivation
    adm = admissions[[
        "subject_id", "hadm_id", "admission_type", "race",
        "insurance", "hospital_expire_flag", "deathtime",
    ]]
    cohort = cohort.merge(adm, on=["subject_id", "hadm_id"], how="left")

    cohort["died_in_icu"] = (
        cohort["deathtime"].notna()
        & (cohort["deathtime"] >= cohort["intime"])
        & (cohort["deathtime"] <= cohort["outtime"])
    ).astype(int)

    n0 = len(cohort)
    # First ICU stay per patient
    cohort = cohort.sort_values(["subject_id", "intime"]).drop_duplicates("subject_id", keep="first")
    n1 = len(cohort)
    # Adults only
    cohort = cohort[cohort["age"] >= MIN_AGE]
    n2 = len(cohort)
    # Stays >= 1 day
    cohort = cohort[cohort["los"] >= MIN_LOS_DAYS]
    n3 = len(cohort)
    # Exclude ICU deaths
    cohort = cohort[cohort["died_in_icu"] == 0]
    n4 = len(cohort)

    print(
        f"Cohort filter trace:\n"
        f"  raw stays                : {n0:>5}\n"
        f"  + first stay per patient : {n1:>5}  (-{n0 - n1})\n"
        f"  + age >= {MIN_AGE}            : {n2:>5}  (-{n1 - n2})\n"
        f"  + LoS >= {MIN_LOS_DAYS} day        : {n3:>5}  (-{n2 - n3})\n"
        f"  + exclude ICU death      : {n4:>5}  (-{n3 - n4})"
    )

    cohort = cohort.rename(columns={"los": TARGET})

    # Derive sepsis flag from diagnoses_icd (ICD-9/ICD-10 codes).
    # Used for Phase 4 subgroup analysis; excluded from features via EXCLUDE_COLS.
    cohort["has_sepsis"] = derive_sepsis_flag(cohort["hadm_id"]).astype(int)
    n_sepsis = int(cohort["has_sepsis"].sum())
    print(f"  sepsis subgroup          : {n_sepsis:>5,}  ({n_sepsis/len(cohort):.1%} of cohort)")

    keep_cols = [
        "subject_id", "hadm_id", "stay_id", "intime", "outtime",
        "age", "gender", "admission_type", "race", "insurance",
        "hospital_expire_flag", "died_in_icu", "has_sepsis", TARGET,
    ]
    return cohort[keep_cols].reset_index(drop=True)


def derive_sepsis_flag(cohort_hadm_ids: pd.Series) -> pd.Series:
    """Return a boolean Series indexed like cohort_hadm_ids: True if the
    admission has any sepsis-related ICD-9 or ICD-10 code in `diagnoses_icd`.

    Codes (MIMIC stores them without the decimal point):
    - ICD-9:  99591 (Sepsis), 99592 (Severe sepsis)
    - ICD-10: A40* (Streptococcal sepsis), A41* (Other sepsis),
              R6520 (Severe sepsis without shock), R6521 (Severe sepsis with shock)
    """
    icd = pd.read_csv(
        MIMIC_IV_HOSP_DIR / "diagnoses_icd.csv.gz",
        usecols=["hadm_id", "icd_code", "icd_version"],
        dtype={"icd_code": str},
    )
    sepsis_9 = (icd["icd_version"] == 9) & icd["icd_code"].isin({"99591", "99592"})
    sepsis_10 = (icd["icd_version"] == 10) & (
        icd["icd_code"].str.startswith("A40")
        | icd["icd_code"].str.startswith("A41")
        | icd["icd_code"].isin({"R6520", "R6521"})
    )
    sepsis_hadm = set(icd.loc[sepsis_9 | sepsis_10, "hadm_id"].unique())
    return cohort_hadm_ids.isin(sepsis_hadm)


# ---------------------------------------------------------------------------
# Event aggregation
# ---------------------------------------------------------------------------

def _flatten_itemid_map(itemid_map: dict[str, list[int]]) -> dict[int, str]:
    """Reverse the {concept: [itemids]} map to {itemid: concept}."""
    return {iid: concept for concept, ids in itemid_map.items() for iid in ids}


def _slope_for_group(g: pd.DataFrame) -> float:
    """Linear-regression slope of `valuenum` against hours-since-intime.

    Returns NaN if fewer than 3 valid observations, or if all timestamps
    are identical (variance-free regressor is undefined).
    """
    if len(g) < 3 or g["_hours"].std() == 0:
        return float("nan")
    return float(np.polyfit(g["_hours"].values, g["valuenum"].values, 1)[0])


def _aggregate_events(
    events: pd.DataFrame,
    cohort: pd.DataFrame,
    time_col: str,
    join_keys: list[str],
    itemid_map: dict[str, list[int]],
    *,
    window_hours: int = EARLY_WINDOW_HOURS,
) -> pd.DataFrame:
    """Window an event stream to the first N hours of each ICU stay and
    aggregate per (stay_id, concept).

    Produces the simple aggregations in `AGG_FUNCS` (min / max / mean / std)
    plus a per-stay `slope` (linear regression of value against hours-since-intime).
    Returns a wide table indexed by `stay_id` with columns named
    `<concept>_<agg>` (e.g., `heart_rate_mean`, `lactate_slope`).
    """
    itemid_to_concept = _flatten_itemid_map(itemid_map)
    events = events[events["itemid"].isin(itemid_to_concept)].copy()
    events["concept"] = events["itemid"].map(itemid_to_concept)

    # Window join: attach intime via the join keys, then keep only events within window
    cols = list(dict.fromkeys(["stay_id", *join_keys, "intime"]))
    intime = cohort[cols]
    events = events.merge(intime, on=join_keys, how="inner")
    cutoff = events["intime"] + pd.Timedelta(hours=window_hours)
    events = events[(events[time_col] >= events["intime"]) & (events[time_col] < cutoff)]

    events = events.dropna(subset=["valuenum"]).copy()
    events["_hours"] = (events[time_col] - events["intime"]).dt.total_seconds() / 3600.0

    # Simple aggregations (min / max / mean / std)
    simple = (
        events.groupby(["stay_id", "concept"])["valuenum"]
        .agg(AGG_FUNCS)
        .unstack("concept")
    )
    simple.columns = [f"{concept}_{agg}" for agg, concept in simple.columns]

    # Slope: linear regression of valuenum vs hours-since-intime
    slopes = (
        events.groupby(["stay_id", "concept"])[["valuenum", "_hours"]]
        .apply(_slope_for_group)
        .unstack("concept")
    )
    slopes.columns = [f"{c}_slope" for c in slopes.columns]

    return simple.join(slopes, how="outer").reset_index()


def _read_csv_filtered(
    path,
    *,
    usecols: list[str],
    parse_dates: list[str],
    keep_itemids: set[int],
    cohort_filter_col: str,
    cohort_filter_values: set,
    chunksize: int = 2_000_000,
    label: str = "events",
) -> pd.DataFrame:
    """Stream a large MIMIC event CSV in chunks, retaining only rows whose
    itemid is in `keep_itemids` AND whose `cohort_filter_col` value belongs
    to the cohort. Lets us scan multi-GB chartevents / labevents without
    materializing the full table in memory.
    """
    kept_chunks: list[pd.DataFrame] = []
    rows_seen = 0
    rows_kept = 0
    reader = pd.read_csv(
        path, usecols=usecols, parse_dates=parse_dates, chunksize=chunksize,
    )
    for i, chunk in enumerate(reader, start=1):
        rows_seen += len(chunk)
        chunk = chunk[
            chunk["itemid"].isin(keep_itemids)
            & chunk[cohort_filter_col].isin(cohort_filter_values)
        ]
        rows_kept += len(chunk)
        if len(chunk):
            kept_chunks.append(chunk)
        print(f"  [{label}] chunk {i}: scanned {rows_seen:>12,} | kept {rows_kept:>10,}", flush=True)
    return pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=usecols)


def compute_vital_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """First-24h vitals from chartevents (chunked + itemid/stay_id filtered)."""
    ce = _read_csv_filtered(
        MIMIC_IV_ICU_DIR / "chartevents.csv.gz",
        usecols=["stay_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        keep_itemids=set(_flatten_itemid_map(VITAL_ITEMIDS).keys()),
        cohort_filter_col="stay_id",
        cohort_filter_values=set(cohort["stay_id"].tolist()),
        label="chartevents",
    )
    return _aggregate_events(
        ce, cohort, time_col="charttime", join_keys=["stay_id"], itemid_map=VITAL_ITEMIDS,
    )


def compute_lab_features(cohort: pd.DataFrame) -> pd.DataFrame:
    """First-24h labs from labevents. Joined via hadm_id (labevents has no stay_id)."""
    le = _read_csv_filtered(
        MIMIC_IV_HOSP_DIR / "labevents.csv.gz",
        usecols=["subject_id", "hadm_id", "charttime", "itemid", "valuenum"],
        parse_dates=["charttime"],
        keep_itemids=set(_flatten_itemid_map(LAB_ITEMIDS).keys()),
        cohort_filter_col="hadm_id",
        cohort_filter_values=set(cohort["hadm_id"].dropna().tolist()),
        label="labevents",
    )
    return _aggregate_events(
        le, cohort, time_col="charttime",
        join_keys=["subject_id", "hadm_id"], itemid_map=LAB_ITEMIDS,
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

# Fahrenheit -> Celsius conversion rules.
# Value-like aggregates carry the -32 offset; spread-like aggregates do not,
# because constants cancel out under differencing. std_C = std_F * 5/9 and
# slope_C = slope_F * 5/9, NOT (x - 32) * 5/9.
_F2C_VALUE_AGGS = {"min", "max", "mean"}
_F2C_SPREAD_AGGS = {"std", "slope"}


def build_dataset() -> pd.DataFrame:
    """Top-level: cohort + first-24h features + target. One row per ICU stay."""
    cohort = build_cohort()
    vitals = compute_vital_features(cohort)
    labs = compute_lab_features(cohort)

    df = cohort.merge(vitals, on="stay_id", how="left").merge(labs, on="stay_id", how="left")

    # Fold Fahrenheit temp into the Celsius columns where Celsius is missing
    for agg in _F2C_VALUE_AGGS | _F2C_SPREAD_AGGS:
        c_col, f_col = f"temp_c_{agg}", f"temp_f_{agg}"
        if c_col in df.columns and f_col in df.columns:
            if agg in _F2C_VALUE_AGGS:
                converted = (df[f_col] - 32) * 5.0 / 9.0
            else:
                converted = df[f_col] * 5.0 / 9.0
            df[c_col] = df[c_col].fillna(converted)
            df = df.drop(columns=[f_col])

    return df


if __name__ == "__main__":
    from src.config import PROCESSED_DIR

    df = build_dataset()
    cache = PROCESSED_DIR / "mimic4_full_icu_stays.parquet"
    df.to_parquet(cache, index=False)
    print(f"\nFinal dataset: {len(df)} stays x {df.shape[1]} columns")
    print(f"Cache written to: {cache}")
    print(f"Feature missingness summary (mean across feature cols):")
    agg_suffixes = list(AGG_FUNCS) + ["slope"]
    feat_cols = [c for c in df.columns if any(c.endswith(f"_{a}") for a in agg_suffixes)]
    print(f"  {df[feat_cols].isna().mean().mean():.1%}")
    print(f"Target stats:\n{df[TARGET].describe().to_string()}")