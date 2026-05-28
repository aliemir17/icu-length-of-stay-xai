"""Synthetic MIMIC-like ICU stays dataset for pipeline development.

Generates a small dataframe with the rough shape and column names of a
MIMIC-derived ICU LoS modeling table. Useful for building the full
pipeline (preprocess -> train -> explain) before real MIMIC access lands.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED, SYNTHETIC_DIR, TARGET


def generate(n: int = 2000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.normal(65, 17, n).clip(18, 95)
    gender = rng.choice(["M", "F"], n)
    admission_type = rng.choice(
        ["EMERGENCY", "ELECTIVE", "URGENT"], n, p=[0.7, 0.2, 0.1]
    )
    has_sepsis = rng.binomial(1, 0.18, n)

    # Vitals (first-24h aggregates)
    heart_rate_mean = rng.normal(88, 18, n).clip(40, 180)
    sbp_mean = rng.normal(120, 22, n).clip(60, 200)
    resp_rate_mean = rng.normal(19, 5, n).clip(8, 40)
    temp_c_mean = rng.normal(36.8, 0.9, n).clip(33, 41)
    spo2_mean = rng.normal(96, 3, n).clip(70, 100)

    # Labs (first-24h)
    lactate_max = rng.gamma(2.0, 1.2, n).clip(0.3, 20)
    creatinine_max = rng.gamma(1.5, 1.0, n).clip(0.2, 12)
    wbc_max = rng.gamma(3.0, 3.0, n).clip(0.5, 60)
    hemoglobin_min = rng.normal(11, 2.2, n).clip(4, 18)
    glucose_max = rng.normal(150, 60, n).clip(40, 600)

    # Outcome: ICU LoS in days, right-skewed, driven by severity signals
    severity = (
        0.020 * (lactate_max - 2)
        + 0.015 * (creatinine_max - 1)
        + 0.010 * (heart_rate_mean - 80) / 10
        + 0.020 * (resp_rate_mean - 18) / 5
        + 0.012 * np.maximum(0, 38 - temp_c_mean)
        + 0.018 * (age - 65) / 10
        + 0.45 * has_sepsis
        + 0.20 * (admission_type == "EMERGENCY").astype(float)
    )
    base = rng.gamma(shape=2.0, scale=1.4, size=n)
    los_days = (base * np.exp(severity)).clip(0.2, 60)

    df = pd.DataFrame(
        {
            "subject_id": np.arange(100000, 100000 + n),
            "hadm_id": np.arange(200000, 200000 + n),
            "icustay_id": np.arange(300000, 300000 + n),
            "age": age.round(1),
            "gender": gender,
            "admission_type": admission_type,
            "has_sepsis": has_sepsis,
            "heart_rate_mean": heart_rate_mean.round(1),
            "sbp_mean": sbp_mean.round(1),
            "resp_rate_mean": resp_rate_mean.round(1),
            "temp_c_mean": temp_c_mean.round(2),
            "spo2_mean": spo2_mean.round(1),
            "lactate_max": lactate_max.round(2),
            "creatinine_max": creatinine_max.round(2),
            "wbc_max": wbc_max.round(2),
            "hemoglobin_min": hemoglobin_min.round(2),
            "glucose_max": glucose_max.round(1),
            TARGET: los_days.round(2),
        }
    )

    # Inject realistic missingness in labs
    for col in ["lactate_max", "creatinine_max", "wbc_max", "hemoglobin_min", "glucose_max"]:
        mask = rng.random(n) < 0.08
        df.loc[mask, col] = np.nan

    return df


def main() -> None:
    df = generate()
    out = SYNTHETIC_DIR / "icu_stays_synth.parquet"
    df.to_parquet(out, index=False)
    print(f"Wrote {len(df)} synthetic ICU stays -> {out}")
    print(df.describe(include="all").T.head(20))


if __name__ == "__main__":
    main()
