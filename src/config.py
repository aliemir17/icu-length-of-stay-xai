"""Central configuration: paths, constants, column names."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

# MIMIC-IV credentialed full release. Layout identical to the demo (hosp/ + icu/).
MIMIC_IV_DIR = RAW_DIR / "mimic-iv-3.1"
MIMIC_IV_HOSP_DIR = MIMIC_IV_DIR / "hosp"
MIMIC_IV_ICU_DIR = MIMIC_IV_DIR / "icu"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, SYNTHETIC_DIR, FIGURES_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

# ICU LoS target column
TARGET = "los_days"

# Time window for early prediction (first N hours after ICU admission)
EARLY_WINDOW_HOURS = 24

# Cohort filters from the README Decisions log
MIN_AGE = 18
MIN_LOS_DAYS = 1.0

# MIMIC-IV itemid maps. Group label -> list of itemids that represent the same concept.
# Values get coalesced (use whatever is available, prefer the first non-null).
VITAL_ITEMIDS: dict[str, list[int]] = {
    "heart_rate": [220045],
    "sbp": [220179, 220050],          # NIBP first, fall back to arterial
    "dbp": [220180, 220051],
    "map": [220181, 220052],
    "resp_rate": [220210, 224690],
    "spo2": [220277],
    "temp_c": [223762],               # Celsius native
    "temp_f": [223761],               # Fahrenheit, converted in build_features
}

LAB_ITEMIDS: dict[str, list[int]] = {
    "lactate": [50813],
    "creatinine": [50912],
    "bun": [51006],
    "wbc": [51301],
    "hemoglobin": [51222],
    "hematocrit": [51221],
    "platelets": [51265],
    "sodium": [50983],
    "potassium": [50971],
    "glucose": [50931],
    "bilirubin": [50885],
    "ph": [50820],
    "anion_gap": [50868],
    "inr": [51237],
}

# Aggregations applied within the first 24h window per (stay_id, concept)
AGG_FUNCS: list[str] = ["min", "max", "mean", "std"]

# Drop feature columns whose share of missing values strictly exceeds this
# threshold. None = disabled (keep all columns regardless of missingness).
# Strict `>` comparison: a column with exactly the threshold value stays.
# Used by `src/features/build.py::split_xy`.
#
# Disabled after the 2026-05-27 ablations: thresholds 0.40 and 0.60 were
# both tested. All 12 (model x task) combos got worse with any drop; the
# damage scaled with how many cols were dropped. Even the >60%-missing
# slope features (only ~25-35% of stays have a computable slope) carry
# net-positive signal — likely orthogonal to mean/std for the same concept
# and high-info on the high-acuity subset where they are present. See
# CLAUDE.md ## Model Improvements (Improvement 3).
MISSINGNESS_DROP_THRESHOLD: float | None = None
