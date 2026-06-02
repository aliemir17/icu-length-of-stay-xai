# ICU Length of Stay — Explainable AI

![Python](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-FF6600)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-1B7AC2)
![SHAP](https://img.shields.io/badge/SHAP-XAI-7E5BEF)
![LIME](https://img.shields.io/badge/LIME-XAI-9B59B6)
![MIMIC-IV](https://img.shields.io/badge/data-MIMIC--IV%203.1-005EB8)
![License](https://img.shields.io/badge/license-MIT-green)
![Thesis](https://img.shields.io/badge/B.Sc.%20thesis-ITU%20IE-red)

B.Sc. thesis project (ITU Industrial Engineering, **final submission 2026-06-08**): predict ICU length of stay from early-admission clinical data and explain each prediction with three independent XAI methods.

**Team:** Ali Emir İBİCİ, Mete Berk TUNÇER, Ali AVCI
**Supervisor:** Mehmet Yasin ULUKUŞ
**Data:** MIMIC-IV 3.1 (credentialed full release, 48,222 ICU stays after cohort filters). A synthetic data path is retained for pipeline development without DUA access.

---

## Results at a glance

### Headline metrics (test set, N = 9,645)

| Task | Best model | Metric | Value | vs Mean baseline |
|---|---|---|---|---|
| Regression (log1p target) | **LGBM** | Test MAE | **2.168 days** | -0.385 d |
| Regression (raw days) | LGBM | Test MAE | 2.456 d | — |
| Classification — 7-day long stay (**main**) | **XGB** | Test AUC-PR | **0.462** | +0.334 vs majority |
| Classification — 4.06-day cohort-mean (*comparison*) | XGB | Test AUC-PR | 0.596 | +0.334 vs majority |

> **Two classification thresholds, one main target.** The 7-day rule is the clinical convention used throughout (SHAP, LIME, permutation, calibration, subgroup fairness). The 4.06-day cohort-mean threshold is reported as an **ablation** so the thesis can show how class balance alone — 87/13 at 7-day vs 74/26 at 4.06-day — drives AUC-PR. The 4.06-day numbers are not used downstream; same models, retrained at the new cutoff, on the identical test split.

### What we built

- **Source-agnostic pipeline**: same code runs on synthetic data and on the full credentialed MIMIC-IV 3.1 release (chunked readers for 433 M chartevents + 158 M labevents rows).
- **152-feature cohort dataset**: vital signs (7 concepts × 5 aggregations = 35), labs (14 × 5 = 70), demographics (5), missing-indicators (40), measurement-counts (2).
- **4 models × 4 tasks**, all Optuna-tuned: Linear (Ridge / Logistic), Random Forest, XGBoost, LightGBM. Tasks: log1p regression, raw-days regression, 7-day classification (main), 4.06-day classification (ablation).
- **Three independent XAI methods**: SHAP (TreeSHAP), LIME, permutation importance — all four models, regression + classification.
- **Calibration + confusion matrix + baselines**: clinically reliable probabilities, optimal F1 threshold per model.
- **Subgroup + fairness analysis**: sepsis, age, gender, race, insurance — 5 axes, all 4 models.

---

## Project layout

```
.
├── data/
│   ├── raw/         # MIMIC raw CSVs — gitignored (PhysioNet DUA)
│   ├── processed/   # Parquet cache (mimic4_full_icu_stays.parquet)
│   └── synthetic/   # Mock data for pipeline development
├── notebooks/       # Sequential analysis notebooks (run in order)
│   ├── 01_eda.ipynb              # EDA + cohort filters + descriptive stats
│   ├── 02_models.ipynb           # Multi-model comparison + baseline vs current
│   ├── 03_explainability.ipynb   # SHAP / LIME / permutation, cross-method
│   ├── 04_validation.ipynb       # Calibration, confusion matrix, baselines
│   └── 05_subgroups.ipynb        # Sepsis/age/gender/race/insurance fairness
├── src/
│   ├── config.py    # Paths, constants, itemid maps, hyperparameters
│   ├── data/        # MIMIC extraction (chunked readers) + synthetic generator
│   ├── features/    # Preprocessing + feature engineering (missing-indicators)
│   ├── models/      # Training + Optuna tuning
│   └── explain/     # SHAP global explainability (standalone CLI)
├── dashboard/       # Streamlit what-if analysis prototype (LGBM + XGB + SHAP)
├── models/          # Trained model artifacts + baseline snapshot (gitignored)
├── reports/
│   ├── figures/     # ~70 PNG figures across all notebooks
│   ├── tables/      # CSVs + Markdown comparison tables
│   └── baseline/    # Phase 1 frozen snapshot for comparison
└── requirements.txt
```

---

## Setup

```powershell
# venv uses Python 3.12
D:\Bitirme\Python\python.exe -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Pipeline commands

```powershell
# Synthetic data path (no external data needed)
python -m src.data.synthetic
python -m src.models.train --source synthetic
python -m src.explain.shap_explain --source synthetic

# MIMIC-IV path (requires the credentialed release at data/raw/mimic-iv-3.1/)
python -m src.data.mimic                       # Build cohort + parquet cache
python -m src.models.train --source mimic4     # Train all 12 model x task combos
python -m src.models.tune  --source mimic4     # Optuna hyperparameter tuning
python -m src.explain.shap_explain --source mimic4
```

MIMIC-IV requires PhysioNet credentialed access (DUA + CITI training).

### Dashboard (what-if analysis)

A Streamlit prototype lives in `dashboard/app.py`. Loads the project's best two pipelines — **LGBM regression** (log1p target) and **XGB classification** at the 7-day threshold — and lets the user pick a real test-set patient, edit demographics + every vital and lab mean (21 fields with an explicit "✓ Measured" toggle) + 3 top-SHAP vital slopes, and inspect the prediction + a per-feature SHAP waterfall for both models side-by-side against the unedited "Original" prediction.

```powershell
streamlit run dashboard/app.py
```

Six stratified preset patients (2 short-stay / 2 long-stay / 2 sepsis-positive) ship with the app; each was hand-picked from the test set under the rule **|model error| < 1.5 d AND no missing features**, so SHAP attributions are clean. A "Random patient" button pulls any other test-set stay.

**Measurement-intent UX.** Every vital and lab `_mean` field carries an explicit `✓ Measured` toggle. Switching it off blanks the value to NaN (the pipeline imputer fills it downstream) and — for the four lab means that have a `_missing` indicator in the model (`lactate`, `bilirubin`, `ph`, `inr`) — flips the paired indicator. The `n_vitals_measured` / `n_labs_measured` features are recomputed live from the toggle state and shown in a sidebar caption (`vitals N/7, labs M/14`).

Not a medical device. Research demo only.

---

## Approach

### Cohort definition

Applied in sequence by `src/data/mimic.py::build_cohort`:

1. First ICU stay per patient (drop subsequent stays)
2. Age ≥ 18 at ICU admission
3. ICU LoS ≥ 1 day (we need first-24h features)
4. Patient did **not** die in the ICU (LoS is censored otherwise)

**Trace:** 94,458 raw → 65,366 (first stay) → 65,366 (age) → 51,838 (LoS ≥ 1) → **48,222** (no ICU death).

### Features (152 total)

```
 7 vital concepts × 5 aggregations (min/max/mean/std/slope)  = 35
14 lab   concepts × 5 aggregations                            = 70
age + 4 categorical (gender, admission_type, race, insurance) =  5
40 missing-indicator binary flags                             = 40
n_vitals_measured + n_labs_measured (acuity counts)           =  2
─────────────────────────────────────────────────────────────────
                                                              = 152
```

Demographic features (`age`, `gender`, `admission_type`, `race`, `insurance`) **are** used by the model and also serve as the stratification axes for the fairness analysis. The columns in `EXCLUDE_COLS` are IDs, timestamps, outcome flags (`hospital_expire_flag`, `died_in_icu`), the sepsis flag (used only for subgroup eval), and the target itself.

### Models

| Model | Final hyperparameters (after Optuna) |
|---|---|
| Linear | `RidgeCV(alphas=(0.1, 1, 10, 100, 1000))` regression; `LogisticRegression(class_weight=balanced)` |
| RF | n_estimators=100, max_depth=18, max_features=0.3, min_samples_leaf=8 |
| XGB regression | n_est=950, depth=6, lr=0.028, min_child_weight=1, reg_alpha=6.9 |
| XGB classification | n_est=950, depth=9, lr=0.014, min_child_weight=36, scale_pos_weight=6.81 |
| LGBM regression | n_est=900, depth=9, num_leaves=76, min_child_samples=89 |
| LGBM classification | n_est=800, depth=10, num_leaves=65, reg_lambda=9.45 |

### Validation

5-fold CV on 80 % training (KFold for regression, StratifiedKFold for classification, seed=42); single held-out 20 % test set drives the headline numbers.

---

## Model Improvement Journey

Iterative improvements from the Phase 1 baseline to the final model, each measured against the frozen baseline snapshot in `models/baseline/`.

| # | Change | Δ best model |
|---|---|---|
| 1 | `Ridge(alpha=1)` → `RidgeCV` + log-space prediction clip (fixes CV overflow on log1p target) | linear/CV MAE: 2.3e+227 → 2.49 |
| 2 | RF speedup: n_estimators 400→200, max_features=sqrt, min_leaf=5 | RF training 26 min → 4.5 min, perf unchanged |
| 3 | Missingness >40 % / >60 % feature drop (ablation) | **Rejected** — all 12 combos got worse; sparse slope features still carry signal |
| 4 | `scale_pos_weight` 5.06 → 6.81 (matches full-data class ratio 87.2/12.8) | XGB clf AUC-PR: -0.002 (noise) — kept for theoretical correctness |
| 5 | **Feature engineering**: +40 missing-indicators, +2 measurement-counts | Linear clf AUC-PR: 0.300 → **0.352** (+17%); LGBM reg test MAE: 2.196 → 2.191 |
| 6 | **Optuna tuning** (50 trials XGB, 30 trials LGBM/RF) | LGBM reg test MAE: 2.191 → **2.168**; XGB clf AUC-PR: 0.444 → **0.462** |

**Cumulative improvement (Phase 1 baseline → final):**
- Best regressor: LGBM test MAE **2.196 → 2.168** (−0.028 d)
- Best classifier: XGB test AUC-PR **0.440 → 0.462** (+0.022, +5 %)
- Linear classifier: AUC-PR **0.300 → 0.352** (+17 %, biggest single jump from feature engineering)

---

## Explainability findings 

**Three XAI methods, four models, two tasks = 12 global rankings + per-patient local explanations.**

### Cross-method agreement (top-10 overlap)

| Model | SHAP-LIME | SHAP-Perm | LIME-Perm |
|---|---|---|---|
| XGB | 6/10 | **8/10** | 5/10 |
| LGBM | 5/10 | 7/10 | 3/10 |
| RF | 6/10 | **9/10** | 5/10 |
| Linear | 3/10 (Coef-LIME) | 7/10 | 1/10 |

**SHAP and permutation importance agree strongly (7-9/10)** — the model's internal logic and its observed behavior point to the same features. LIME diverges moderately due to its local linear approximation. Regression vs Classification SHAP top-10 overlap: **7/10**.

### Top features (consistent across all tree models and methods)

`spo2_slope`, `dbp_slope`, `resp_rate_mean`, `n_labs_measured`, `temp_c_mean`, `admission_type`, `hematocrit_slope_missing`.

**Novel finding:** `n_labs_measured` (count of distinct lab tests ordered in first 24h) appears as a top-5 SHAP feature across all tree models. This *measurement intensity as acuity proxy* has not been explicitly reported in prior ICU LoS literature.

### Feature group analysis (% of total SHAP contribution)

| Group | XGB | LGBM | RF | Linear |
|---|---|---|---|---|
| Vital values | 30 % | 7 % | 26 % | 31 % |
| Vital slopes | 12 % | 1 % | 8 % | 13 % |
| Lab values | 24 % | 33 % | 24 % | 24 % |
| Missing indicators | 17 % | 31 % | 27 % | 17 % |
| Demographics | 7 % | 21 % | 3 % | 7 % |

Tree models rely heavily on vital trends + missing indicators; linear model favors vital absolute values + demographics. Model architecture affects not only accuracy but **which clinical factors are surfaced as explanations**.

### Linear vs tree divergence

Linear model's Ridge coefficients favor lab values (lactate, pH, bilirubin) while tree models favor vital trends (slopes) and missing-indicators — supporting the thesis argument that **explanation choice depends on model choice**.

---

## Validation rigor

- **Calibration plots** for all 4 classifiers (both thresholds) — tree models well-calibrated, linear is somewhat overconfident
- **Confusion matrices** at default 0.5 + optimal F1 threshold per model
  - 7-day (main): XGB threshold **0.525** → long-stay recall 54.3 %; LGBM threshold 0.532 → recall 57.0 %
  - 4.06-day (ablation): XGB threshold **0.480** → long-stay recall 62.9 %; LGBM threshold 0.469 → recall 65.6 %
- **Mean predictor baseline**: MAE 2.553 d, R² -0.043 — best ML model beats it by **0.385 days** (15 %)
- **Majority-class classification baseline**: 87.2 % accuracy but F1 = 0 — confirms why AUC-PR is the right metric

### Threshold ablation (same models, same test split)

| Model | AUC-PR 7d | AUC-PR 4.06d | F1 7d (opt thr) | F1 4.06d (opt thr) |
|---|---:|---:|---:|---:|
| Linear | 0.352 | 0.518 | 0.412 (thr=0.647) | 0.517 (thr=0.484) |
| RF     | 0.424 | 0.560 | 0.447 (thr=0.406) | 0.542 (thr=0.401) |
| XGB    | **0.462** | **0.596** | **0.475** (thr=0.525) | **0.566** (thr=0.480) |
| LGBM   | 0.455 | 0.595 | 0.477 (thr=0.532) | 0.562 (thr=0.469) |

The +0.13 to +0.17 jump in AUC-PR comes from class balance alone (positive share 12.8 % → 26.2 %), not better learning. We keep 7-day as the primary target because (a) it matches clinical planning units used in the literature (Hempel et al.), and (b) the 4.06-day cutoff is a moving target — it shifts with cohort changes.

---

## Subgroup & Fairness Analysis

Performance broken down across 5 clinically meaningful axes (all evaluated on the **same models**, just stratified at test time):

### Sepsis subgroup (13.8 % prevalence)

| Subgroup | actual_mean LoS | XGB test MAE | XGB test AUC-PR |
|---|---|---|---|
| non-sepsis (N=8,319) | 3.61 d | 1.88 d (52 % relative error) | 0.371 |
| sepsis (N=1,326) | 6.73 d | 3.98 d (59 % relative error) | **0.660** |

Sepsis-stratified top-10 SHAP comparison shows **7/10 features shared** — but sepsis-specific top features are `ph_min`, `ph_slope_missing`, `ph_std_missing` (acid-base disturbance). Non-sepsis-specific: `dbp_mean`, `hematocrit_slope_missing`, `map_mean`.

### Age groups

Regression MAE **decreases** with age (80+: 1.84 d, best). Classification AUC-PR **decreases** with age (80+: 0.34, worst). Trade-off: older patients have narrower LoS distribution (easier regression) but harder long-stay identification (multiple comorbidities confound).

### Fairness disparity table (max-min gap per axis)

| Model | Sepsis | Age | Gender | Race | Insurance |
|---|---|---|---|---|---|
| XGB MAE gap (d) | 2.10 | 0.48 | **0.07** | 0.74 | 0.53 |
| XGB AUC-PR gap | 0.29 | 0.22 | **0.01** | 0.12 | 0.22 |

**Gender bias: essentially zero (~0.07 d, 1 % relative)** — model is fair across genders.
**Race & insurance disparities are 10x larger than gender** — Other/Unknown race and Medicaid patients have noticeably worse MAE. This is a known clinical AI fairness concern that warrants further investigation.

---

## Data inventory

What the pipeline pulls from MIMIC and how each piece becomes a model feature. Source of truth in code: [src/config.py](src/config.py) and [src/data/mimic.py](src/data/mimic.py).

### Tables used

| Module | File                  | Full rows  | Used for                                           |
|--------|-----------------------|------------|----------------------------------------------------|
| icu    | `icustays.csv.gz`     |     94,458 | Cohort base; `los` (target), `intime` / `outtime`  |
| hosp   | `patients.csv.gz`     |    364,627 | `gender`, `anchor_age`, `anchor_year` (age calc)   |
| hosp   | `admissions.csv.gz`   |    546,028 | `admission_type`, `race`, `insurance`, `deathtime` |
| icu    | `chartevents.csv.gz`  |     ~433 M | First-24h vital aggregates (chunked + filtered)    |
| hosp   | `labevents.csv.gz`    |     ~158 M | First-24h lab aggregates (chunked + filtered)      |
| hosp   | `diagnoses_icd.csv.gz`|     ~5 M   | Sepsis flag derivation (ICD-9 99591/92, ICD-10 A40\*/A41\*/R6520/R6521) |

### First-24h window

For each row in `chartevents` / `labevents`, only events where `intime ≤ charttime < intime + 24h` are kept. The model predicts LoS from clinical information available within the first day of the stay — matching the "early planning" framing of the interim report and the literature standard (Hempel et al. 2023, Hasan et al. 2023). The cutoff also acts as a leakage guard.

### Vital itemids (7 concepts)

| Concept     | itemid(s)        | Notes                                              |
|-------------|------------------|----------------------------------------------------|
| heart_rate  | 220045           |                                                    |
| sbp         | 220179, 220050   | NIBP first, fall back to arterial line             |
| dbp         | 220180, 220051   | Same NIBP-then-arterial fallback                   |
| map         | 220181, 220052   | Same                                               |
| resp_rate   | 220210, 224690   | Spontaneous + ventilator-set rates merged          |
| spo2        | 220277           |                                                    |
| temp        | 223762 + 223761  | Celsius native; Fahrenheit converted and merged    |

### Lab itemids (14 concepts)

| Concept     | itemid | Clinical relevance                         |
|-------------|--------|--------------------------------------------|
| lactate     | 50813  | Tissue hypoperfusion, sepsis               |
| creatinine  | 50912  | Renal function                             |
| bun         | 51006  | Renal function                             |
| wbc         | 51301  | Infection / inflammation                   |
| hemoglobin  | 51222  | Bleeding, anemia                           |
| hematocrit  | 51221  | Volume status, bleeding                    |
| platelets   | 51265  | DIC, sepsis, bleeding risk                 |
| sodium      | 50983  | Volume / metabolic status                  |
| potassium   | 50971  | Cardiac risk                               |
| glucose     | 50931  | Stress hyperglycemia, diabetes             |
| bilirubin   | 50885  | Hepatic function                           |
| ph          | 50820  | Acid-base balance                          |
| anion_gap   | 50868  | Metabolic acidosis                         |
| inr         | 51237  | Coagulation, sepsis                        |

### Aggregations

Per `(stay_id, concept)`, within the first-24h window, five statistics are computed: **min, max, mean, std, slope**. Slope is the ordinary-least-squares slope of `valuenum` against hours-since-`intime`; it captures trajectory direction (rising vs falling) and is NaN when fewer than 3 measurements fall inside the window. Implementation: [src/data/mimic.py](src/data/mimic.py).

---

## Status

**Done** — All technical phases (1-5) complete:
- ✅ Project scaffolding, source-agnostic pipeline
- ✅ Cohort definition (48,222 stays), 152-feature dataset, chunked readers
- ✅ 4 models × 3 tasks, Optuna-tuned
- ✅ XAI centerpiece: SHAP + LIME + permutation, cross-method agreement
- ✅ Validation: calibration, confusion matrix, baselines
- ✅ Subgroup analysis: 5 fairness axes