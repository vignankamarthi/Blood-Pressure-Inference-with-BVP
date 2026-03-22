# Process Flow Chart: Cuffless Blood Pressure Estimation

**Vignan Kamarthi** | DS4400 Final Project + Research Publication

---

## Experiment Overview

Two-track experiment with 4-configuration ablation study.
34 total model trainings across feature engineering and deep learning.

```
PulseDB v2.0 (5.2M segments, 5,361 subjects, 125 Hz)
    |
    +-- PPG_Record (raw photoplethysmography)
    +-- ECG_F (electrocardiogram)
    +-- ABP_Raw (arterial blood pressure -- also source of SBP/DBP labels)
    |
    +==========================================+
    |        TWO PARALLEL TRACKS               |
    +==========================================+
    |                                          |
    v                                          v
TRACK 1: Feature Engineering + ML          TRACK 2: Deep Learning
(CPU, 56 cores)                            (GPU, V100/H200)
    |                                          |
    v                                          v
Rust Feature Extraction              Raw 1250-point windows
40 features per signal:              directly to neural networks
  Catch22 (22)                       No feature extraction
  Entropy (10)                           |
  Stats (8)                              v
    |                                4 Architectures:
    v                                  1D CNN
StandardScaler                         LSTM
(per column, train only)               CNN-LSTM
    |                                  Transformer
    v                                    |
5 Models:                                |
  Ridge                                  |
  Decision Tree                          |
  Random Forest                          |
  XGBoost                                |
  LightGBM                               |
    |                                    |
    +================+=================+
                     |
                     v
          2-PHASE EXPERIMENT
    |
    v
Phase 1: PPG-Only Championship (18 trainings)
  5 ML + 4 DL models, both SBP and DBP
  Crown best ML model and best DL model
    |
    v
Phase 2: Ablation with Champions (16 trainings)
  Best ML + best DL across 4 signal configs:
    |
    +-- PPG only          (40 ML features / 1250x1 DL)
    +-- PPG + ECG         (80 ML features / 1250x2 DL)
    +-- PPG + ABP         (80 ML features / 1250x2 DL)
    +-- PPG + ECG + ABP   (120 ML features / 1250x3 DL)
    |
    v
EVALUATION (clinical standards)
```

---

## Data Acquisition

```
+---------------------------------------------------------------+
|                     DATA ACQUISITION                          |
|                                                               |
|  PulseDB v2.0                                                |
|  5.2M ten-second segments from 5,361 subjects                |
|  Sources: MIMIC-III (USA) + VitalDB (South Korea)            |
|  3 signals per segment:                                      |
|    PPG_Record -- raw photoplethysmography (wearable-viable)   |
|    ECG_F      -- electrocardiogram (clinical)                 |
|    ABP_Raw    -- arterial blood pressure (invasive)           |
|  Labels: SBP/DBP derived from ABP waveform                   |
|  Pre-defined subject-level train/test splits                 |
|  Sampling rate: 125 Hz, segment length: 1250 points          |
+---------------------------------------------------------------+
```

---

## Track 1: Feature Engineering + Traditional ML

```
+---------------------------------------------------------------+
|               FEATURE EXTRACTION (Rust)                       |
|          40 features per signal, 56-core parallel             |
|                                                               |
|  Applied to each signal independently (PPG, ECG, ABP)        |
|  Same 40 features extracted from raw signal -- no pre-norm    |
|                                                               |
|  +------------------+  +----------------+  +--------------+  |
|  | Catch22 (22)     |  | Entropy (10)   |  | Stats (8)    |  |
|  | Autocorrelation  |  | Permutation    |  | Mean, median |  |
|  | Distribution     |  | Complexity     |  | Std, skew    |  |
|  | Fluctuation(DFA) |  | Fisher-Shannon |  | Kurtosis     |  |
|  | Spectral(Welch)  |  | Renyi, Tsallis |  | RMS          |  |
|  | Binary/motif     |  | Sample         |  | Min, max     |  |
|  | Local prediction |  | Approximate    |  |              |  |
|  +------------------+  +----------------+  +--------------+  |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                  PREPROCESSING                                |
|                                                               |
|  No pre-extraction normalization                              |
|  Post-extraction: StandardScaler PER FEATURE COLUMN           |
|    - Each feature independently normalized (mean=0, std=1)    |
|    - Fitted on training subjects ONLY                         |
|    - Required for Ridge, neutral for trees                    |
|  Subject-level train/test separation enforced                 |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|          HYPERPARAMETER OPTIMIZATION (Optuna)                 |
|  100 trials per model, TPE sampler, SQLite persistence        |
|  Objective: minimize MAE on cross-validated training set      |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|              ML MODEL TRAINING (CPU, 56 cores)                |
|          Separate models for SBP and DBP                      |
|                                                               |
|  Ridge        -- L2 linear baseline                           |
|  Decision Tree -- nonlinear baseline, feature importance      |
|  Random Forest -- bagging, variance reduction                 |
|  XGBoost      -- gradient boosting (Newton step)              |
|  LightGBM     -- histogram boosting (5.2M row efficiency)     |
|                                                               |
|  All with exhaustive checkpointing:                           |
|    RF: every 50 trees | XGB: every 25 rounds                 |
|    LGB: every 25 iters | Optuna: per trial (SQLite)          |
+---------------------------------------------------------------+
```

---

## Track 2: Deep Learning on Raw Signals

```
+---------------------------------------------------------------+
|              RAW SIGNAL INPUT (no feature extraction)          |
|                                                               |
|  Input shape depends on ablation config:                      |
|    PPG only:      1250 x 1 channels                          |
|    PPG + ECG:     1250 x 2 channels                          |
|    PPG + ABP:     1250 x 2 channels                          |
|    All three:     1250 x 3 channels                          |
|                                                               |
|  Per-channel StandardScaler (fit on training subjects only)   |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|           DL MODEL TRAINING (GPU, V100/H200)                  |
|          Separate models for SBP and DBP                      |
|                                                               |
|  1D CNN       -- Conv1D layers detect local pulse patterns    |
|  LSTM         -- Captures temporal dependencies               |
|  CNN-LSTM     -- Hybrid: local + temporal                     |
|  Transformer  -- Multi-head attention on time series          |
|                                                               |
|  All models: MSE loss, Adam, LR scheduling, early stopping   |
|  Checkpointing: model + optimizer state every epoch           |
|  Optuna: architecture + hyperparameter search (100 trials)    |
+---------------------------------------------------------------+
```

---

## Ablation Study Design

```
+---------------------------------------------------------------+
|               2-PHASE EXPERIMENT DESIGN                       |
|                                                               |
|  PHASE 1: PPG-Only Championship (18 trainings)               |
|    5 ML models x 2 targets (SBP, DBP) = 10                  |
|    4 DL models x 2 targets (SBP, DBP) = 8                   |
|    --> Crown best ML model and best DL model                 |
|                                                               |
|  PHASE 2: Ablation with Champions (16 trainings)             |
|    Best ML x 4 configs x 2 targets = 8                       |
|    Best DL x 4 configs x 2 targets = 8                       |
|                                                               |
|  Configs:                                                     |
|    A. PPG only          (cuffless wearable)                   |
|    B. PPG + ECG         (does ECG help?)                      |
|    C. PPG + ABP         (gold standard ceiling)               |
|    D. PPG + ECG + ABP   (absolute upper bound)               |
|                                                               |
|  Question answered: Is PPG alone sufficient for               |
|  clinically meaningful BP estimation?                         |
+---------------------------------------------------------------+
```

---

## Evaluation

```
+---------------------------------------------------------------+
|                      EVALUATION                               |
|                                                               |
|  Accuracy:                                                    |
|    ME, SD, MAE, RMSE, R-squared, Median AE, Max Error        |
|                                                               |
|  Clinical Standards:                                          |
|    AAMI compliance (ME < 5 mmHg, SD < 8 mmHg)                |
|    BHS grading (A/B/C/D by % within 5/10/15 mmHg)            |
|                                                               |
|  Analysis:                                                    |
|    Bland-Altman agreement plots                               |
|    Error stratified by BP category                            |
|    Feature importance (Track 1 models)                        |
|    Cross-track comparison (ML vs DL)                          |
|    Cross-config comparison (ablation results)                 |
|    Subject-level aggregation                                  |
|                                                               |
|  Generalization:                                              |
|    Calibration-based and calibration-free test sets            |
+---------------------------------------------------------------+
```
