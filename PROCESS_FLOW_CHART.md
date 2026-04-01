# Process Flow Chart: Cuffless Blood Pressure Estimation

**Vignan Kamarthi** | DS4400 Final Project + Research Publication

---

## Experiment Overview

Two-track experiment: classical ML with signal ablation + deep learning on raw PPG.
72 total model evaluations + GradientSHAP interpretability analysis.

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
(CPU, 56 cores)                            (GPU, H200/A100)
    |                                          |
    v                                          v
Rust Feature Extraction              Raw PPG waveform only
40 features per signal:              1250 samples at 125 Hz
  Catch22 (22)                       No feature extraction
  Entropy (10)                       DL learns its own features
  Stats (8)                              |
    |                                    v
    v                                2 Architectures:
StandardScaler                         ResNet-BiGRU (primary)
(per column, train only)               ResNet-1D (baseline)
    |                                    |
    v                                    v
5 Models (x4 signal configs):       PPG only (no ablation):
  Ridge                              Both SBP and DBP targets
  Decision Tree                          |
  Random Forest                          v
  XGBoost                           GradientSHAP
  LightGBM                          Temporal importance maps
    |                                (which PPG regions -> BP)
    |                                    |
    +=================+=================+
                      |
                      v
           EVALUATION (same for both tracks)
             CalFree + CalBased + AAMI test sets
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
|                                                               |
|  Loaded via h5py targeted reads (not mat73):                 |
|    Subject files: 21-223x speedup over mat73                 |
|    Info files: targeted field reads on 3.4 GB HDF5           |
|  Stored as pre-allocated 2D float64 .npz arrays             |
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
|  Runs per-subset with per-directory checkpointing            |
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
|  Key-based merge: (file_name, segment_id) -> .npz labels     |
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
|  Signal Ablation (4 configs):                                |
|    PPG only (40 features)                                    |
|    PPG + ECG (80 features)                                   |
|    PPG + ABP (80 features)                                   |
|    PPG + ECG + ABP (120 features)                            |
|                                                               |
|  All with exhaustive checkpointing:                           |
|    RF: every 50 trees | XGB: every 25 rounds                 |
|    LGB: every 25 iters | Optuna: per trial (SQLite)          |
|                                                               |
|  Interpretability: impurity/gain (trees), coefficients (Ridge)|
+---------------------------------------------------------------+
```

---

## Track 2: Deep Learning on Raw PPG

```
+---------------------------------------------------------------+
|              RAW SIGNAL INPUT (PPG only)                      |
|                                                               |
|  Input: (batch, 1, 1250) -- single-channel raw PPG           |
|  No multi-signal ablation for DL                             |
|  DL learns its own features end-to-end from the waveform     |
|                                                               |
|  Per-channel StandardScaler (fit on training subjects only)   |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|           DL MODEL TRAINING (GPU, H200/A100)                  |
|          Separate models for SBP and DBP                      |
|                                                               |
|  ResNet-BiGRU  -- 1D residual blocks capture morphological    |
|    (primary)      features (waveform shape), bidirectional    |
|                   GRU captures temporal dependencies across   |
|                   the full 10-second window (beat-to-beat)    |
|                   Fan 2026: MAE 4.78/2.98 on VitalDB         |
|                                                               |
|  ResNet-1D     -- 1D residual blocks + global avg pool       |
|    (baseline)     No temporal component (ablation baseline)   |
|                   Moulaeifard 2025: best on PulseDB           |
|                                                               |
|  MSE loss, Adam, LR scheduling, early stopping               |
|  Checkpointing: model + optimizer state every epoch           |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|           INTERPRETABILITY (GradientSHAP)                     |
|                                                               |
|  GradientSHAP via captum (PyTorch-native)                    |
|  Approximates Shapley values using integrated gradients       |
|  with stochastic baselines                                   |
|                                                               |
|  Output: per-timestep importance map (1250 values)           |
|    Shows which regions of the PPG waveform drive BP prediction|
|    Expected: dicrotic notch / reflected wave regions          |
|    (arterial stiffness correlates with BP)                   |
|                                                               |
|  Novel contribution: no prior work applies GradientSHAP      |
|  directly to raw PPG temporal maps for BP estimation         |
+---------------------------------------------------------------+
```

---

## Experiment Matrix

```
+---------------------------------------------------------------+
|                   EXPERIMENT DESIGN                           |
|                                                               |
|  CLASSICAL ML (Track 1):                                     |
|    5 models x 4 signal configs x 2 targets x 3 test sets    |
|    = 60 evaluations                                          |
|    Ablation answers: does adding ECG/ABP to Catch22          |
|    features improve over PPG alone?                          |
|                                                               |
|  DEEP LEARNING (Track 2):                                    |
|    2 architectures x 1 signal (PPG) x 2 targets x 3 tests   |
|    = 12 evaluations                                          |
|    Comparison answers: does ResNet-BiGRU's temporal           |
|    modeling outperform ResNet-1D's local-only approach?       |
|                                                               |
|  CROSS-TRACK:                                                |
|    Best classical (feature-engineered) vs best DL (raw PPG)  |
|    Both on PPG-only for fair comparison                      |
|    Interpretability: impurity/gain vs GradientSHAP maps      |
|                                                               |
|  TOTAL: 72 evaluations + GradientSHAP analysis              |
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
|    Feature importance -- impurity/gain (classical models)     |
|    GradientSHAP temporal maps (DL models)                    |
|    Cross-track comparison (classical ML vs DL)                |
|    Cross-config comparison (ablation results, Track 1 only)   |
|                                                               |
|  Generalization:                                              |
|    CalFree (primary -- hardest, no subject calibration)       |
|    CalBased (calibration data available)                      |
|    AAMI (clinical standard test set)                         |
+---------------------------------------------------------------+
```
