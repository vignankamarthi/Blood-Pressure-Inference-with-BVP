# Process Flow Chart: Cuffless Blood Pressure Estimation from PPG

**Vignan Kamarthi** | DS4400 Final Project + Research

---

```
+---------------------------------------------------------------+
|                     DATA ACQUISITION                          |
|                                                               |
|  PulseDB v2.0 -- large-scale PPG dataset                     |
|  5.2M ten-second segments from 5,361 subjects                |
|  Raw PPG waveforms at 125 Hz                                 |
|  Ground-truth SBP/DBP from invasive arterial blood pressure  |
|  Pre-defined subject-level train/test splits                 |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                   FEATURE EXTRACTION                          |
|               Pure Rust implementation                        |
|                                                               |
|  +------------------+  +----------------+  +--------------+  |
|  | Catch22          |  | Entropy        |  | Statistical  |  |
|  | 22 canonical     |  | Permutation,   |  | Mean, std,   |  |
|  | time-series      |  | sample,        |  | skewness,    |  |
|  | features:        |  | approximate,   |  | kurtosis,    |  |
|  | autocorrelation, |  | spectral,      |  | RMS, min,    |  |
|  | distribution,    |  | Fisher-Shannon,|  | max, median  |  |
|  | fluctuation,     |  | Renyi, Tsallis |  |              |  |
|  | spectral         |  | complexity     |  | 8 features   |  |
|  | 22 features      |  | 11 features    |  |              |  |
|  +------------------+  +----------------+  +--------------+  |
|                                                               |
|  ~41 features per segment | Parallelized across 56 CPU cores |
|  Modular framework: each group selectable independently      |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                  PREPROCESSING                                |
|                                                               |
|  No pre-extraction normalization                              |
|  All features extracted from raw PPG signal directly          |
|  (Entropy measures are scale-invariant by construction;       |
|   Catch22/stats features preserve scale-dependent info)       |
|                                                               |
|  Post-extraction: StandardScaler PER FEATURE COLUMN           |
|    - Each of the 41 features independently normalized         |
|      to mean=0, std=1                                         |
|    - No inter-feature standardization (each feature gets      |
|      its own mean/std from its own column only)               |
|    - Fitted on training subjects ONLY                         |
|    - Applied (transform) to all data                          |
|    - Required for Ridge (L2 penalty is scale-sensitive)        |
|    - Neutral for tree-based models (split on thresholds)      |
|                                                               |
|  Subject-level train/test separation enforced                 |
|  No data leakage between splits                              |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|             HYPERPARAMETER OPTIMIZATION                       |
|                                                               |
|  Optuna Bayesian search (100 trials per model)                |
|  Cross-validated on training set                              |
|  Persistent storage (SQLite) -- resumable across sessions     |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                  MODEL TRAINING                               |
|          Separate models for SBP and DBP                      |
|                                                               |
|  +-------------------+                                        |
|  | Ridge Regression  |  Linear baseline with L2               |
|  |                   |  regularization. Interpretable          |
|  |                   |  coefficients show which features       |
|  |                   |  have linear relationships with BP.     |
|  +-------------------+                                        |
|           |                                                   |
|           v                                                   |
|  +-------------------+                                        |
|  | Decision Tree     |  Nonlinear baseline. Gini-based        |
|  |                   |  feature importance. Visualizable       |
|  |                   |  splits reveal which PPG properties     |
|  |                   |  the model uses first.                  |
|  +-------------------+                                        |
|           |                                                   |
|           v                                                   |
|  +-------------------+                                        |
|  | Random Forest     |  Bagging ensemble of decision trees.   |
|  |                   |  Reduces variance of individual trees.  |
|  |                   |  Aggregated feature importance is more  |
|  |                   |  stable than a single tree.             |
|  +-------------------+                                        |
|           |                                                   |
|           v                                                   |
|  +-------------------+                                        |
|  | XGBoost           |  Gradient boosting. Each tree corrects  |
|  |                   |  the residual errors of the previous    |
|  |                   |  ensemble. Newton-step optimization     |
|  |                   |  using both gradient and Hessian.       |
|  +-------------------+                                        |
|           |                                                   |
|           v                                                   |
|  +-------------------+                                        |
|  | LightGBM          |  Histogram-based gradient boosting.     |
|  |                   |  Leaf-wise tree growth for efficiency   |
|  |                   |  on large datasets. Same boosting math  |
|  |                   |  as XGBoost, optimized for 5.2M rows.  |
|  +-------------------+                                        |
|                                                               |
|  Progression: linear -> tree -> bagged -> boosted (x2)        |
|  Each step adds model complexity with clear justification     |
+---------------------------------------------------------------+
                            |
                            v
+---------------------------------------------------------------+
|                      EVALUATION                               |
|  Using clinical evaluation practices aligned with standards   |
|  applied in real-world BP device validation                   |
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
|    Feature importance comparison across models                |
|    Subject-level aggregation                                  |
|                                                               |
|  Generalization:                                              |
|    Evaluated on both calibration-based and calibration-free   |
|    test sets to measure subject-independent performance       |
+---------------------------------------------------------------+
```
