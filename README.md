# Blood Pressure Inference with BVP

**Cuffless blood pressure estimation from PPG waveforms using Catch22 + entropy feature extraction and ensemble learning.**

Predicts systolic (SBP) and diastolic (DBP) blood pressure from raw photoplethysmography signals. Pure Rust feature extraction (40 features) feeds into a 5-model ML pipeline with clinical-grade evaluation (AAMI, BHS, Bland-Altman).

**Dataset**: [PulseDB v2.0](https://github.com/pulselabteam/PulseDB) -- 5.2M segments, 5,361 subjects, MIMIC-III (USA) + VitalDB (South Korea)
**Course**: DS4400 Machine Learning 1 (Northeastern University, Spring 2026)

---

## Pipeline

```
PulseDB PPG (125 Hz, 10-sec segments)
    |
    v
Feature Extraction [Rust, 56-core parallel]
    |-- Catch22: 22 canonical time-series features
    |-- Entropy:  10 measures (ordinal + amplitude-based)
    |-- Stats:     8 descriptives
    |
    = 40 features per segment
    |
    v
StandardScaler [per feature column, fit on training subjects only]
    |
    v
Model Training [Optuna tuning, 100 trials/model]
    |-- Ridge Regression   (linear baseline)
    |-- Decision Tree      (nonlinear baseline)
    |-- Random Forest      (bagging)
    |-- XGBoost            (gradient boosting)
    |-- LightGBM           (histogram boosting)
    |
    v
Clinical Evaluation
    |-- MAE, RMSE, R-squared
    |-- AAMI compliance (ME < 5 mmHg, SD < 8 mmHg)
    |-- BHS grading (A/B/C/D)
    |-- Bland-Altman agreement
    |-- BP-category stratified analysis
```

---

## Repository Structure

```
Blood-Pressure-Inference-with-BVP/
├── Feature-Extraction-Rust-Complete/  # Git submodule -- Rust feature library
├── src/
│   ├── data_loader.py                # PulseDB loading, StandardScaler
│   ├── models.py                     # 5 models with per-model checkpointing
│   ├── tuning.py                     # Optuna (SQLite persistence, auto-resume)
│   ├── evaluation.py                 # AAMI, BHS, Bland-Altman, stratified eval
│   └── utils.py                      # Atomic writes, logging, serialization
├── scripts/
│   ├── train_models.py               # --resume, --fresh, --tune
│   ├── evaluate.py                   # Standalone evaluation
│   ├── generate_subsets.py           # Python replacement for MATLAB script
│   ├── download_pulsedb.sh           # Resumable dataset download (26 parts)
│   ├── setup_cluster.sh              # NEU Explorer one-time setup
│   ├── extract_features.sbatch       # SLURM: 56 CPUs, 128G, auto-resume
│   ├── tune_models.sbatch            # SLURM: Optuna SQLite survives wall-time
│   └── train_models.sbatch           # SLURM: per-model checkpoints
├── configs/
│   ├── model_configs.json            # Default hyperparameters
│   └── feature_configs.json          # Extraction + normalization config
├── tests/                            # 40 pytest tests
├── reports/
│   ├── proposal/                     # Submitted 2/27
│   ├── milestone/                    # Due 3/20
│   └── final/                        # Due 4/10
└── data/                             # PulseDB (gitignored, cluster-only)
```

---

## Checkpointing

Every operation is resumable. SLURM wall-time kills lose zero work.

| Component | Checkpoint Granularity |
|-----------|----------------------|
| Rust extraction | Per subject, atomic CSV writes |
| Random Forest | Every 50 trees (warm_start) |
| XGBoost | Every 25 boosting rounds |
| LightGBM | Every 25 iterations |
| Optuna tuning | Per trial (SQLite auto-persist) |

```bash
# Resume any interrupted pipeline stage
python scripts/train_models.py --resume
```

---

## Quick Start

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/vignankamarthi/Blood-Pressure-Inference-with-BVP.git
cd Blood-Pressure-Inference-with-BVP

# Install Python dependencies
pip install -r requirements.txt

# Build Rust feature extractor
cd Feature-Extraction-Rust-Complete && cargo build --release && cd ..

# Run tests
cargo test --manifest-path Feature-Extraction-Rust-Complete/Cargo.toml
python -m pytest tests/ -v
```

---

## Normalization Strategy

Raw PPG signal goes directly into all feature extractors (no pre-extraction normalization). Post-extraction, a `StandardScaler` normalizes each of the 40 feature columns independently -- fitted on training subjects only, applied to all data. No per-subject normalization (validated failure mode from prior research: 75.4% CV to 32.8% LOSO collapse).
