# Blood Pressure Inference with BVP

**Cuffless blood pressure estimation from physiological signals using Catch22 + entropy feature extraction, ensemble learning, and deep learning.**

Two-track experiment: handcrafted features + traditional ML vs. learned features + deep learning. 4-configuration ablation study across PPG, ECG, and ABP signals proves PPG-only sufficiency for wearable deployment.

**Dataset**: [PulseDB v2.0](https://github.com/pulselabteam/PulseDB) -- 5.2M segments, 5,361 subjects, MIMIC-III (USA) + VitalDB (South Korea)

---

## Process Flow

```
PulseDB v2.0 (5.2M segments, 5,361 subjects, 125 Hz)
    |
    +-- PPG_Record (raw photoplethysmography -- wearable-viable)
    +-- ECG_F (electrocardiogram -- clinical)
    +-- ABP_Raw (arterial blood pressure -- invasive, gold standard)
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
  Catch22 (22)                           |
  Entropy (10)                           v
  Stats (8)                          4 Architectures:
    |                                  1D CNN
    v                                  LSTM
StandardScaler                         CNN-LSTM
(per column, train only)               Transformer
    |                                    |
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
    Phase 1: PPG-Only Championship
      9 models (5 ML + 4 DL) x 2 targets = 18 trainings
      Crown best ML model and best DL model
                     |
    Phase 2: Ablation with Champions
      Best ML + best DL x 4 configs x 2 targets = 16 trainings
        +-- PPG only        (cuffless wearable)
        +-- PPG + ECG       (does ECG help?)
        +-- PPG + ABP       (gold standard ceiling)
        +-- PPG + ECG + ABP (absolute upper bound)
                     |
                     v
              EVALUATION
      MAE, RMSE, R-squared
      AAMI compliance (ME < 5, SD < 8 mmHg)
      BHS grading (A/B/C/D)
      Bland-Altman agreement
      Cross-track comparison (ML vs DL)
      Cross-config comparison (ablation)
```

**Total: 34 model trainings. Question answered: is PPG alone sufficient for clinically meaningful BP estimation?**

---

## Repository Structure

```
Blood-Pressure-Inference-with-BVP/
├── Feature-Extraction-Rust-Complete/  # Git submodule -- Rust feature library
├── src/
│   ├── data_loader.py                # PulseDB loading, StandardScaler
│   ├── models.py                     # 5 ML models with per-model checkpointing
│   ├── tuning.py                     # Optuna (SQLite persistence, auto-resume)
│   ├── evaluation.py                 # AAMI, BHS, Bland-Altman, stratified eval
│   ├── dl_models.py                  # 1D CNN, LSTM, CNN-LSTM, Transformer
│   ├── dl_training.py                # PyTorch training loop + checkpointing
│   ├── dl_data.py                    # PyTorch Dataset for raw signal windows
│   └── utils.py                      # Atomic writes, logging, serialization
├── scripts/
│   ├── train_models.py               # ML training: --resume, --config, --tune
│   ├── train_dl_models.py            # DL training: --config, --model, --target
│   ├── evaluate.py                   # Standalone evaluation
│   ├── generate_subsets.py           # Python replacement for MATLAB script
│   ├── inspect_mat_helper.py         # GATE G5: verify .mat structure
│   ├── *.sbatch                      # SLURM scripts for all cluster operations
│   └── setup_cluster.sh              # NEU Explorer one-time setup
├── configs/                          # Model + feature configuration
├── tests/                            # 51 pytest tests + 24 Rust tests
├── reports/                          # Proposal, milestone, final report
└── data/                             # PulseDB (gitignored, cluster-only, 963 GB)
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
| DL training | Every epoch (model + optimizer state) |

---

## Quick Start

```bash
# Clone with submodule
git clone --recurse-submodules https://github.com/vignankamarthi/Blood-Pressure-Inference-with-BVP.git
cd Blood-Pressure-Inference-with-BVP

# Install dependencies
pip install -r requirements.txt

# Build Rust feature extractor
cd Feature-Extraction-Rust-Complete && cargo build --release && cd ..

# Run tests (75 total: 24 Rust + 51 Python)
cargo test --manifest-path Feature-Extraction-Rust-Complete/Cargo.toml
python -m pytest tests/ -v
```

---

## Normalization

**Track 1 (ML):** Raw signal goes directly into all feature extractors. Post-extraction, `StandardScaler` normalizes each of the 40 feature columns independently -- fitted on training subjects only. No per-subject normalization (validated failure mode: 75.4% CV to 32.8% LOSO collapse).

**Track 2 (DL):** Per-channel `StandardScaler` on raw 1250-point windows. Fitted on training subjects only.
