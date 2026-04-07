# Blood Pressure Inference with BVP

**Cuffless blood pressure estimation from physiological signals using Catch22 + entropy feature extraction, ensemble learning, and deep learning.**

Two-track experiment: handcrafted features + traditional ML vs. learned features + deep learning. 2-configuration ablation (PPG vs PPG+ECG) tests whether adding ECG to PPG improves cuffless BP estimation in a wearable-plausible setup.

**Dataset**: [PulseDB v2.0](https://github.com/pulselabteam/PulseDB) -- 5.2M segments, 5,361 subjects, MIMIC-III (USA) + VitalDB (South Korea)

---

## Process Flow

```
PulseDB v2.0 (5.2M segments, 5,361 subjects, 125 Hz)
    |
    +-- PPG_Record (raw photoplethysmography -- wearable-viable)
    +-- ECG_F (electrocardiogram -- clinical)
    +-- ABP_Raw (arterial blood pressure -- LABEL SOURCE ONLY;
         SBP/DBP are peaks/troughs of this waveform, so ABP
         is never used as a predictive feature)
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
40 features per signal:              1250 samples, 1 channel
  Catch22 (22)                       DL learns its own features
  Entropy (10)                           |
  Stats (8)                              v
    |                                2 Architectures:
    v                                  ResNet-BiGRU (primary)
StandardScaler                         ResNet-1D (baseline)
(per column, train only)                 |
    |                                    v
    v                                GradientSHAP
5 Models x 2 signal configs:        temporal importance maps
  Ridge                                  |
  Decision Tree                          |
  Random Forest                          |
  XGBoost                                |
  LightGBM                               |
    |                                    |
    +================+=================+
                     |
                     v
              EVALUATION
                     |
    Classical ML: 5 models x 2 signal ablations x 2 targets
      = 30 evaluations (CalFree + CalBased + AAMI)
      Ablation: PPG | PPG+ECG
                     |
    Deep Learning: 2 architectures x PPG only x 2 targets
      = 12 evaluations (CalFree + CalBased + AAMI)
      No ablation -- DL on raw PPG end-to-end
                     |
    Cross-track: best classical vs best DL on PPG-only
                     |
                     v
         42 total evaluations + GradientSHAP
      MAE, RMSE, R-squared
      AAMI compliance (ME < 5, SD < 8 mmHg)
      BHS grading (A/B/C/D)
      Bland-Altman agreement
      Cross-track comparison (ML vs DL)
      Cross-config comparison (ablation)
```

**Total: 24 model trainings (10 classical x 2 configs + 4 DL on PPG only), 42 evaluations + GradientSHAP analysis.**

---

## Key Results (CalFree Test Set, 111,600 samples)

CalFree is the hardest evaluation: test subjects have zero calibration data in training.

**Classical ML (Track 1) -- Best: LightGBM**

| Model | SBP MAE | SBP R2 | DBP MAE | DBP R2 | AAMI | BHS |
|-------|---------|--------|---------|--------|------|-----|
| LightGBM (PPG) | 14.46 | 0.211 | 8.54 | 0.201 | FAIL | D |
| LightGBM (PPG+ECG) | 14.43 | 0.219 | 8.66 | 0.194 | FAIL | D |
| Random Forest (PPG) | 14.69 | 0.193 | 8.66 | 0.177 | FAIL | D |
| XGBoost (PPG) | 15.52 | 0.113 | 8.91 | 0.147 | FAIL | D |
| Ridge (PPG) | 15.50 | 0.113 | 9.14 | 0.115 | FAIL | D |
| Decision Tree (PPG) | 15.28 | 0.124 | 8.96 | 0.131 | FAIL | D |

**Deep Learning (Track 2) -- Best: ResNet-BiGRU**

| Model | SBP MAE | SBP R2 | DBP MAE | DBP R2 | AAMI | BHS (DBP) |
|-------|---------|--------|---------|--------|------|-----------|
| ResNet-BiGRU | 13.61 | 0.266 | 7.97 | 0.284 | FAIL | **C** |
| ResNet-1D | 13.84 | 0.253 | 7.90 | 0.315 | FAIL | D |

**Key Findings:**
- No model achieves AAMI clinical compliance (all SD > 8 mmHg)
- ResNet-BiGRU DBP achieves BHS Grade C (40.8% within 5 mmHg, 69.7% within 10 mmHg)
- DL outperforms classical ML on both SBP and DBP without any feature engineering
- Adding ECG to PPG provides negligible improvement (+0.03 mmHg SBP MAE), suggesting PPG alone captures the relevant hemodynamic information
- GradientSHAP reveals the model concentrates on timesteps 7.4-9.0s (last 2-3 seconds of the waveform) for SBP, and splits attention between early (0.5-0.7s) and late (8.7-9.5s) regions for DBP
- Results consistent with Moulaeifard 2025 PulseDB benchmark (SBP MAE 13.9)

---

## Repository Structure

```
Blood-Pressure-Inference-with-BVP/
├── Feature-Extraction-Rust-Complete/  # Rust feature extraction library (embedded)
├── src/
│   ├── data_loader.py                # PulseDB loading, StandardScaler
│   ├── models.py                     # 5 ML models with per-model checkpointing
│   ├── tuning.py                     # Optuna (SQLite persistence, auto-resume)
│   ├── evaluation.py                 # AAMI, BHS, Bland-Altman, stratified eval
│   ├── dl_models.py                  # ResNet-BiGRU, ResNet-1D (raw PPG -> BP)
│   ├── dl_training.py                # PyTorch training loop + checkpointing
│   ├── dl_data.py                    # PyTorch Dataset for raw signal windows
│   └── utils.py                      # Atomic writes, logging, serialization
├── scripts/
│   ├── train_models.py               # ML training: --resume, --config, --tune
│   ├── train_dl_models.py            # DL training: --config, --model, --target
│   ├── evaluate.py                   # Standalone evaluation
│   ├── run_gradient_shap.py          # GradientSHAP temporal attribution analysis
│   ├── generate_subsets.py           # PulseDB subject-level subset generation
│   ├── merge_features_labels.py      # Key-based feature + label merge
│   ├── inspect_*.py / probe_*.py     # Data verification scripts
│   ├── *.sbatch                      # SLURM scripts for all cluster operations
│   └── setup_cluster.sh              # NEU Explorer one-time setup
├── configs/                          # Model + feature + ablation configuration
├── tests/                            # 58 pytest tests + 24 Rust tests
├── results/                          # Evaluation metrics, leaderboards, SHAP analysis
├── reports/                          # Proposal, milestone, final report (LaTeX)
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
git clone https://github.com/vignankamarthi/Blood-Pressure-Inference-with-BVP.git
cd Blood-Pressure-Inference-with-BVP

# Install dependencies
pip install -r requirements.txt

# Build Rust feature extractor
cd Feature-Extraction-Rust-Complete && cargo build --release && cd ..

# Run tests (82 total: 24 Rust + 58 Python)
cargo test --manifest-path Feature-Extraction-Rust-Complete/Cargo.toml
python -m pytest tests/ -v
```

---

## Normalization

**Track 1 (ML):** Raw signal goes directly into all feature extractors. Post-extraction, `StandardScaler` normalizes each of the 40 feature columns independently -- fitted on training subjects only. No per-subject normalization (validated failure mode: 75.4% CV to 32.8% LOSO collapse).

**Track 2 (DL):** Per-channel `StandardScaler` on raw 1250-point windows. Fitted on training subjects only.
