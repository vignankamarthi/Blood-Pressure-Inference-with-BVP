# Feature Extraction Rust Complete

**Pure Rust implementation of 40 canonical time-series features for physiological signal analysis.**

Novel Catch22 implementation in Rust (no existing crate), entropy measures adapted from validated AI4Pain research (ICMI 2025), and statistical descriptives. Designed for large-scale PPG feature extraction on HPC clusters.

---

## Features

| Framework | Count | Features |
|-----------|-------|----------|
| **Catch22** | 22 | Autocorrelation, distribution, fluctuation analysis (DFA), spectral, binary transitions, local prediction, periodicity |
| **Entropy** | 10 | Permutation, statistical complexity, Fisher-Shannon, Fisher information, Renyi, Tsallis, sample, approximate |
| **Statistical** | 8 | Mean, median, std, skewness, kurtosis, RMS, min, max |

- **Rayon parallelization**: Scales across all available CPU cores
- **Atomic checkpointing**: `--resume` picks up from exact interruption point
- **TDD validated**: Numerical agreement with pycatch22, ordpy, numpy/scipy

---

## Quick Start

```bash
# Build optimized binary
cargo build --release

# Extract all 40 features from CSV signal files
./target/release/feature_extraction_rust_complete extract \
    --input-dir data/raw \
    --output-dir data/features \
    --frameworks catch22,entropy,stats \
    --workers 56

# Resume interrupted extraction
./target/release/feature_extraction_rust_complete extract \
    --input-dir data/raw \
    --output-dir data/features \
    --resume
```

---

## Architecture

```
src/
├── main.rs              # CLI (clap): extract, validate subcommands
├── lib.rs               # Public API
├── catch22/             # 22 Catch22 features (novel Rust implementation)
│   ├── autocorrelation  # CO_f1ecac, CO_FirstMin, CO_trev, CO_Embed2, CO_HistogramAMI
│   ├── distribution     # DN_HistogramMode (5,10), DN_OutlierInclude (p,n)
│   ├── fluctuation      # SC_FluctAnal (DFA, RS range)
│   ├── binary           # SB_BinaryStats, SB_TransitionMatrix, SB_MotifThree
│   ├── local            # FC_LocalSimple (mean1, mean3), IN_AutoMutualInfo
│   └── successive       # MD_hrv_classic, PD_Periodicity, SP_Summaries (Welch)
├── entropy/             # 10 entropy measures (adapted from AI4Pain V2)
│   ├── permutation      # Bandt-Pompe permutation entropy
│   ├── complexity       # Statistical complexity, Fisher information
│   ├── generalized      # Renyi, Tsallis (q-parameterized)
│   ├── sample           # Sample entropy (Richman-Moorman)
│   └── approximate      # Approximate entropy (Pincus)
├── stats/               # 8 statistical descriptives
├── signal_processing    # NaN removal, quality validation
├── data_loader          # Parallel CSV loading
├── types                # FeatureRow, ExtractionConfig, checkpoint structs
└── utils                # Atomic writes, checkpoint save/load
```

---

## Output Format

Single CSV with 40 feature columns per segment:

```
file_name, segment_id, signal_length, nan_percentage,
dn_histogram_mode_5, ..., co_embed2_dist_tau_d_expfit_meandiff_2,  [22 Catch22]
permutation_entropy, ..., approximate_entropy,                      [10 Entropy]
stat_mean, ..., stat_max                                            [8 Stats]
```

---

## Validation

```bash
# Generate Python reference values
python tests/generate_reference.py

# Run Rust tests against reference
cargo test

# 24 unit tests covering signal processing, stats, entropy
```

Tolerances: Catch22 < 1e-6, entropy < 1e-4, stats < 1e-10 vs Python references.

---

## Context

Built for the Blood Pressure Inference project (DS4400 Final Project + research). Consumed as a git submodule by [Blood-Pressure-Inference-with-BVP](https://github.com/vignankamarthi/Blood-Pressure-Inference-with-BVP). Extracts features from PulseDB v2.0 PPG waveforms (5.2M segments, 125 Hz, 10-second windows) on NEU Explorer cluster (56 CPUs).
