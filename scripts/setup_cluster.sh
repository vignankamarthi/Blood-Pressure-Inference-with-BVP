#!/bin/bash
# One-time cluster setup for Blood Pressure Inference pipeline.
# Run from: /projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference/

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
SCRATCH_DIR="/scratch/kamarthi.v/bp-checkpoints"

echo "=== Blood Pressure Inference -- Cluster Setup ==="
echo "Project: $PROJECT_DIR"
echo "Scratch: $SCRATCH_DIR"

# 1. Create directories
mkdir -p "$PROJECT_DIR"/{data/raw,data/features,logs,results}
mkdir -p "$SCRATCH_DIR"/{models,optuna}
echo "[OK] Directories created"

# 2. Install Rust toolchain (if not already installed)
if ! command -v rustup &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    echo "[OK] Rust installed"
else
    echo "[OK] Rust already installed: $(rustc --version)"
fi

# 3. Build Rust binary (release mode)
echo "Building Rust feature extraction binary..."
cd "$PROJECT_DIR/Feature-Extraction-Rust-Complete"
cargo build --release
echo "[OK] Rust binary compiled: target/release/feature_extraction_rust_complete"

# 4. Create Python venv
echo "Creating Python virtual environment..."
cd "$PROJECT_DIR"
if [ ! -d ".venv" ]; then
    ~/python/bin/python3.11 -m venv .venv
    echo "[OK] venv created"
else
    echo "[OK] venv already exists"
fi

# 5. Install Python dependencies (via srun on compute node)
echo "Installing Python packages (on compute node)..."
srun --partition=short --time=00:15:00 --mem=4G --cpus-per-task=2 \
    bash -c "cd $PROJECT_DIR && source .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"
echo "[OK] Python packages installed"

# 6. Verify installation
echo "Verifying installation..."
srun --partition=short --time=00:05:00 --mem=2G --cpus-per-task=1 \
    bash -c "cd $PROJECT_DIR && source .venv/bin/activate && python -c 'import sklearn, xgboost, lightgbm, optuna; print(\"All ML packages OK\")'"

# 7. Test Rust binary
echo "Testing Rust binary..."
"$PROJECT_DIR/Feature-Extraction-Rust-Complete/target/release/feature_extraction_rust_complete" --help
echo "[OK] Rust binary works"

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Upload PulseDB data to $PROJECT_DIR/data/raw/"
echo "  2. Run: sbatch scripts/extract_features.sbatch"
echo "  3. Run: sbatch scripts/tune_models.sbatch"
echo "  4. Run: sbatch scripts/train_models.sbatch"
