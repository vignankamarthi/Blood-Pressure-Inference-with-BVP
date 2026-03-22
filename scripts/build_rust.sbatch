#!/usr/bin/bash
#SBATCH --job-name=bp-build-rust
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Compile Rust feature extraction binary on cluster.
# GATE G8: if compilation fails, STOP.

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"

echo "=== Rust Build ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

# Ensure Rust toolchain exists
if ! command -v rustup &> /dev/null; then
    echo "Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

source "$HOME/.cargo/env" 2>/dev/null

echo "Rust version: $(rustc --version)"
echo "Cargo version: $(cargo --version)"

cd Feature-Extraction-Rust-Complete
cargo build --release 2>&1

if [ $? -eq 0 ]; then
    echo ""
    echo "BUILD SUCCESS"
    echo "Binary: $(pwd)/target/release/feature_extraction_rust_complete"
    ls -lh target/release/feature_extraction_rust_complete
else
    echo ""
    echo "BUILD FAILED -- GATE G8 TRIGGERED"
    echo "Review error output above"
    exit 1
fi

echo ""
echo "End: $(date)"
