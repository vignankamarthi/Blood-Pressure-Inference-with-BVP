#!/usr/bin/bash
#SBATCH --job-name=bp-subsets
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Generate train/test subsets from PulseDB segment + info files.
# Extracts PPG_Record, ECG_F, ABP_Raw (all 3 signals per segment).

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== Subset Generation ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python scripts/generate_subsets.py \
    --data-dir data/raw \
    --output-dir data/subsets

echo ""
echo "End: $(date)"
