#!/usr/bin/bash
#SBATCH --job-name=bp-merge
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Merge Rust feature CSVs with .npz labels for all ablation configs.

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

echo "=== Merge Features + Labels ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python scripts/merge_features_labels.py \
    --feature-dir data/features \
    --subset-dir data/subsets \
    --config-path configs/ablation_configs.json

echo ""
echo "End: $(date)"
