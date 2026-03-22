#!/usr/bin/bash
#SBATCH --job-name=bp-inspect
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# GATE G5: Inspect PulseDB .mat file structure.
# Check logs/ output to verify expected fields exist before proceeding.

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

export DATA_DIR="$PROJECT_DIR/data/raw"

echo "=== PulseDB .mat Inspection (GATE G5) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

python scripts/inspect_mat_helper.py

echo ""
echo "End: $(date)"
echo "Check output above for GATE G5 PASS/FAIL"
