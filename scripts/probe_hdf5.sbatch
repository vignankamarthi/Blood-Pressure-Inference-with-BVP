#!/usr/bin/bash
#SBATCH --job-name=bp-probe-hdf5
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Probe HDF5 structure of PulseDB .mat files using h5py.
# Output tells us how to dereference object references for targeted reads.
# Read logs/bp-probe-hdf5_*.log for results.

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
cd "$PROJECT_DIR"
source .venv/bin/activate

export DATA_DIR="$PROJECT_DIR/data/raw"

echo "=== HDF5 Structure Probe ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"
echo ""

python scripts/probe_hdf5_helper.py

echo ""
echo "End: $(date)"
