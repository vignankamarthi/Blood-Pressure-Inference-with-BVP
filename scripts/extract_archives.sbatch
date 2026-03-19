#!/usr/bin/bash
#SBATCH --job-name=bp-extract-7z
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Extract PulseDB multi-part archives.
# Resumable: re-run safely -- skips already-extracted files (-aos).
# Logs to logs/ for cluster-to-local transfer.

set -uo pipefail
# NOTE: not using -e (exit on error) because 7zz piped through tee
# can return non-zero even on success, killing the script prematurely

export PATH="$HOME/bin:$PATH"
PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
DATA_DIR="$PROJECT_DIR/data/raw"
LOG="$PROJECT_DIR/logs/extract_progress.log"

cd "$DATA_DIR"
mkdir -p "$PROJECT_DIR/logs"

echo "=== PulseDB Archive Extraction ===" | tee "$LOG"
echo "Job ID: $SLURM_JOB_ID" | tee -a "$LOG"
echo "Start: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Check 7zz exists
if ! command -v 7zz &> /dev/null; then
    echo "ERROR: 7zz not found. Install to ~/bin/ first." | tee -a "$LOG"
    exit 1
fi

# --- MIMIC ---
if [ -f "PulseDB_MIMIC.zip.001" ]; then
    MIMIC_COUNT_BEFORE=$(ls MIMIC/PulseDB_MIMIC/ 2>/dev/null | wc -l)
    echo "Extracting MIMIC (existing files: $MIMIC_COUNT_BEFORE)..." | tee -a "$LOG"
    echo "  Using -aos (skip existing files for resume)" | tee -a "$LOG"

    7zz x PulseDB_MIMIC.zip.001 -oMIMIC -aos -y 2>&1 | tee -a "$LOG"

    MIMIC_COUNT_AFTER=$(ls MIMIC/PulseDB_MIMIC/ 2>/dev/null | wc -l)
    echo "MIMIC done: $MIMIC_COUNT_AFTER files" | tee -a "$LOG"
else
    echo "SKIP: PulseDB_MIMIC.zip.001 not found" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"

# --- VitalDB ---
if [ -f "PulseDB_Vital.zip.001" ]; then
    VITAL_COUNT_BEFORE=$(ls VitalDB/PulseDB_Vital/ 2>/dev/null | wc -l)
    echo "Extracting VitalDB (existing files: $VITAL_COUNT_BEFORE)..." | tee -a "$LOG"
    echo "  Using -aos (skip existing files for resume)" | tee -a "$LOG"

    7zz x PulseDB_Vital.zip.001 -oVitalDB -aos -y 2>&1 | tee -a "$LOG"

    VITAL_COUNT_AFTER=$(ls VitalDB/PulseDB_Vital/ 2>/dev/null | wc -l)
    echo "VitalDB done: $VITAL_COUNT_AFTER files" | tee -a "$LOG"
else
    echo "SKIP: PulseDB_Vital.zip.001 not found" | tee -a "$LOG"
fi

echo "" | tee -a "$LOG"

# --- Flatten Info_Files if nested ---
if [ -d "Info_Files/Info_Files" ]; then
    echo "Flattening nested Info_Files..." | tee -a "$LOG"
    mv Info_Files/Info_Files/* Info_Files/ 2>/dev/null
    rmdir Info_Files/Info_Files 2>/dev/null
    echo "Info_Files flattened" | tee -a "$LOG"
fi

# --- Summary ---
echo "" | tee -a "$LOG"
echo "=== Extraction Complete ===" | tee -a "$LOG"
echo "End: $(date)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "File counts:" | tee -a "$LOG"
echo "  MIMIC: $(ls MIMIC/PulseDB_MIMIC/ 2>/dev/null | wc -l) files" | tee -a "$LOG"
echo "  VitalDB: $(ls VitalDB/PulseDB_Vital/ 2>/dev/null | wc -l) files" | tee -a "$LOG"
echo "  Info: $(ls Info_Files/*.mat 2>/dev/null | wc -l) files" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Disk usage:" | tee -a "$LOG"
du -sh "$DATA_DIR" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Next: python scripts/generate_subsets.py --data-dir data/raw --output-dir data/subsets" | tee -a "$LOG"
