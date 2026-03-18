#!/bin/bash
#SBATCH --job-name=bp-download
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=kamarthi.v@northeastern.edu

# Download PulseDB v2.0 (MIMIC + VitalDB) directly to cluster storage.
# Resumable: re-submit if wall time runs out, completed parts are skipped.

set -euo pipefail

PROJECT_DIR="/projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference"
DATA_DIR="$PROJECT_DIR/data/raw"

mkdir -p "$DATA_DIR/Info_Files" "$PROJECT_DIR/logs"
cd "$DATA_DIR"

echo "=== PulseDB Download on Explorer ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Target: $DATA_DIR"
echo "Start: $(date)"

# -------------------------------------------------------------------
# Info files (from Google Drive via gdown)
# -------------------------------------------------------------------
echo ""
echo "=== Downloading Info Files ==="

# Activate venv for gdown
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
    pip install gdown -q 2>/dev/null
    cd "$DATA_DIR/Info_Files"
    gdown --folder "https://drive.google.com/drive/folders/1-0IAdIHT9AXpQ8I1Z-xS5PqB3Yeo-bNY" 2>&1 || echo "gdown failed -- download Info files manually"
    # Flatten if nested
    if [ -d "Info_Files" ]; then
        mv Info_Files/* . 2>/dev/null
        rmdir Info_Files 2>/dev/null
    fi
    cd "$DATA_DIR"
    echo "[OK] Info files"
fi

# -------------------------------------------------------------------
# MIMIC Dataset (16 parts)
# -------------------------------------------------------------------
echo ""
echo "=== Downloading MIMIC (16 parts) ==="

MIMIC_URLS=(
    "https://rutgers.box.com/shared/static/7l8n3tn9tr0602tdss1x7e3uliahlibp.001"
    "https://rutgers.box.com/shared/static/zco48rvz5dog72970679foen6hct15c8.002"
    "https://rutgers.box.com/shared/static/x22qpmelx6sz3wgkm5qyc0eis429361f.003"
    "https://rutgers.box.com/shared/static/xj25sqnluiz6s4z8tzzm5phk00ohp6e8.004"
    "https://rutgers.box.com/shared/static/dxus2lsoop02chaspnwipwrf0g4wmenr.005"
    "https://rutgers.box.com/shared/static/rts6sj441laenm2sy1qcemg7ke4om3j6.006"
    "https://rutgers.box.com/shared/static/vor4hjllld7a0c3nzef8uptbb4ut3koo.007"
    "https://rutgers.box.com/shared/static/a2qg2p4ebyrooji3z88djlokji65tlf3.008"
    "https://rutgers.box.com/shared/static/uh6kbiuqgnib5wakiv6o35gkpusyamc7.009"
    "https://rutgers.box.com/shared/static/h6eyhkkx48pf3ce3th1clwj43hn98j5c.010"
    "https://rutgers.box.com/shared/static/e93dp94hxpkas45yc59n289s2wvkafgi.011"
    "https://rutgers.box.com/shared/static/iuvyuw7dmlxvbjvt53dj49wqn3gelqni.012"
    "https://rutgers.box.com/shared/static/qxx6tjz8c3778601ib3icu6o1rranmc7.013"
    "https://rutgers.box.com/shared/static/ip2ninwqj8437l9fyffjprnk90ptnx9k.014"
    "https://rutgers.box.com/shared/static/yrtbo0lg8mjhaw624iw9bbhk1obbocwd.015"
    "https://rutgers.box.com/shared/static/wmzndowgfa5xi3tvtqahxkld3ngdyjds.016"
)

for i in "${!MIMIC_URLS[@]}"; do
    PART=$(printf "%03d" $((i + 1)))
    FILE="PulseDB_MIMIC.zip.$PART"
    URL="${MIMIC_URLS[$i]}"

    if [ -f "$FILE" ] && [ "$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE" 2>/dev/null)" -gt 1000000000 ]; then
        echo "  [SKIP] $FILE ($(du -h "$FILE" | cut -f1))"
    else
        echo "  [DOWNLOAD] $FILE ..."
        curl -L -o "$FILE" -C - "$URL"
        echo "  [OK] $FILE ($(du -h "$FILE" | cut -f1))"
    fi
done

# -------------------------------------------------------------------
# VitalDB Dataset (10 parts)
# -------------------------------------------------------------------
echo ""
echo "=== Downloading VitalDB (10 parts) ==="

VITAL_URLS=(
    "https://rutgers.box.com/shared/static/vtxoksmn7emeaxypb2prywgwscuefoqa.001"
    "https://rutgers.box.com/shared/static/euzkek7c3xoy62jisheuxqar7z5y8xig.002"
    "https://rutgers.box.com/shared/static/49lngo0benxfjw193jnqz9tctlyb3qam.003"
    "https://rutgers.box.com/shared/static/jf4fwgkmhry20mf5tcg9t0wxvky64um0.004"
    "https://rutgers.box.com/shared/static/2lgxysbskfuapsaan4jypvmm8316fdkc.005"
    "https://rutgers.box.com/shared/static/x27ktb4qsx43razwo4tjmxq9v1ro0x3y.006"
    "https://rutgers.box.com/shared/static/q0t36fikgf3pimhvnerwwnovfr0umtp8.007"
    "https://rutgers.box.com/shared/static/ihckx2g0f981g5yz2x8v5rgwndl6yebw.008"
    "https://rutgers.box.com/shared/static/y8j14h8tvi5b3du8nap9dnura1omfrk6.009"
    "https://rutgers.box.com/shared/static/fu0m9tx33jkxywq32shh0g8dg3not15u.010"
)

for i in "${!VITAL_URLS[@]}"; do
    PART=$(printf "%03d" $((i + 1)))
    FILE="PulseDB_Vital.zip.$PART"
    URL="${VITAL_URLS[$i]}"

    if [ -f "$FILE" ] && [ "$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE" 2>/dev/null)" -gt 1000000000 ]; then
        echo "  [SKIP] $FILE ($(du -h "$FILE" | cut -f1))"
    else
        echo "  [DOWNLOAD] $FILE ..."
        curl -L -o "$FILE" -C - "$URL"
        echo "  [OK] $FILE ($(du -h "$FILE" | cut -f1))"
    fi
done

# -------------------------------------------------------------------
# Extract (if 7z available)
# -------------------------------------------------------------------
echo ""
echo "=== Extracting ==="

if command -v 7z &> /dev/null; then
    if [ -f "PulseDB_MIMIC.zip.001" ]; then
        echo "  Extracting MIMIC..."
        7z x "PulseDB_MIMIC.zip.001" -o"MIMIC" -aoa
        echo "  [OK] MIMIC extracted"
    fi
    if [ -f "PulseDB_Vital.zip.001" ]; then
        echo "  Extracting VitalDB..."
        7z x "PulseDB_Vital.zip.001" -o"VitalDB" -aoa
        echo "  [OK] VitalDB extracted"
    fi
else
    echo "  7z not found. Install: sudo yum install p7zip"
    echo "  Then run manually:"
    echo "    cd $DATA_DIR"
    echo "    7z x PulseDB_MIMIC.zip.001 -oMIMIC"
    echo "    7z x PulseDB_Vital.zip.001 -oVitalDB"
fi

echo ""
echo "=== Download Complete ==="
echo "End: $(date)"
echo ""
echo "Disk usage:"
du -sh "$DATA_DIR"
echo ""
echo "Next: python scripts/generate_subsets.py --data-dir data/raw --output-dir data/subsets"
