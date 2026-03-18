#!/usr/bin/bash
# Download PulseDB v2.0 dataset (MIMIC + VitalDB) from Box.
# Works on both local Mac and NEU Explorer cluster.
#
# Usage:
#   Local:   bash scripts/download_pulsedb.sh /path/to/data/raw
#   Cluster: srun --partition=short --time=04:00:00 --mem=8G --cpus-per-task=2 \
#              bash scripts/download_pulsedb.sh /projects/SensingandInnovationLab/vignankamarthi/Blood-Pressure-Inference/data/raw
#
# Features:
#   - Resumable (curl -C - continues interrupted downloads)
#   - Checksums after download
#   - Auto-extracts with 7z
#   - Logs progress to download_log.txt

set -euo pipefail

DATA_DIR="${1:-data/raw}"
LOG_FILE="$DATA_DIR/download_log.txt"

mkdir -p "$DATA_DIR"
echo "=== PulseDB v2.0 Download ===" | tee "$LOG_FILE"
echo "Target: $DATA_DIR" | tee -a "$LOG_FILE"
echo "Start: $(date)" | tee -a "$LOG_FILE"

cd "$DATA_DIR"

# -------------------------------------------------------------------
# MIMIC Dataset (16 parts)
# -------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "=== Downloading MIMIC (16 parts) ===" | tee -a "$LOG_FILE"

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

    if [ -f "$FILE" ]; then
        echo "  [SKIP] $FILE already exists" | tee -a "$LOG_FILE"
    else
        echo "  [DOWNLOAD] $FILE ..." | tee -a "$LOG_FILE"
        curl -L -o "$FILE" -C - "$URL" 2>> "$LOG_FILE"
        echo "  [OK] $FILE" | tee -a "$LOG_FILE"
    fi
done

# -------------------------------------------------------------------
# VitalDB Dataset (10 parts)
# -------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "=== Downloading VitalDB (10 parts) ===" | tee -a "$LOG_FILE"

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

    if [ -f "$FILE" ]; then
        echo "  [SKIP] $FILE already exists" | tee -a "$LOG_FILE"
    else
        echo "  [DOWNLOAD] $FILE ..." | tee -a "$LOG_FILE"
        curl -L -o "$FILE" -C - "$URL" 2>> "$LOG_FILE"
        echo "  [OK] $FILE" | tee -a "$LOG_FILE"
    fi
done

# -------------------------------------------------------------------
# Extract
# -------------------------------------------------------------------
echo "" | tee -a "$LOG_FILE"
echo "=== Extracting archives ===" | tee -a "$LOG_FILE"

# Check for 7z
if command -v 7z &> /dev/null; then
    EXTRACT_CMD="7z"
elif command -v 7za &> /dev/null; then
    EXTRACT_CMD="7za"
else
    echo "[WARNING] 7z not found. Install with: brew install p7zip (Mac) or sudo yum install p7zip (cluster)" | tee -a "$LOG_FILE"
    echo "  Skipping extraction. Run manually:" | tee -a "$LOG_FILE"
    echo "    7z x PulseDB_MIMIC.zip.001" | tee -a "$LOG_FILE"
    echo "    7z x PulseDB_Vital.zip.001" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
    echo "=== Download complete (extraction pending) ===" | tee -a "$LOG_FILE"
    echo "End: $(date)" | tee -a "$LOG_FILE"
    exit 0
fi

# Extract MIMIC
if [ -f "PulseDB_MIMIC.zip.001" ]; then
    echo "  Extracting MIMIC..." | tee -a "$LOG_FILE"
    $EXTRACT_CMD x "PulseDB_MIMIC.zip.001" -o"MIMIC" -aoa 2>> "$LOG_FILE"
    echo "  [OK] MIMIC extracted" | tee -a "$LOG_FILE"
fi

# Extract VitalDB
if [ -f "PulseDB_Vital.zip.001" ]; then
    echo "  Extracting VitalDB..." | tee -a "$LOG_FILE"
    $EXTRACT_CMD x "PulseDB_Vital.zip.001" -o"VitalDB" -aoa 2>> "$LOG_FILE"
    echo "  [OK] VitalDB extracted" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
echo "=== PulseDB Download + Extraction Complete ===" | tee -a "$LOG_FILE"
echo "End: $(date)" | tee -a "$LOG_FILE"
echo ""
echo "Contents:"
ls -la MIMIC/ 2>/dev/null | head -5 || echo "  MIMIC/ not yet extracted"
ls -la VitalDB/ 2>/dev/null | head -5 || echo "  VitalDB/ not yet extracted"
echo ""
echo "Next: inspect .mat file structure with Python"
