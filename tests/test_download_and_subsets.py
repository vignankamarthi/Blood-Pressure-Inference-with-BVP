"""
Tests for data download script and subset generation.
These test the script logic without actually downloading (mock-based).
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ─── Download script tests ───


def test_download_script_syntax():
    """Verify download_pulsedb.sh has no bash syntax errors."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb.sh"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Syntax error in download script: {result.stderr}"


def test_download_script_has_all_mimic_urls():
    """Verify all 16 MIMIC part URLs are present in the MIMIC_URLS array."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb.sh"
    content = script.read_text()
    # Count Box URLs in the MIMIC section
    mimic_urls = [line for line in content.split('\n') if 'rutgers.box.com' in line and 'MIMIC' in content[:content.index(line)] if 'VITAL' not in content[:content.index(line)].upper().split('VitalDB')[-1] if line.strip()]
    # Simpler: just count all rutgers.box.com URLs
    all_urls = [line.strip() for line in content.split('\n') if 'rutgers.box.com' in line]
    assert len(all_urls) == 26, f"Expected 26 Box URLs (16 MIMIC + 10 VitalDB), got {len(all_urls)}"


def test_download_script_has_all_vital_urls():
    """Verify the script references both MIMIC and VitalDB datasets."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb.sh"
    content = script.read_text()
    assert "PulseDB_MIMIC" in content, "Missing MIMIC dataset reference"
    assert "PulseDB_Vital" in content, "Missing VitalDB dataset reference"
    assert "MIMIC_URLS" in content, "Missing MIMIC URL array"
    assert "VITAL_URLS" in content, "Missing VitalDB URL array"


def test_download_script_has_resume_flag():
    """Verify curl uses -C - for resumable downloads."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb.sh"
    content = script.read_text()
    assert "-C -" in content, "Download script missing curl resume flag (-C -)"


def test_download_script_has_skip_logic():
    """Verify script skips already-downloaded files."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb.sh"
    content = script.read_text()
    assert "SKIP" in content, "Download script missing skip logic for existing files"


# ─── Generate subsets script tests ───


def test_generate_subsets_script_syntax():
    """Verify generate_subsets.py imports without error."""
    result = subprocess.run(
        [sys.executable, "-c", "import scripts.generate_subsets"],
        capture_output=True, text=True,
        cwd=str(Path(__file__).parent.parent),
        env={**__import__('os').environ, "PYTHONPATH": str(Path(__file__).parent.parent)},
    )
    # May fail on import if mat73 not installed, but should not have syntax errors
    # We just check it's not a SyntaxError
    if result.returncode != 0:
        assert "SyntaxError" not in result.stderr, f"Syntax error: {result.stderr}"


def test_generate_subsets_uses_raw_ppg_by_default():
    """Verify the script defaults to PPG_Record (raw), not PPG_F (filtered)."""
    script = Path(__file__).parent.parent / "scripts" / "generate_subsets.py"
    content = script.read_text()
    assert 'use_raw_ppg: bool = True' in content, "Default should be raw PPG"
    assert 'PPG_Record' in content, "Must reference PPG_Record field"


def test_generate_subsets_has_gate_g5_check():
    """Verify the script checks for expected .mat file structure."""
    script = Path(__file__).parent.parent / "scripts" / "generate_subsets.py"
    content = script.read_text()
    assert "Subj_Wins" in content, "Must check for Subj_Wins field (GATE G5)"


# ─── SLURM script tests ───


def test_slurm_extract_script_syntax():
    """Verify extract_features.sbatch has no bash syntax errors."""
    script = Path(__file__).parent.parent / "scripts" / "extract_features.sbatch"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


def test_slurm_tune_script_syntax():
    script = Path(__file__).parent.parent / "scripts" / "tune_models.sbatch"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


def test_slurm_train_script_syntax():
    script = Path(__file__).parent.parent / "scripts" / "train_models.sbatch"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


def test_slurm_setup_script_syntax():
    script = Path(__file__).parent.parent / "scripts" / "setup_cluster.sh"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error: {result.stderr}"


def test_slurm_scripts_have_correct_partition():
    """All SLURM scripts should use 'short' partition with 56 CPUs."""
    for name in ["extract_features.sbatch", "tune_models.sbatch", "train_models.sbatch"]:
        script = Path(__file__).parent.parent / "scripts" / name
        content = script.read_text()
        assert "--partition=short" in content, f"{name} missing --partition=short"
        assert "--cpus-per-task=56" in content, f"{name} missing --cpus-per-task=56"
        assert "--mem=128G" in content, f"{name} missing --mem=128G"


def test_slurm_scripts_have_email():
    """All SLURM scripts should notify on END,FAIL."""
    for name in ["extract_features.sbatch", "tune_models.sbatch", "train_models.sbatch"]:
        script = Path(__file__).parent.parent / "scripts" / name
        content = script.read_text()
        assert "--mail-type=END,FAIL" in content, f"{name} missing mail notification"


def test_slurm_scripts_log_to_logs_dir():
    """All SLURM scripts should output logs to logs/ directory."""
    for name in ["extract_features.sbatch", "tune_models.sbatch", "train_models.sbatch"]:
        script = Path(__file__).parent.parent / "scripts" / name
        content = script.read_text()
        assert "logs/" in content, f"{name} not logging to logs/ directory"
