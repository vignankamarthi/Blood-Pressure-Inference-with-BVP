"""
Tests for data download script and subset generation.
Tests the script logic without actual data (mock-based).
Parser tests use mock dicts matching the exact PulseDB Info .mat structure
discovered via inspect_info.sbatch (dict-of-lists, nested [value] wrapping).
"""
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

# ─── Download script tests ───


def test_download_sbatch_syntax():
    """Verify download_pulsedb_cluster.sbatch has no bash syntax errors."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb_cluster.sbatch"
    result = subprocess.run(
        ["bash", "-n", str(script)],
        capture_output=True, text=True
    )
    assert result.returncode == 0, f"Syntax error in download script: {result.stderr}"


def test_download_sbatch_has_slurm_directives():
    """Verify the cluster download script has required SLURM directives."""
    script = Path(__file__).parent.parent / "scripts" / "download_pulsedb_cluster.sbatch"
    content = script.read_text()
    assert "#SBATCH" in content, "Missing SLURM directives"
    assert "--mail-type" in content, "Missing mail notification"


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


# ─── Info file parser tests (mock data matching discovered PulseDB format) ───


def _make_mock_info_data(n=3):
    """Build a mock Info dict matching exact PulseDB structure: dict-of-lists, nested [value]."""
    return {
        'AAMI_Test_Subset': {
            'Subj_Name': [['p072634_0'], ['p091470_0'], ['p085541_1']][:n],
            'Subj_SegIDX': [[np.array(1.)], [np.array(5.)], [np.array(157.)]][:n],
            'Source': [['MIMIC'], ['MIMIC'], ['VitalDB']][:n],
            'Seg_SBP': [[np.array(96.01)], [np.array(77.14)], [np.array(120.5)]][:n],
            'Seg_DBP': [[np.array(61.22)], [np.array(58.03)], [np.array(80.1)]][:n],
            'Subj_Age': [[np.array(61.)], [np.array(70.)], [np.array(56.)]][:n],
            'Subj_Gender': [['M'], ['M'], ['F']][:n],
            'Subj_BMI': [[np.array(np.nan)], [np.array(np.nan)], [np.array(25.3)]][:n],
            'Subj_Height': [[np.array(np.nan)], [np.array(np.nan)], [np.array(165.)]][:n],
            'Subj_Weight': [[np.array(np.nan)], [np.array(np.nan)], [np.array(68.)]][:n],
        }
    }


@patch('scripts.generate_subsets.load_mat_file')
def test_load_info_file_parses_all_records(mock_load):
    """Parser returns correct number of records from mock Info data."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.generate_subsets import load_info_file
    import scripts.generate_subsets as gs
    import logging
    gs.logger = logging.getLogger("test")

    mock_load.return_value = _make_mock_info_data(3)
    records = load_info_file(Path("fake.mat"))
    assert len(records) == 3


@patch('scripts.generate_subsets.load_mat_file')
def test_load_info_file_unwraps_strings(mock_load):
    """Parser correctly unwraps nested [str] -> str."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.generate_subsets import load_info_file
    import scripts.generate_subsets as gs
    import logging
    gs.logger = logging.getLogger("test")

    mock_load.return_value = _make_mock_info_data(3)
    records = load_info_file(Path("fake.mat"))
    assert records[0]['Subj_Name'] == 'p072634_0'
    assert records[2]['Subj_Name'] == 'p085541_1'
    assert records[0]['Source'] == 'MIMIC'
    assert records[2]['Source'] == 'VitalDB'
    assert records[0]['Subj_Gender'] == 'M'
    assert records[2]['Subj_Gender'] == 'F'


@patch('scripts.generate_subsets.load_mat_file')
def test_load_info_file_unwraps_numerics(mock_load):
    """Parser correctly unwraps nested [0-d ndarray] -> int/float."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.generate_subsets import load_info_file
    import scripts.generate_subsets as gs
    import logging
    gs.logger = logging.getLogger("test")

    mock_load.return_value = _make_mock_info_data(3)
    records = load_info_file(Path("fake.mat"))
    assert records[0]['Subj_SegIDX'] == 1
    assert records[1]['Subj_SegIDX'] == 5
    assert records[2]['Subj_SegIDX'] == 157
    assert isinstance(records[0]['Subj_SegIDX'], int)
    assert abs(records[0]['Seg_SBP'] - 96.01) < 0.001
    assert abs(records[0]['Seg_DBP'] - 61.22) < 0.001
    assert abs(records[1]['Subj_Age'] - 70.0) < 0.001


@patch('scripts.generate_subsets.load_mat_file')
def test_load_info_file_raises_on_missing_required_field(mock_load):
    """Parser raises ValueError if Subj_Name or Subj_SegIDX is missing."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.generate_subsets import load_info_file
    import scripts.generate_subsets as gs
    import logging
    gs.logger = logging.getLogger("test")

    mock_load.return_value = {'Test_Subset': {'Subj_Name': [['p001_0']]}}
    with pytest.raises(ValueError, match="Missing required field"):
        load_info_file(Path("fake.mat"))


@patch('scripts.generate_subsets.load_mat_file')
def test_load_info_file_handles_optional_fields(mock_load):
    """Parser works when optional fields (Source, Seg_SBP, etc.) are absent."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from scripts.generate_subsets import load_info_file
    import scripts.generate_subsets as gs
    import logging
    gs.logger = logging.getLogger("test")

    mock_load.return_value = {
        'Minimal_Subset': {
            'Subj_Name': [['p001_0'], ['p002_1']],
            'Subj_SegIDX': [[np.array(1.)], [np.array(3.)]],
        }
    }
    records = load_info_file(Path("fake.mat"))
    assert len(records) == 2
    assert 'Source' not in records[0]
    assert 'Seg_SBP' not in records[0]


def test_subj_name_split_extracts_correct_id():
    """Verify p{ID}_{case} -> p{ID} mapping works for variable-length IDs."""
    cases = [
        ("p072634_0", "p072634"),
        ("p000160_0", "p000160"),
        ("p001_1", "p001"),
        ("p12345678_0", "p12345678"),
    ]
    for subj_name, expected_id in cases:
        assert subj_name.split('_')[0] == expected_id


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
