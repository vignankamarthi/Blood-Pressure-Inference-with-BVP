"""Tests for ablation config system."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ablation_configs_exist():
    config_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
    assert config_path.exists(), "ablation_configs.json not found"


def test_ablation_configs_has_2_configs():
    # ABP configs dropped 2026-04-05 (feature-source leakage audit).
    config_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
    with open(config_path) as f:
        configs = json.load(f)
    assert len(configs) == 2, f"Expected 2 configs, got {len(configs)}"


def test_ablation_config_names():
    config_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
    with open(config_path) as f:
        configs = json.load(f)
    expected = {"ppg", "ppg_ecg"}
    assert set(configs.keys()) == expected


def test_ablation_config_feature_counts():
    config_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
    with open(config_path) as f:
        configs = json.load(f)

    for name, cfg in configs.items():
        n_signals = len(cfg["signals"])
        expected_features = n_signals * 40
        assert cfg["n_features"] == expected_features, \
            f"{name}: expected {expected_features} features ({n_signals} signals x 40), got {cfg['n_features']}"


def test_ablation_config_dl_channels():
    config_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
    with open(config_path) as f:
        configs = json.load(f)

    for name, cfg in configs.items():
        assert cfg["dl_channels"] == len(cfg["signals"]), \
            f"{name}: dl_channels should equal number of signals"


def test_ppg_is_in_all_configs():
    """PPG must be present in every config (it's the primary signal)."""
    config_path = Path(__file__).parent.parent / "configs" / "ablation_configs.json"
    with open(config_path) as f:
        configs = json.load(f)

    for name, cfg in configs.items():
        assert "ppg" in cfg["signals"], f"{name} is missing 'ppg' signal"


def test_generate_subsets_extracts_all_signals():
    """Verify generate_subsets.py references all 3 signal fields."""
    script = Path(__file__).parent.parent / "scripts" / "generate_subsets.py"
    content = script.read_text()
    assert "ECG_F" in content, "generate_subsets.py missing ECG_F extraction"
    assert "ABP_Raw" in content, "generate_subsets.py missing ABP_Raw extraction"
    assert "PPG_Record" in content, "generate_subsets.py missing PPG_Record extraction"


def test_generate_subsets_saves_all_signal_keys():
    """Verify .npz output includes all 3 signal arrays."""
    script = Path(__file__).parent.parent / "scripts" / "generate_subsets.py"
    content = script.read_text()
    assert "ecg_signals" in content, "generate_subsets.py missing ecg_signals in .npz output"
    assert "abp_signals" in content, "generate_subsets.py missing abp_signals in .npz output"
    assert "ppg_signals" in content, "generate_subsets.py missing ppg_signals in .npz output"


def test_train_models_has_config_flag():
    """Verify train_models.py accepts --config argument."""
    script = Path(__file__).parent.parent / "scripts" / "train_models.py"
    content = script.read_text()
    assert "--config" in content, "train_models.py missing --config argument"
    assert "ppg_ecg" in content, "train_models.py missing ppg_ecg config option"


def test_slurm_scripts_accept_config():
    """Verify SLURM scripts use CONFIG env var."""
    for name in ["train_models.sbatch", "tune_models.sbatch"]:
        script = Path(__file__).parent.parent / "scripts" / name
        content = script.read_text()
        assert "CONFIG" in content, f"{name} missing CONFIG env var"


def test_extract_features_runs_3_signals():
    """Verify extraction script runs Rust 3 times (ppg, ecg, abp)."""
    script = Path(__file__).parent.parent / "scripts" / "extract_features.sbatch"
    content = script.read_text()
    assert "for SIGNAL in ppg ecg abp" in content, \
        "extract_features.sbatch missing 3-signal loop"
