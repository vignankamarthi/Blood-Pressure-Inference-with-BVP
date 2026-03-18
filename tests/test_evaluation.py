"""Tests for src/evaluation.py"""
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation import (
    compute_metrics, aami_compliance, bhs_grading,
    bland_altman, bp_category, evaluate_model
)


def test_compute_metrics_perfect():
    """Perfect predictions should give zero error."""
    y_true = np.array([120.0, 130.0, 140.0])
    y_pred = np.array([120.0, 130.0, 140.0])
    m = compute_metrics(y_true, y_pred)
    assert abs(m["MAE"]) < 1e-10
    assert abs(m["RMSE"]) < 1e-10
    assert abs(m["R2"] - 1.0) < 1e-10
    assert abs(m["ME"]) < 1e-10


def test_compute_metrics_known():
    """Known error values."""
    y_true = np.array([100.0, 110.0, 120.0, 130.0])
    y_pred = np.array([102.0, 108.0, 123.0, 127.0])
    m = compute_metrics(y_true, y_pred)
    assert m["MAE"] == pytest.approx(2.5, abs=1e-10)
    assert m["n_samples"] == 4


def test_aami_pass():
    metrics = {"ME": 2.0, "SD": 5.0}
    result = aami_compliance(metrics)
    assert result["grade"] == "PASS"


def test_aami_fail_me():
    metrics = {"ME": 6.0, "SD": 5.0}
    result = aami_compliance(metrics)
    assert result["grade"] == "FAIL"


def test_aami_fail_sd():
    metrics = {"ME": 2.0, "SD": 9.0}
    result = aami_compliance(metrics)
    assert result["grade"] == "FAIL"


def test_bhs_grade_a():
    """All predictions within 1 mmHg should be grade A."""
    y_true = np.arange(100, 200, dtype=float)
    y_pred = y_true + np.random.uniform(-1, 1, size=len(y_true))
    result = bhs_grading(y_true, y_pred)
    assert result["grade"] == "A"


def test_bhs_grade_d():
    """Large errors should be grade D."""
    y_true = np.arange(100, 200, dtype=float)
    y_pred = y_true + 30.0  # 30 mmHg systematic error
    result = bhs_grading(y_true, y_pred)
    assert result["grade"] == "D"


def test_bland_altman():
    y_true = np.array([100.0, 120.0, 140.0])
    y_pred = np.array([102.0, 118.0, 142.0])
    ba = bland_altman(y_true, y_pred)
    assert "bias" in ba
    assert "lower_limit" in ba
    assert "upper_limit" in ba
    assert ba["lower_limit"] < ba["bias"] < ba["upper_limit"]


def test_bp_category():
    assert bp_category(110, 70) == "normal"
    assert bp_category(125, 75) == "elevated"
    assert bp_category(135, 85) == "stage1"
    assert bp_category(150, 95) == "stage2"


def test_evaluate_model_saves_json():
    with tempfile.TemporaryDirectory() as tmpdir:
        y_true = np.array([120.0, 130.0, 140.0, 150.0])
        y_pred = np.array([122.0, 128.0, 143.0, 147.0])
        result = evaluate_model("ridge", y_true, y_pred, "sbp", Path(tmpdir))
        json_path = Path(tmpdir) / "ridge_sbp_metrics.json"
        assert json_path.exists()
        assert result["model"] == "ridge"
        assert result["target"] == "sbp"
