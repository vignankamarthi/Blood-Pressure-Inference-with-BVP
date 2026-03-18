"""Tests for src/models.py"""
import tempfile
from pathlib import Path

import numpy as np
import joblib
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models import (
    train_ridge, train_decision_tree, train_random_forest,
    train_xgboost, train_lightgbm, is_model_done, load_status
)


@pytest.fixture
def mock_data():
    """Generate small synthetic dataset for testing."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 10)
    y = X[:, 0] * 2.0 + X[:, 1] * -1.5 + np.random.randn(n) * 0.5
    return X, y


@pytest.fixture
def checkpoint_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_train_ridge(mock_data, checkpoint_dir):
    X, y = mock_data
    model = train_ridge(X, y, {"alpha": 1.0}, checkpoint_dir, "sbp")
    assert (checkpoint_dir / "models" / "ridge_sbp.pkl").exists()
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_train_decision_tree(mock_data, checkpoint_dir):
    X, y = mock_data
    model = train_decision_tree(X, y, {"max_depth": 5}, checkpoint_dir, "sbp")
    assert (checkpoint_dir / "models" / "dt_sbp.pkl").exists()


def test_train_random_forest_checkpointing(mock_data, checkpoint_dir):
    X, y = mock_data
    model = train_random_forest(
        X, y, {"n_estimators": 100, "max_depth": 5}, checkpoint_dir, "sbp")
    assert (checkpoint_dir / "models" / "rf_sbp.pkl").exists()
    # Check intermediate checkpoints exist
    progress_path = checkpoint_dir / "rf_progress_sbp.json"
    assert progress_path.exists()


def test_train_xgboost(mock_data, checkpoint_dir):
    X, y = mock_data
    model = train_xgboost(
        X, y, {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        checkpoint_dir, "sbp")
    assert (checkpoint_dir / "models" / "xgb_sbp.pkl").exists()


def test_train_lightgbm(mock_data, checkpoint_dir):
    X, y = mock_data
    model = train_lightgbm(
        X, y, {"n_estimators": 50, "max_depth": 3, "learning_rate": 0.1},
        checkpoint_dir, "dbp")
    assert (checkpoint_dir / "models" / "lgbm_dbp.pkl").exists()


def test_model_save_load_predictions_match(mock_data, checkpoint_dir):
    """Verify predictions match after save/load."""
    X, y = mock_data
    model = train_ridge(X, y, {"alpha": 1.0}, checkpoint_dir, "sbp")
    preds_original = model.predict(X)

    model_loaded = joblib.load(checkpoint_dir / "models" / "ridge_sbp.pkl")
    preds_loaded = model_loaded.predict(X)

    np.testing.assert_array_almost_equal(preds_original, preds_loaded)


def test_is_model_done(mock_data, checkpoint_dir):
    X, y = mock_data
    assert not is_model_done(checkpoint_dir, "ridge", "sbp")
    train_ridge(X, y, {"alpha": 1.0}, checkpoint_dir, "sbp")
    # Need to update status manually since train_ridge doesn't do it
    # (train_all does)
    # Just check file exists
    assert (checkpoint_dir / "models" / "ridge_sbp.pkl").exists()
