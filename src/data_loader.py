"""Data loading for PulseDB v2.0 and Rust-extracted feature CSVs."""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

from .utils import atomic_json_write, load_json, timer

logger = logging.getLogger("bp_pipeline")


def load_feature_csv(path: Path) -> pd.DataFrame:
    """Load a Rust-extracted feature CSV."""
    logger.info(f"Loading features from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Columns: {list(df.columns)}")
    return df


def load_cached_features(
    feature_path: Path,
    cache_path: Path,
) -> pd.DataFrame:
    """Load features with parquet caching for fast reload."""
    if cache_path.exists():
        logger.info(f"Loading cached features from {cache_path}")
        return pd.read_parquet(cache_path)

    df = load_feature_csv(feature_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    logger.info(f"Cached features to {cache_path}")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get the 41 feature column names (excluding metadata)."""
    metadata_cols = {"file_name", "segment_id", "signal_length", "nan_percentage",
                     "entropy_dimension", "entropy_tau", "subject_id",
                     "sbp", "dbp", "split"}
    return [c for c in df.columns if c not in metadata_cols]


def prepare_train_test(
    df: pd.DataFrame,
    target: str,
    checkpoint_dir: Path,
    force_refit: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Prepare train/test splits with StandardScaler.

    Per-column StandardScaler fitted on training subjects ONLY, transform all.
    This is the LOCKED normalization strategy.

    Args:
        df: Feature DataFrame with 'split' column ('train'/'test')
        target: 'sbp' or 'dbp'
        checkpoint_dir: Directory for saving scaler
        force_refit: If True, refit scaler even if cached

    Returns:
        X_train, y_train, X_test, y_test, scaler
    """
    feature_cols = get_feature_columns(df)
    logger.info(f"Using {len(feature_cols)} feature columns for target={target}")

    # Split
    train_mask = df["split"] == "train"
    test_mask = df["split"] == "test"

    X_train_raw = df.loc[train_mask, feature_cols].values.astype(np.float64)
    y_train = df.loc[train_mask, target].values.astype(np.float64)
    X_test_raw = df.loc[test_mask, feature_cols].values.astype(np.float64)
    y_test = df.loc[test_mask, target].values.astype(np.float64)

    logger.info(f"  Train: {X_train_raw.shape}, Test: {X_test_raw.shape}")

    # StandardScaler -- PER COLUMN, fit on training ONLY
    scaler_path = checkpoint_dir / f"scaler_{target}.pkl"
    if scaler_path.exists() and not force_refit:
        logger.info(f"  Loading cached scaler from {scaler_path}")
        scaler = joblib.load(scaler_path)
    else:
        logger.info("  Fitting StandardScaler on training data (per column)")
        scaler = StandardScaler()
        scaler.fit(X_train_raw)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)
        logger.info(f"  Saved scaler to {scaler_path}")

    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Handle NaN/inf from scaling
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, y_train, X_test, y_test, scaler


def load_pulsedb_mat(mat_dir: Path) -> pd.DataFrame:
    """
    Load PulseDB v2.0 .mat files.
    Extracts raw PPG segments + SBP/DBP labels.

    GATE G5: If .mat structure differs from docs, STOP.
    """
    try:
        import mat73
    except ImportError:
        raise ImportError("mat73 required for loading PulseDB .mat files. pip install mat73")

    logger.info(f"Loading PulseDB from {mat_dir}")
    records = []

    mat_files = sorted(mat_dir.glob("*.mat"))
    logger.info(f"  Found {len(mat_files)} .mat files")

    for mat_file in mat_files:
        try:
            data = mat73.loadmat(str(mat_file))
            # Expected fields: PPG_Record, SBP, DBP, subject_id
            # GATE G5: verify structure
            if "PPG_Record" not in data and "ppg" not in data:
                logger.warning(f"  GATE G5: Unexpected .mat structure in {mat_file.name}: {list(data.keys())}")
                continue

            ppg_key = "PPG_Record" if "PPG_Record" in data else "ppg"
            ppg = np.array(data[ppg_key])

            if ppg.ndim == 1:
                ppg = ppg.reshape(-1, 1)

            n_segments = ppg.shape[1] if ppg.ndim > 1 else 1

            for seg_idx in range(n_segments):
                segment = ppg[:, seg_idx] if ppg.ndim > 1 else ppg
                records.append({
                    "file_name": mat_file.stem,
                    "segment_id": f"{mat_file.stem}_seg{seg_idx:04d}",
                    "signal": segment.tolist(),
                    "signal_length": len(segment),
                })
        except Exception as e:
            logger.error(f"  Failed to load {mat_file.name}: {e}")

    logger.info(f"  Loaded {len(records)} segments")
    return pd.DataFrame(records)
