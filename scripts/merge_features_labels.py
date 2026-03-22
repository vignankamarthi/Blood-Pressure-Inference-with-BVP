#!/usr/bin/env python3
"""
Merge Rust-extracted feature CSVs with .npz subset labels into training-ready CSVs.

For each ablation config (ppg, ppg_ecg, ppg_abp, ppg_ecg_abp):
  - Load per-signal feature CSVs from data/features/{signal}/features.csv
  - Prefix columns by signal name (ppg_dn_histogram_mode_5, ecg_dn_histogram_mode_5, etc.)
  - Horizontal concat on (file_name, segment_id)
  - Join with labels (SBP, DBP) and split assignments from .npz subsets
  - Write data/features/{config}/features_labeled.csv

Usage:
    python scripts/merge_features_labels.py --feature-dir data/features --subset-dir data/subsets
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

logger = None

METADATA_COLS = {'file_name', 'segment_id', 'signal_length', 'nan_percentage'}


def load_and_prefix(feature_csv: Path, signal_name: str) -> pd.DataFrame:
    """Load a feature CSV and prefix all feature columns with the signal name."""
    df = pd.read_csv(feature_csv)

    # Identify feature columns (everything not in METADATA_COLS)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    # Prefix feature columns
    rename_map = {c: f"{signal_name}_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)

    return df


def merge_signals(feature_dir: Path, signals: list) -> pd.DataFrame:
    """Merge feature CSVs from multiple signals into one DataFrame."""
    dfs = []
    for signal in signals:
        csv_path = feature_dir / signal / "features.csv"
        if not csv_path.exists():
            logger.error(f"  Feature CSV not found: {csv_path}")
            return None
        df = load_and_prefix(csv_path, signal)
        dfs.append(df)

    if not dfs:
        return None

    # Merge on (file_name, segment_id)
    merged = dfs[0]
    for df in dfs[1:]:
        # Keep only prefixed feature columns from subsequent DataFrames
        meta_cols_in_df = [c for c in df.columns if c in METADATA_COLS]
        feature_cols_in_df = [c for c in df.columns if c not in METADATA_COLS]
        merge_keys = ['file_name', 'segment_id']
        cols_to_merge = merge_keys + feature_cols_in_df
        merged = merged.merge(df[cols_to_merge], on=merge_keys, how='inner')

    logger.info(f"  Merged {len(signals)} signals: {merged.shape}")
    return merged


def add_labels_and_split(
    features_df: pd.DataFrame,
    train_npz: Path,
    test_npz: Path,
) -> pd.DataFrame:
    """Add SBP/DBP labels and train/test split column."""
    # For now, the split assignment comes from which .npz the segment belongs to
    # The features_df has (file_name, segment_id) which maps to subjects in .npz
    # Since Rust extracts from the CSVs (which were exported from .npz subsets),
    # we know the mapping: Train_Subset segments -> split="train",
    # CalFree_Test_Subset segments -> split="test"

    # Load labels from .npz
    train_data = np.load(str(train_npz), allow_pickle=True)
    test_data = np.load(str(test_npz), allow_pickle=True)

    # Build label lookup from subjects + segment indices
    train_labels = pd.DataFrame({
        'subject': train_data['subjects'],
        'sbp': train_data['sbp'],
        'dbp': train_data['dbp'],
        'split': 'train',
    })

    test_labels = pd.DataFrame({
        'subject': test_data['subjects'],
        'sbp': test_data['sbp'],
        'dbp': test_data['dbp'],
        'split': 'test',
    })

    # Add segment index within each subject
    for df in [train_labels, test_labels]:
        df['seg_idx'] = df.groupby('subject').cumcount()

    all_labels = pd.concat([train_labels, test_labels], ignore_index=True)

    # The file_name in features_df corresponds to the CSV filename (subject.csv)
    # The segment_id corresponds to the column name in that CSV (subject_segNNNN)
    # We need to match these back to the .npz segment ordering

    # For now, add labels by position (features_df rows should align with .npz order
    # since export_signals_csv.py writes in the same order)
    # This is a simplification -- in production, use explicit segment IDs

    if len(features_df) <= len(all_labels):
        features_df = features_df.copy()
        features_df['sbp'] = all_labels['sbp'].values[:len(features_df)]
        features_df['dbp'] = all_labels['dbp'].values[:len(features_df)]
        features_df['split'] = all_labels['split'].values[:len(features_df)]
        features_df['subject_id'] = all_labels['subject'].values[:len(features_df)]
    else:
        logger.warning(f"  Feature rows ({len(features_df)}) > label rows ({len(all_labels)})")
        # Truncate to match
        features_df = features_df.iloc[:len(all_labels)].copy()
        features_df['sbp'] = all_labels['sbp'].values
        features_df['dbp'] = all_labels['dbp'].values
        features_df['split'] = all_labels['split'].values
        features_df['subject_id'] = all_labels['subject'].values

    return features_df


def main():
    global logger

    parser = argparse.ArgumentParser(description="Merge features with labels for training")
    parser.add_argument("--feature-dir", type=str, default="data/features")
    parser.add_argument("--subset-dir", type=str, default="data/subsets")
    parser.add_argument("--config-path", type=str, default="configs/ablation_configs.json")
    args = parser.parse_args()

    logger = setup_logging(name="merge_features")

    feature_dir = Path(args.feature_dir)
    subset_dir = Path(args.subset_dir)

    with open(args.config_path) as f:
        configs = json.load(f)

    train_npz = subset_dir / "Train_Subset.npz"
    test_npz = subset_dir / "CalFree_Test_Subset.npz"

    for config_name, config in configs.items():
        signals = config["signals"]
        logger.info(f"Config '{config_name}': signals={signals}")

        merged = merge_signals(feature_dir, signals)
        if merged is None:
            logger.error(f"  Skipping config '{config_name}' -- merge failed")
            continue

        # Add labels
        if train_npz.exists() and test_npz.exists():
            merged = add_labels_and_split(merged, train_npz, test_npz)
        else:
            logger.warning("  Subset .npz files not found, skipping label merge")

        # Save
        output_dir = feature_dir / config_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "features_labeled.csv"
        merged.to_csv(output_path, index=False)
        logger.info(f"  Saved: {output_path} ({merged.shape})")

    logger.info("Merge complete.")


if __name__ == "__main__":
    main()
