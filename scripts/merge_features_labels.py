#!/usr/bin/env python3
"""
Merge Rust-extracted feature CSVs with .npz subset labels into training-ready CSVs.

Key-based merge using (file_name, segment_id) -> (subject, within-subject seg_idx).
Produces one features_labeled.csv per (ablation_config, test_set) combination.

For each ablation config (ppg, ppg_ecg, ppg_abp, ppg_ecg_abp):
  - Load per-subset feature CSVs from data/features/{signal}/{subset}/features.csv
  - Prefix feature columns by signal name
  - Horizontal merge across signals on (file_name, segment_id)
  - Key-based join with .npz labels using parsed subject + seg_idx
  - Write data/features/{config}/features_labeled[_{test_set}].csv

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

# Which .npz provides labels for which subset directory
SUBSET_NPZ_MAP = {
    'Train_Subset': 'Train_Subset.npz',
    'CalBased_Test_Subset': 'CalBased_Test_Subset.npz',
    'CalFree_Test_Subset': 'CalFree_Test_Subset.npz',
    'AAMI_Test_Subset': 'AAMI_Test_Subset.npz',
    'AAMI_Cal_Subset': 'AAMI_Cal_Subset.npz',
}

# Test sets to evaluate against (each paired with training data)
TEST_SETS = ['CalFree_Test_Subset', 'CalBased_Test_Subset', 'AAMI_Test_Subset']


def parse_segment_key(file_name: str, segment_id: str):
    """
    Extract (subject, seg_idx) from Rust output columns.

    file_name = "p072634_0.csv" -> subject = "p072634_0"
    segment_id = "p072634_0_seg0003" -> seg_idx = 3
    """
    subject = file_name.replace('.csv', '')
    seg_part = segment_id.rsplit('_seg', 1)[-1]
    seg_idx = int(seg_part)
    return subject, seg_idx


def build_label_lookup(npz_path: Path) -> pd.DataFrame:
    """
    Build a DataFrame with (subject, seg_idx) -> (sbp, dbp) from an .npz file.

    The .npz subjects array has one entry per segment. Within each subject,
    seg_idx is the cumulative zero-based index (matching CSV column order
    from export_signals_csv.py).
    """
    data = np.load(str(npz_path))
    subjects = data['subjects']
    sbp = data['sbp']
    dbp = data['dbp']

    df = pd.DataFrame({
        'subject': subjects,
        'sbp': sbp,
        'dbp': dbp,
    })
    df['seg_idx'] = df.groupby('subject').cumcount()
    return df


def load_and_prefix(feature_dir: Path, signal_name: str, subset_name: str) -> pd.DataFrame:
    """Load a per-subset feature CSV and prefix feature columns with signal name."""
    csv_path = feature_dir / signal_name / subset_name / "features.csv"
    if not csv_path.exists():
        logger.error(f"  Feature CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    logger.info(f"  Loaded {signal_name}/{subset_name}: {df.shape}")

    # Prefix feature columns (not metadata)
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]
    rename_map = {c: f"{signal_name}_{c}" for c in feature_cols}
    df = df.rename(columns=rename_map)
    return df


def merge_signals(feature_dir: Path, signals: list, subset_name: str) -> pd.DataFrame:
    """Merge feature CSVs from multiple signals for one subset on (file_name, segment_id)."""
    dfs = []
    for signal in signals:
        df = load_and_prefix(feature_dir, signal, subset_name)
        if df is None:
            return None
        dfs.append(df)

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merge_keys = ['file_name', 'segment_id']
        feature_cols = [c for c in df.columns if c not in METADATA_COLS]
        cols_to_merge = merge_keys + feature_cols
        merged = merged.merge(df[cols_to_merge], on=merge_keys, how='inner')

    return merged


def add_labels_keyed(features_df: pd.DataFrame, npz_path: Path) -> pd.DataFrame:
    """
    Key-based label merge using (subject, seg_idx) parsed from Rust output columns.
    """
    label_df = build_label_lookup(npz_path)

    features_df = features_df.copy()
    subjects = []
    seg_indices = []
    for _, row in features_df.iterrows():
        subj, idx = parse_segment_key(row['file_name'], row['segment_id'])
        subjects.append(subj)
        seg_indices.append(idx)

    features_df['subject'] = subjects
    features_df['seg_idx'] = seg_indices

    merged = features_df.merge(label_df, on=['subject', 'seg_idx'], how='inner')

    n_unmatched = len(features_df) - len(merged)
    if n_unmatched > 0:
        logger.warning(f"  {n_unmatched} feature rows did not match labels")

    merged = merged.rename(columns={'subject': 'subject_id'})
    merged = merged.drop(columns=['seg_idx'])

    return merged


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

    for config_name, config in configs.items():
        signals = config["signals"]
        logger.info(f"Config '{config_name}': signals={signals}")

        for test_set in TEST_SETS:
            test_npz_name = SUBSET_NPZ_MAP.get(test_set)
            if not test_npz_name:
                continue
            test_npz = subset_dir / test_npz_name
            train_npz = subset_dir / "Train_Subset.npz"

            if not test_npz.exists():
                logger.info(f"  Skipping {test_set}: {test_npz} not found")
                continue
            if not train_npz.exists():
                logger.error(f"  Train_Subset.npz not found: {train_npz}")
                continue

            logger.info(f"  [{config_name}/{test_set}] Merging...")

            # Merge train features across signals
            train_features = merge_signals(feature_dir, signals, 'Train_Subset')
            if train_features is None:
                logger.error(f"  [{config_name}/{test_set}] Train feature merge failed")
                continue

            train_labeled = add_labels_keyed(train_features, train_npz)
            train_labeled['split'] = 'train'
            logger.info(f"  [{config_name}/{test_set}] Train: {len(train_labeled)} rows")

            # Merge test features across signals
            test_features = merge_signals(feature_dir, signals, test_set)
            if test_features is None:
                logger.error(f"  [{config_name}/{test_set}] Test feature merge failed")
                continue

            test_labeled = add_labels_keyed(test_features, test_npz)
            test_labeled['split'] = 'test'
            logger.info(f"  [{config_name}/{test_set}] Test: {len(test_labeled)} rows")

            # Combine train + test
            combined = pd.concat([train_labeled, test_labeled], ignore_index=True)

            # Output path: CalFree is default (no suffix), others get suffix
            output_dir = feature_dir / config_name
            output_dir.mkdir(parents=True, exist_ok=True)

            if test_set == 'CalFree_Test_Subset':
                output_path = output_dir / "features_labeled.csv"
            else:
                output_path = output_dir / f"features_labeled_{test_set}.csv"

            combined.to_csv(output_path, index=False)
            logger.info(f"  [{config_name}/{test_set}] Saved: {output_path} ({combined.shape})")

    logger.info("Merge complete.")


if __name__ == "__main__":
    main()
