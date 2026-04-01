#!/usr/bin/env python3
"""
Export .npz subset signals to per-signal CSV directories for Rust feature extraction.

The Rust CLI expects CSVs where each column is a signal segment.
This script writes one CSV per subject per signal type.

Usage:
    python scripts/export_signals_csv.py --subset-dir data/subsets --output-dir data/signals
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

logger = None

SIGNAL_KEYS = {
    "ppg": "ppg_signals",
    "ecg": "ecg_signals",
    "abp": "abp_signals",
}


def export_subset(subset_path: Path, output_base: Path):
    """Export one .npz subset to per-signal CSV directories."""
    logger.info(f"Loading {subset_path.name}...")
    data = np.load(str(subset_path), allow_pickle=True)

    subjects = data['subjects']
    n_segments = len(subjects)
    subset_name = subset_path.stem  # e.g., "Train_Subset"

    logger.info(f"  {n_segments} segments, {len(np.unique(subjects))} unique subjects")

    for signal_name, signal_key in SIGNAL_KEYS.items():
        if signal_key not in data:
            logger.warning(f"  Signal '{signal_key}' not found in {subset_path.name}, skipping")
            continue

        signals = data[signal_key]
        signal_dir = output_base / signal_name / subset_name
        signal_dir.mkdir(parents=True, exist_ok=True)

        # Group segments by subject
        unique_subjects = np.unique(subjects)
        n_exported = 0
        n_skipped = 0

        for subj in unique_subjects:
            # Skip if already exported (resume after wall-time kill)
            csv_path = signal_dir / f"{subj}.csv"
            if csv_path.exists():
                n_exported += 1
                continue

            mask = subjects == subj
            subj_signals = signals[mask]

            # Filter out empty signals
            valid_signals = []
            for sig in subj_signals:
                sig_arr = np.array(sig, dtype=np.float64)
                if len(sig_arr) > 0:
                    valid_signals.append(sig_arr)

            if not valid_signals:
                n_skipped += 1
                continue

            # Write CSV: each column is a segment, rows are time samples
            # Pad to max length (should be 1250 for all, but handle variable)
            max_len = max(len(s) for s in valid_signals)
            csv_data = {}
            for i, sig in enumerate(valid_signals):
                col_name = f"{subj}_seg{i:04d}"
                if len(sig) < max_len:
                    sig = np.pad(sig, (0, max_len - len(sig)), constant_values=np.nan)
                csv_data[col_name] = sig

            df = pd.DataFrame(csv_data)
            csv_path = signal_dir / f"{subj}.csv"
            df.to_csv(csv_path, index=False)
            n_exported += 1

        logger.info(f"  {signal_name}: exported {n_exported} subjects, skipped {n_skipped}")


def main():
    global logger

    parser = argparse.ArgumentParser(description="Export .npz signals to per-signal CSVs")
    parser.add_argument("--subset-dir", type=str, default="data/subsets")
    parser.add_argument("--output-dir", type=str, default="data/signals")
    args = parser.parse_args()

    logger = setup_logging(name="export_signals")

    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)

    npz_files = sorted(subset_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No .npz files found in {subset_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} subset files")

    for npz_file in npz_files:
        export_subset(npz_file, output_dir)

    logger.info("Export complete.")
    for signal_name in SIGNAL_KEYS:
        signal_dir = output_dir / signal_name
        if signal_dir.exists():
            n_files = sum(1 for _ in signal_dir.rglob("*.csv"))
            logger.info(f"  {signal_name}: {n_files} CSV files")


if __name__ == "__main__":
    main()
