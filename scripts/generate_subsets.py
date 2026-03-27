#!/usr/bin/env python3
"""
Python equivalent of PulseDB's Generate_Subsets.m
Combines segment files + info files into train/test subset files.

CRITICAL DIFFERENCE from MATLAB script:
  - MATLAB uses PPG_F (filtered/normalized PPG)
  - We use PPG_Record (raw, unnormalized) per LOCKED normalization decision
  - No pre-extraction normalization. StandardScaler applied AFTER feature extraction.

Usage:
    python scripts/generate_subsets.py --data-dir data/raw --output-dir data/subsets

Prerequisites:
    - Segment files extracted: data/raw/MIMIC/*.mat, data/raw/VitalDB/*.mat
    - Info files downloaded: data/raw/Info_Files/*.mat
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import setup_logging

logger = None


def load_mat_file(path: Path) -> Dict:
    """Load a .mat file (v7.3 or earlier)."""
    try:
        import mat73
        return mat73.loadmat(str(path))
    except Exception:
        from scipy.io import loadmat
        return loadmat(str(path), squeeze_me=True)


def load_info_file(info_path: Path) -> List[Dict]:
    """
    Load an Info .mat file.
    Returns list of {Subj_Name: str, Subj_SegIDX: int} dicts.
    """
    data = load_mat_file(info_path)

    # Info files have a single field containing the struct array
    # Find the main field (not __header__, __version__, __globals__)
    for key in data:
        if not key.startswith('_'):
            info_data = data[key]
            break
    else:
        raise ValueError(f"No data field found in {info_path}")

    # TODO: Parse info_data into list of {Subj_Name: str, Subj_SegIDX: int} dicts.
    # The exact format depends on how mat73 loads PulseDB Info files.
    # Run inspect_info.sbatch first, then implement the parser for the actual format.
    records = []

    logger.info(f"  Info data type: {type(info_data).__name__}")
    if isinstance(info_data, dict):
        logger.info(f"  Keys: {sorted(info_data.keys())[:15]}")

    # PLACEHOLDER: implement once format is known from inspect_info output
    logger.warning(f"  Parser not yet implemented for this format -- run inspect_info.sbatch first")

    logger.info(f"  Loaded {len(records)} segment references from {info_path.name}")
    return records


def generate_subset(
    mimic_path: Path,
    vital_path: Path,
    info_records: List[Dict],
    output_path: Path,
    use_raw_ppg: bool = True,
):
    """
    Generate a subset file from segment files + info records.

    Args:
        mimic_path: Path to MIMIC segment .mat files
        vital_path: Path to VitalDB segment .mat files
        info_records: List of {Subj_Name, Subj_SegIDX} dicts
        output_path: Where to save the output
        use_raw_ppg: If True, use PPG_Record (raw). If False, use PPG_F (filtered).
    """
    ppg_field = "PPG_Record" if use_raw_ppg else "PPG_F"
    logger.info(f"  Using PPG field: {ppg_field}")

    # Group records by subject
    from collections import defaultdict
    subject_segments = defaultdict(list)
    for rec in info_records:
        subject_segments[rec['Subj_Name']].append(rec['Subj_SegIDX'])

    # Signal field mapping
    signal_fields = {
        'ppg': ppg_field,
        'ecg': 'ECG_F',
        'abp': 'ABP_Raw',
    }

    # Pre-allocate output lists
    subjects = []
    ppg_signals = []
    ecg_signals = []
    abp_signals = []
    sbp_values = []
    dbp_values = []
    ages = []
    genders = []

    n_subjects = len(subject_segments)
    n_processed = 0
    n_failed = 0

    for subj_name, seg_indices in subject_segments.items():
        subj_id = subj_name[:7]
        source = int(subj_name[-1])  # 0=MIMIC, 1=VitalDB

        if source == 0:
            seg_path = mimic_path / f"{subj_id}.mat"
        else:
            seg_path = vital_path / f"{subj_id}.mat"

        if not seg_path.exists():
            logger.warning(f"  Segment file not found: {seg_path}")
            n_failed += 1
            continue

        try:
            seg_data = load_mat_file(seg_path)
        except Exception as e:
            logger.error(f"  Failed to load {seg_path}: {e}")
            n_failed += 1
            continue

        # Navigate to the segments array
        # Structure: Subj_Wins is array of structs
        if 'Subj_Wins' in seg_data:
            segments = seg_data['Subj_Wins']
        else:
            logger.warning(f"  No 'Subj_Wins' in {seg_path}: {list(seg_data.keys())}")
            n_failed += 1
            continue

        for idx in seg_indices:
            try:
                # MATLAB is 1-indexed, Python is 0-indexed
                seg = segments[idx - 1] if isinstance(segments, (list, np.ndarray)) else segments

                # Extract metadata
                if isinstance(seg, dict):
                    sbp = float(seg.get('SegSBP', np.nan))
                    dbp = float(seg.get('SegDBP', np.nan))
                    age = float(seg.get('Age', np.nan))
                    gender = str(seg.get('Gender', 'Unknown'))
                elif hasattr(seg, 'SegSBP'):
                    sbp = float(getattr(seg, 'SegSBP', np.nan))
                    dbp = float(getattr(seg, 'SegDBP', np.nan))
                    age = float(getattr(seg, 'Age', np.nan))
                    gender = str(getattr(seg, 'Gender', 'Unknown'))
                else:
                    logger.debug(f"  Segment {idx} missing metadata fields")
                    continue

                # Extract all 3 signals (PPG required, ECG/ABP optional)
                def get_signal(seg, field_name):
                    if isinstance(seg, dict):
                        val = seg.get(field_name, [])
                    elif hasattr(seg, field_name):
                        val = getattr(seg, field_name)
                    else:
                        return np.array([], dtype=np.float64)
                    return np.array(val, dtype=np.float64).flatten()

                ppg = get_signal(seg, signal_fields['ppg'])
                ecg = get_signal(seg, signal_fields['ecg'])
                abp = get_signal(seg, signal_fields['abp'])

                if len(ppg) == 0:
                    continue

                subjects.append(subj_name)
                ppg_signals.append(ppg)
                ecg_signals.append(ecg)
                abp_signals.append(abp)
                sbp_values.append(sbp)
                dbp_values.append(dbp)
                ages.append(age)
                genders.append(gender)

            except (IndexError, KeyError, TypeError) as e:
                logger.debug(f"  Segment {idx} error for {subj_name}: {e}")
                continue

        n_processed += 1
        if n_processed % 100 == 0:
            logger.info(f"  Processed {n_processed}/{n_subjects} subjects ({len(subjects)} segments)")

    logger.info(f"  Final: {len(subjects)} segments from {n_processed} subjects ({n_failed} failed)")

    # Save as numpy arrays (more portable than .mat for our pipeline)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        subjects=np.array(subjects),
        ppg_signals=np.array(ppg_signals, dtype=object),  # variable length
        ecg_signals=np.array(ecg_signals, dtype=object),
        abp_signals=np.array(abp_signals, dtype=object),
        sbp=np.array(sbp_values, dtype=np.float64),
        dbp=np.array(dbp_values, dtype=np.float64),
        ages=np.array(ages, dtype=np.float64),
        genders=np.array(genders),
    )
    logger.info(f"  Saved to {output_path}")


def main():
    global logger

    parser = argparse.ArgumentParser(description="Generate PulseDB subsets (Python equivalent of Generate_Subsets.m)")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                        help="Root directory with MIMIC/, VitalDB/, Info_Files/")
    parser.add_argument("--output-dir", type=str, default="data/subsets",
                        help="Output directory for subset files")
    parser.add_argument("--use-filtered-ppg", action="store_true",
                        help="Use PPG_F (filtered) instead of PPG_Record (raw). DEFAULT: raw (locked decision)")
    args = parser.parse_args()

    logger = setup_logging(name="generate_subsets")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mimic_path = data_dir / "MIMIC"
    vital_path = data_dir / "VitalDB"
    info_dir = data_dir / "Info_Files"

    use_raw = not args.use_filtered_ppg

    # Check prerequisites
    if not mimic_path.exists():
        logger.error(f"MIMIC segment directory not found: {mimic_path}")
        logger.error("Run scripts/download_pulsedb.sh first")
        sys.exit(1)
    if not info_dir.exists():
        logger.error(f"Info files directory not found: {info_dir}")
        logger.error("Download Info_Files from PulseDB Box/Drive link")
        sys.exit(1)

    # Generate each subset
    subsets = [
        ("Train_Info.mat", "Train_Subset.npz"),
        ("CalBased_Test_Info.mat", "CalBased_Test_Subset.npz"),
        ("CalFree_Test_Info.mat", "CalFree_Test_Subset.npz"),
    ]

    # Optional AAMI subsets
    if (info_dir / "AAMI_Test_Info.mat").exists():
        subsets.extend([
            ("AAMI_Test_Info.mat", "AAMI_Test_Subset.npz"),
            ("AAMI_Cal_Info.mat", "AAMI_Cal_Subset.npz"),
        ])

    for info_name, output_name in subsets:
        info_path = info_dir / info_name
        if not info_path.exists():
            logger.warning(f"Info file not found, skipping: {info_path}")
            continue

        logger.info(f"Generating {output_name}...")
        info_records = load_info_file(info_path)
        generate_subset(
            mimic_path, vital_path,
            info_records,
            output_dir / output_name,
            use_raw_ppg=use_raw,
        )

    logger.info("All subsets generated.")


if __name__ == "__main__":
    main()
