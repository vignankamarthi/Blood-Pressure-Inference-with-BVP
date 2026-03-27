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
    Load a PulseDB Info .mat file.

    Info files are HDF5 (.mat v7.3) with a single top-level key containing
    a dict-of-lists. Each list has N elements (one per segment). Elements
    are single-item lists wrapping str or 0-d ndarray.

    Returns list of dicts with keys:
        Subj_Name (str), Subj_SegIDX (int), Source (str),
        Seg_SBP (float), Seg_DBP (float),
        Subj_Age (float), Subj_Gender (str)
    """
    data = load_mat_file(info_path)

    # Info files have a single field containing the data dict
    # Find the main field (not __header__, __version__, __globals__)
    info_data = None
    for key in data:
        if not key.startswith('_'):
            info_data = data[key]
            break
    if info_data is None:
        raise ValueError(f"No data field found in {info_path}")

    if not isinstance(info_data, dict):
        raise ValueError(
            f"Expected dict, got {type(info_data).__name__} in {info_path}"
        )

    # Validate required fields
    for field in ('Subj_Name', 'Subj_SegIDX'):
        if field not in info_data:
            raise ValueError(f"Missing required field '{field}' in {info_path}")

    n_segments = len(info_data['Subj_Name'])
    logger.info(f"  Found {n_segments} segment entries in {info_path.name}")

    # Unwrap helpers for the nested [value] structure
    def unwrap_str(lst, idx, default='Unknown'):
        try:
            val = lst[idx]
            return str(val[0]) if isinstance(val, list) else str(val)
        except (IndexError, TypeError):
            return default

    def unwrap_float(lst, idx, default=float('nan')):
        try:
            val = lst[idx]
            return float(val[0]) if isinstance(val, list) else float(val)
        except (IndexError, TypeError, ValueError):
            return default

    # Check which optional fields are present
    has_source = 'Source' in info_data
    has_sbp = 'Seg_SBP' in info_data
    has_dbp = 'Seg_DBP' in info_data
    has_age = 'Subj_Age' in info_data
    has_gender = 'Subj_Gender' in info_data

    records = []
    for i in range(n_segments):
        rec = {
            'Subj_Name': unwrap_str(info_data['Subj_Name'], i),
            'Subj_SegIDX': int(unwrap_float(info_data['Subj_SegIDX'], i)),
        }
        if has_source:
            rec['Source'] = unwrap_str(info_data['Source'], i)
        if has_sbp:
            rec['Seg_SBP'] = unwrap_float(info_data['Seg_SBP'], i)
        if has_dbp:
            rec['Seg_DBP'] = unwrap_float(info_data['Seg_DBP'], i)
        if has_age:
            rec['Subj_Age'] = unwrap_float(info_data['Subj_Age'], i)
        if has_gender:
            rec['Subj_Gender'] = unwrap_str(info_data['Subj_Gender'], i)
        records.append(rec)

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

    # Resolve subdirectory structure (PulseDB archives extract with subdirectory)
    mimic_subdir = mimic_path / "PulseDB_MIMIC"
    vital_subdir = vital_path / "PulseDB_Vital"
    if not mimic_subdir.exists():
        mimic_subdir = mimic_path  # fallback to flat structure
    if not vital_subdir.exists():
        vital_subdir = vital_path

    # Group records by subject, preserving Source and metadata from Info
    from collections import defaultdict
    subject_segments = defaultdict(list)
    subject_source = {}
    subject_meta = {}  # Per-segment metadata keyed by (subj_name, seg_idx)

    for rec in info_records:
        subj_name = rec['Subj_Name']
        seg_idx = rec['Subj_SegIDX']
        subject_segments[subj_name].append(seg_idx)

        # Store source (prefer explicit Source field, fallback to name suffix)
        if 'Source' in rec:
            subject_source[subj_name] = rec['Source']
        elif subj_name not in subject_source:
            suffix = subj_name.rsplit('_', 1)[-1]
            subject_source[subj_name] = 'MIMIC' if suffix == '0' else 'VitalDB'

        # Store per-segment metadata from Info file
        subject_meta[(subj_name, seg_idx)] = {
            'sbp': rec.get('Seg_SBP', np.nan),
            'dbp': rec.get('Seg_DBP', np.nan),
            'age': rec.get('Subj_Age', np.nan),
            'gender': rec.get('Subj_Gender', 'Unknown'),
        }

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
        subj_id = subj_name.split('_')[0]  # "p072634_0" -> "p072634"
        source = subject_source.get(subj_name, 'MIMIC')

        if source == 'MIMIC':
            seg_path = mimic_subdir / f"{subj_id}.mat"
        else:
            seg_path = vital_subdir / f"{subj_id}.mat"

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

        # Navigate to the segments structure
        if 'Subj_Wins' not in seg_data:
            logger.warning(f"  No 'Subj_Wins' in {seg_path}: {list(seg_data.keys())}")
            n_failed += 1
            continue

        segments = seg_data['Subj_Wins']

        # Normalize: mat73 loads struct arrays as dict-of-lists (column-oriented).
        # Convert to list-of-dicts (row-oriented) for uniform segment indexing.
        if isinstance(segments, dict):
            field_names = list(segments.keys())
            n_segs = len(next(iter(segments.values())))
            segments = [
                {k: segments[k][i] for k in field_names}
                for i in range(n_segs)
            ]

        for idx in seg_indices:
            try:
                # MATLAB is 1-indexed, Python is 0-indexed
                py_idx = idx - 1
                if py_idx < 0 or py_idx >= len(segments):
                    logger.debug(f"  Segment {idx} out of range for {subj_name} (has {len(segments)})")
                    continue

                seg = segments[py_idx]

                # Use metadata from Info file (already parsed, official PulseDB values)
                meta = subject_meta.get((subj_name, idx), {})
                sbp = meta.get('sbp', np.nan)
                dbp = meta.get('dbp', np.nan)
                age = meta.get('age', np.nan)
                gender = meta.get('gender', 'Unknown')

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
        logger.error("Run sbatch scripts/download_pulsedb_cluster.sbatch first")
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
        output_path = output_dir / output_name

        if not info_path.exists():
            logger.warning(f"Info file not found, skipping: {info_path}")
            continue

        # Checkpoint: skip subsets that already exist (resume after wall-time kill)
        if output_path.exists():
            logger.info(f"SKIP {output_name} -- already exists ({output_path})")
            continue

        logger.info(f"Generating {output_name}...")
        info_records = load_info_file(info_path)
        generate_subset(
            mimic_path, vital_path,
            info_records,
            output_path,
            use_raw_ppg=use_raw,
        )

    logger.info("All subsets generated.")


if __name__ == "__main__":
    main()
