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
from typing import Dict, List

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


def load_subject_signals_h5py(
    path: Path,
    seg_indices: List[int],
    fields: List[str],
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load specific signal fields for specific segments using h5py.

    MATLAB v7.3 .mat files store struct arrays as HDF5 object references.
    This reads only the requested fields for the requested segments,
    avoiding full deserialization (~30x faster than mat73.loadmat).

    Layout (confirmed by probe):
      /Subj_Wins/<field> -- Dataset, shape (1, N), dtype=object (references)
      Each ref dereferences to Dataset, shape (1, 1250), dtype=float64

    Args:
        path: Path to the subject .mat file
        seg_indices: 1-indexed segment indices (MATLAB convention)
        fields: Field names to extract (e.g., ['PPG_Record', 'ECG_F', 'ABP_Raw'])

    Returns:
        {matlab_idx: {field_name: np.ndarray (1D, float64)}}
        Missing fields -> empty arrays. Out-of-range indices omitted.
    """
    import h5py

    result = {}
    with h5py.File(str(path), 'r') as f:
        if 'Subj_Wins' not in f:
            raise ValueError(f"No 'Subj_Wins' in {path}: {list(f.keys())}")

        sw = f['Subj_Wins']

        # Determine segment count from the first available field
        n_segs = None
        for field_name in fields:
            if field_name in sw:
                ds = sw[field_name]
                n_segs = ds.shape[1] if ds.ndim == 2 else ds.shape[0]
                break

        if n_segs is None:
            return result

        for matlab_idx in seg_indices:
            py_idx = matlab_idx - 1
            if py_idx < 0 or py_idx >= n_segs:
                continue

            seg_data = {}
            for field_name in fields:
                if field_name not in sw:
                    seg_data[field_name] = np.array([], dtype=np.float64)
                    continue

                ds = sw[field_name]
                ref = ds[0, py_idx] if ds.ndim == 2 else ds[py_idx]
                data = f[ref][()].flatten().astype(np.float64)
                seg_data[field_name] = data

            result[matlab_idx] = seg_data

    return result


def load_info_file_h5py(info_path: Path) -> List[Dict]:
    """
    Load a PulseDB Info .mat file using h5py for targeted reads.

    Layout (confirmed by probe):
      /<subset_name>/<field> -- Dataset, shape (1, N), dtype=object (references)
      String fields (Subj_Name, Source, Gender): deref to (len, 1) uint16 -> chr()
      Float fields (SegIDX, SBP, DBP, Age): deref to (1, 1) float64

    Returns list of dicts with keys:
        Subj_Name (str), Subj_SegIDX (int), Source (str),
        Seg_SBP (float), Seg_DBP (float),
        Subj_Age (float), Subj_Gender (str)
    """
    import h5py

    records = []
    with h5py.File(str(info_path), 'r') as f:
        # Find main key (not #refs#)
        main_key = None
        for k in f.keys():
            if not k.startswith('#'):
                main_key = k
                break
        if main_key is None:
            raise ValueError(f"No data key found in {info_path}")

        main = f[main_key]

        # Validate required fields
        for field in ('Subj_Name', 'Subj_SegIDX'):
            if field not in main:
                raise ValueError(f"Missing required field '{field}' in {info_path}")

        # Get entry count
        ds_name = main['Subj_Name']
        n_entries = ds_name.shape[1] if ds_name.ndim == 2 else ds_name.shape[0]
        logger.info(f"  Found {n_entries} segment entries in {info_path.name}")

        def read_string(ds, idx):
            ref = ds[0, idx] if ds.ndim == 2 else ds[idx]
            data = f[ref][()]
            return ''.join(chr(c) for c in data.flatten())

        def read_float(ds, idx):
            ref = ds[0, idx] if ds.ndim == 2 else ds[idx]
            data = f[ref][()]
            return float(data.flatten()[0])

        # Check which optional fields are present
        has_source = 'Source' in main
        has_sbp = 'Seg_SBP' in main
        has_dbp = 'Seg_DBP' in main
        has_age = 'Subj_Age' in main
        has_gender = 'Subj_Gender' in main

        # Cache dataset handles (avoid repeated dict lookups)
        ds_segidx = main['Subj_SegIDX']
        ds_source = main['Source'] if has_source else None
        ds_sbp = main['Seg_SBP'] if has_sbp else None
        ds_dbp = main['Seg_DBP'] if has_dbp else None
        ds_age = main['Subj_Age'] if has_age else None
        ds_gender = main['Subj_Gender'] if has_gender else None

        for i in range(n_entries):
            rec = {
                'Subj_Name': read_string(ds_name, i),
                'Subj_SegIDX': int(read_float(ds_segidx, i)),
            }
            if has_source:
                rec['Source'] = read_string(ds_source, i)
            if has_sbp:
                rec['Seg_SBP'] = read_float(ds_sbp, i)
            if has_dbp:
                rec['Seg_DBP'] = read_float(ds_dbp, i)
            if has_age:
                rec['Subj_Age'] = read_float(ds_age, i)
            if has_gender:
                rec['Subj_Gender'] = read_string(ds_gender, i)
            records.append(rec)

            if (i + 1) % 50000 == 0:
                logger.info(f"  Parsed {i + 1}/{n_entries} Info entries")

    logger.info(f"  Loaded {len(records)} segment references from {info_path.name}")
    return records


def load_info_file(info_path: Path) -> List[Dict]:
    """
    Load a PulseDB Info .mat file. Tries h5py first (fast), falls back to mat73.

    Returns list of dicts with keys:
        Subj_Name (str), Subj_SegIDX (int), Source (str),
        Seg_SBP (float), Seg_DBP (float),
        Subj_Age (float), Subj_Gender (str)
    """
    # Fast path: h5py targeted reads
    try:
        return load_info_file_h5py(info_path)
    except Exception as e:
        logger.debug(f"  h5py failed for Info file, falling back to mat73: {e}")

    # Slow fallback: mat73 full deserialization
    data = load_mat_file(info_path)

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

    for field in ('Subj_Name', 'Subj_SegIDX'):
        if field not in info_data:
            raise ValueError(f"Missing required field '{field}' in {info_path}")

    n_segments = len(info_data['Subj_Name'])
    logger.info(f"  Found {n_segments} segment entries in {info_path.name}")

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

    # Pre-allocate fixed-size 2D arrays (all signals are 1250 samples)
    SIGNAL_LEN = 1250
    n_total = len(info_records)  # upper bound on segments
    ppg_arr = np.zeros((n_total, SIGNAL_LEN), dtype=np.float64)
    ecg_arr = np.zeros((n_total, SIGNAL_LEN), dtype=np.float64)
    abp_arr = np.zeros((n_total, SIGNAL_LEN), dtype=np.float64)
    sbp_arr = np.full(n_total, np.nan, dtype=np.float64)
    dbp_arr = np.full(n_total, np.nan, dtype=np.float64)
    age_arr = np.full(n_total, np.nan, dtype=np.float64)
    subjects = []
    genders = []
    write_idx = 0

    n_subjects = len(subject_segments)
    n_processed = 0
    n_failed = 0

    import time as _time
    t_start = _time.time()

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

        target_fields = [signal_fields['ppg'], signal_fields['ecg'], signal_fields['abp']]

        # Fast path: h5py targeted reads (only 3 fields, only needed segments)
        try:
            seg_signals = load_subject_signals_h5py(seg_path, seg_indices, target_fields)
        except Exception as e:
            logger.debug(f"  h5py failed for {seg_path}, falling back to mat73: {e}")
            # Slow fallback: mat73 full deserialization
            try:
                seg_data = load_mat_file(seg_path)
            except Exception as e2:
                logger.error(f"  Failed to load {seg_path}: {e2}")
                n_failed += 1
                continue

            if 'Subj_Wins' not in seg_data:
                logger.warning(f"  No 'Subj_Wins' in {seg_path}: {list(seg_data.keys())}")
                n_failed += 1
                continue

            segments = seg_data['Subj_Wins']
            if isinstance(segments, dict):
                field_names_mat = list(segments.keys())
                n_segs = len(next(iter(segments.values())))
                seg_signals = {}
                for idx in seg_indices:
                    py_idx = idx - 1
                    if 0 <= py_idx < n_segs:
                        seg_signals[idx] = {
                            fn: np.array(segments[fn][py_idx], dtype=np.float64).flatten()
                            if fn in segments else np.array([], dtype=np.float64)
                            for fn in target_fields
                        }

        for idx in seg_indices:
            if idx not in seg_signals:
                continue

            signals = seg_signals[idx]
            meta = subject_meta.get((subj_name, idx), {})

            ppg = signals.get(signal_fields['ppg'], np.array([], dtype=np.float64))
            ecg = signals.get(signal_fields['ecg'], np.array([], dtype=np.float64))
            abp = signals.get(signal_fields['abp'], np.array([], dtype=np.float64))

            if len(ppg) == 0:
                continue

            # Write directly into pre-allocated arrays (no list overhead)
            sig_len = min(len(ppg), SIGNAL_LEN)
            ppg_arr[write_idx, :sig_len] = ppg[:sig_len]
            ecg_arr[write_idx, :min(len(ecg), SIGNAL_LEN)] = ecg[:SIGNAL_LEN] if len(ecg) > 0 else 0
            abp_arr[write_idx, :min(len(abp), SIGNAL_LEN)] = abp[:SIGNAL_LEN] if len(abp) > 0 else 0
            sbp_arr[write_idx] = meta.get('sbp', np.nan)
            dbp_arr[write_idx] = meta.get('dbp', np.nan)
            age_arr[write_idx] = meta.get('age', np.nan)
            subjects.append(subj_name)
            genders.append(meta.get('gender', 'Unknown'))
            write_idx += 1

        n_processed += 1
        if n_processed % 100 == 0:
            elapsed = _time.time() - t_start
            rate = n_processed / elapsed
            logger.info(f"  Processed {n_processed}/{n_subjects} subjects ({write_idx} segments) [{rate:.1f} subj/s]")

    logger.info(f"  Final: {write_idx} segments from {n_processed} subjects ({n_failed} failed)")

    # Trim to actual size and save (uncompressed -- float signals don't compress well)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Saving {write_idx} segments to {output_path}...")
    np.savez(
        str(output_path),
        subjects=np.array(subjects),
        ppg_signals=ppg_arr[:write_idx],
        ecg_signals=ecg_arr[:write_idx],
        abp_signals=abp_arr[:write_idx],
        sbp=sbp_arr[:write_idx],
        dbp=dbp_arr[:write_idx],
        ages=age_arr[:write_idx],
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
