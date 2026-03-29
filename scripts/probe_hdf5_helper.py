#!/usr/bin/env python3
"""
Probe HDF5 structure of PulseDB subject .mat files using h5py.
Dumps the exact layout needed to implement targeted field reads
(object reference shape, dereferenced data shape/dtype, timing).

Usage:
    DATA_DIR=data/raw python scripts/probe_hdf5_helper.py
"""

import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np


def probe_file(path: str, label: str):
    """Dump HDF5 structure for one subject .mat file."""
    print(f"\n{'='*60}")
    print(f"Probing: {label}")
    print(f"File: {path}")
    print(f"File size: {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"{'='*60}")

    with h5py.File(path, 'r') as f:
        # 1. Top-level keys
        print(f"\nTop-level keys: {list(f.keys())}")

        # 2. Subj_Wins structure
        if 'Subj_Wins' not in f:
            print("ERROR: No 'Subj_Wins' group/dataset")
            print(f"  Available: {list(f.keys())}")
            return

        sw = f['Subj_Wins']
        print(f"\nSubj_Wins type: {type(sw).__name__}")
        if isinstance(sw, h5py.Group):
            print(f"Subj_Wins keys ({len(sw)}): {sorted(sw.keys())}")
        elif isinstance(sw, h5py.Dataset):
            print(f"Subj_Wins shape: {sw.shape}, dtype: {sw.dtype}")

        # 3. Inspect each target signal field
        target_fields = ['PPG_Record', 'ECG_F', 'ABP_Raw', 'PPG_F']
        for field_name in target_fields:
            print(f"\n--- {field_name} ---")
            if field_name not in sw:
                print(f"  NOT FOUND in Subj_Wins")
                continue

            ds = sw[field_name]
            print(f"  HDF5 type: {type(ds).__name__}")

            if isinstance(ds, h5py.Dataset):
                print(f"  shape: {ds.shape}")
                print(f"  dtype: {ds.dtype}")
                print(f"  ndim: {ds.ndim}")

                if ds.dtype == h5py.ref_dtype:
                    print(f"  ** Contains HDF5 object references **")

                    # Dereference first element
                    for elem_idx in [0, 1]:
                        try:
                            if ds.ndim == 2:
                                ref = ds[0, elem_idx]
                                idx_desc = f"[0, {elem_idx}]"
                            else:
                                ref = ds[elem_idx]
                                idx_desc = f"[{elem_idx}]"

                            derefed = f[ref]
                            print(f"\n  Deref {idx_desc}:")
                            print(f"    type: {type(derefed).__name__}")
                            if isinstance(derefed, h5py.Dataset):
                                print(f"    shape: {derefed.shape}")
                                print(f"    dtype: {derefed.dtype}")
                                data = derefed[()]
                                print(f"    value shape after read: {data.shape}")
                                print(f"    flattened length: {data.flatten().shape[0]}")
                                print(f"    first 5 values: {data.flatten()[:5]}")
                            elif isinstance(derefed, h5py.Group):
                                print(f"    group keys: {list(derefed.keys())}")
                        except Exception as e:
                            print(f"  Deref {idx_desc} failed: {e}")
                            break
                else:
                    # Direct data (not references)
                    print(f"  Direct data (not object references)")
                    try:
                        sample = ds[()]
                        print(f"  value shape: {sample.shape}")
                        print(f"  first 5: {sample.flatten()[:5]}")
                    except Exception as e:
                        print(f"  Read failed: {e}")

            elif isinstance(ds, h5py.Group):
                print(f"  Is a Group with keys: {list(ds.keys())[:10]}")

        # 4. Check #refs# group
        print(f"\n--- #refs# group ---")
        if '#refs#' in f:
            refs_grp = f['#refs#']
            ref_keys = list(refs_grp.keys())
            print(f"  Total items: {len(ref_keys)}")
            print(f"  First 5 keys: {ref_keys[:5]}")
            # Inspect first ref item
            if ref_keys:
                first_item = refs_grp[ref_keys[0]]
                print(f"  [{ref_keys[0]}] type: {type(first_item).__name__}")
                if isinstance(first_item, h5py.Dataset):
                    print(f"  [{ref_keys[0]}] shape: {first_item.shape}, dtype: {first_item.dtype}")
        else:
            print("  No #refs# group found")

        # 5. Count segments
        print(f"\n--- Segment count ---")
        if 'PPG_Record' in sw:
            ds = sw['PPG_Record']
            if ds.ndim == 2:
                n_segs = ds.shape[1]
                print(f"  PPG_Record shape: {ds.shape} -> {n_segs} segments (axis=1)")
            elif ds.ndim == 1:
                n_segs = ds.shape[0]
                print(f"  PPG_Record shape: {ds.shape} -> {n_segs} segments (axis=0)")
            else:
                n_segs = 0
                print(f"  PPG_Record shape: {ds.shape} (unexpected ndim={ds.ndim})")
        else:
            n_segs = 0
            print("  PPG_Record not found, cannot count segments")

        # 6. Timing: targeted reads vs full file
        if n_segs > 0 and 'PPG_Record' in sw and sw['PPG_Record'].dtype == h5py.ref_dtype:
            print(f"\n--- Timing ---")

            # Single segment, 3 fields
            start = time.time()
            for field_name in ['PPG_Record', 'ECG_F', 'ABP_Raw']:
                if field_name not in sw:
                    continue
                ds = sw[field_name]
                ref = ds[0, 0] if ds.ndim == 2 else ds[0]
                _ = f[ref][()]
            t_single = time.time() - start
            print(f"  1 segment x 3 fields: {t_single:.4f}s")

            # All segments, 3 fields
            start = time.time()
            for i in range(n_segs):
                for field_name in ['PPG_Record', 'ECG_F', 'ABP_Raw']:
                    if field_name not in sw:
                        continue
                    ds = sw[field_name]
                    ref = ds[0, i] if ds.ndim == 2 else ds[i]
                    _ = f[ref][()]
            t_all = time.time() - start
            print(f"  {n_segs} segments x 3 fields: {t_all:.4f}s")
            print(f"  Per-segment average: {t_all / n_segs:.4f}s")

            # Compare: mat73 full load
            print(f"\n  (For reference: mat73 full load of this file takes ~30s)")
            print(f"  Estimated speedup: ~{30.0 / max(t_all, 0.001):.0f}x")


def probe_info_file(path: str):
    """Dump HDF5 structure of a PulseDB Info .mat file."""
    print(f"\n{'='*60}")
    print(f"Probing INFO file: {path}")
    print(f"File size: {os.path.getsize(path) / 1e6:.1f} MB")
    print(f"{'='*60}")

    with h5py.File(path, 'r') as f:
        print(f"\nTop-level keys: {list(f.keys())}")

        # Find main data key (not #refs#)
        main_key = None
        for k in f.keys():
            if not k.startswith('#'):
                main_key = k
                break

        if main_key is None:
            print("ERROR: No data key found")
            return

        print(f"\nMain key: '{main_key}'")
        main = f[main_key]
        print(f"Type: {type(main).__name__}")

        if isinstance(main, h5py.Group):
            print(f"Sub-keys ({len(main)}): {sorted(main.keys())}")

            # Inspect each field we care about
            target_fields = ['Subj_Name', 'Subj_SegIDX', 'Source',
                             'Seg_SBP', 'Seg_DBP', 'Subj_Age', 'Subj_Gender']
            for field_name in target_fields:
                print(f"\n--- {field_name} ---")
                if field_name not in main:
                    print("  NOT FOUND")
                    continue

                ds = main[field_name]
                print(f"  type: {type(ds).__name__}")
                if isinstance(ds, h5py.Dataset):
                    print(f"  shape: {ds.shape}")
                    print(f"  dtype: {ds.dtype}")

                    if ds.dtype == h5py.ref_dtype:
                        print(f"  ** Object references **")
                        # Deref first 2 elements
                        for i in [0, 1]:
                            try:
                                ref = ds[0, i] if ds.ndim == 2 else ds[i]
                                derefed = f[ref]
                                print(f"  Deref [{i}]: type={type(derefed).__name__}", end="")
                                if isinstance(derefed, h5py.Dataset):
                                    data = derefed[()]
                                    print(f", shape={data.shape}, dtype={data.dtype}")
                                    # Try to decode if uint16 (likely string)
                                    if data.dtype in (np.uint16, np.uint8):
                                        try:
                                            s = ''.join(chr(c) for c in data.flatten())
                                            print(f"  Decoded string: '{s}'")
                                        except Exception:
                                            print(f"  Raw values: {data.flatten()[:10]}")
                                    else:
                                        print(f"  Values: {data.flatten()[:5]}")
                                else:
                                    print()
                            except Exception as e:
                                print(f"  Deref [{i}] failed: {e}")
                    else:
                        print(f"  Direct data")
                        try:
                            data = ds[()]
                            print(f"  Shape: {data.shape}")
                            print(f"  First 5: {data.flatten()[:5]}")
                        except Exception as e:
                            print(f"  Read failed: {e}")

            # Timing: read all 7 target fields for first 100 entries
            print(f"\n--- Timing ---")
            start = time.time()
            count = 0
            for field_name in target_fields:
                if field_name not in main:
                    continue
                ds = main[field_name]
                if ds.dtype == h5py.ref_dtype:
                    n = min(100, ds.shape[1] if ds.ndim == 2 else ds.shape[0])
                    for i in range(n):
                        ref = ds[0, i] if ds.ndim == 2 else ds[i]
                        _ = f[ref][()]
                    count = n
            elapsed = time.time() - start
            print(f"  7 fields x {count} entries: {elapsed:.4f}s")
            if count > 0:
                print(f"  Per-entry: {elapsed / count:.6f}s")

                # Extrapolate for full file
                total_entries = ds.shape[1] if ds.ndim == 2 else ds.shape[0]
                est_total = (elapsed / count) * total_entries
                print(f"  Total entries: {total_entries}")
                print(f"  Estimated full read: {est_total:.1f}s")

        elif isinstance(main, h5py.Dataset):
            print(f"Shape: {main.shape}, dtype: {main.dtype}")


def main():
    data_dir = os.environ.get('DATA_DIR', 'data/raw')

    mimic_dir = Path(data_dir) / "MIMIC" / "PulseDB_MIMIC"
    vital_dir = Path(data_dir) / "VitalDB" / "PulseDB_Vital"

    probed = False
    for d, label in [(mimic_dir, "MIMIC"), (vital_dir, "VitalDB")]:
        if not d.exists():
            print(f"Directory not found: {d}")
            continue
        files = sorted(d.glob("*.mat"))[:1]
        for fpath in files:
            probe_file(str(fpath), label)
            probed = True

    if not probed:
        print("ERROR: No .mat files found to probe")
        print(f"  Checked: {mimic_dir}")
        print(f"  Checked: {vital_dir}")
        sys.exit(1)

    # Probe Info files (use smallest first for quick validation)
    info_dir = Path(data_dir) / "Info_Files"
    if info_dir.exists():
        info_files = sorted(info_dir.glob("*.mat"), key=lambda p: p.stat().st_size)
        # Probe the smallest Info file (AAMI_Test_Info.mat ~4.8 MB)
        if info_files:
            probe_info_file(str(info_files[0]))

    print(f"\n{'='*60}")
    print("Probe complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
