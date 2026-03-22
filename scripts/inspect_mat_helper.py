#!/usr/bin/env python3
"""
Inspect PulseDB .mat file structure (GATE G5).
Loads one MIMIC and one VitalDB file and prints field names, shapes, dtypes.
If structure differs from expected, this is a STOP point.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_file(path: str, label: str):
    """Inspect a single .mat file."""
    import mat73
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Inspecting: {label}")
    print(f"File: {path}")
    print(f"{'='*60}")

    data = mat73.loadmat(path)
    print(f"\nTop-level keys: {list(data.keys())}")

    for key, val in data.items():
        if key.startswith('_'):
            continue
        print(f"\n--- {key} ---")
        if isinstance(val, list):
            print(f"  Type: list, length={len(val)}")
            if len(val) > 0:
                item = val[0]
                if isinstance(item, dict):
                    print(f"  [0] type: dict")
                    print(f"  [0] keys: {sorted(item.keys())}")
                    for k, v in sorted(item.items()):
                        if hasattr(v, 'shape'):
                            print(f"    {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
                        elif isinstance(v, (int, float)):
                            print(f"    {k}: {type(v).__name__}, value={v}")
                        elif isinstance(v, str):
                            print(f"    {k}: str, value='{v}'")
                        elif isinstance(v, np.ndarray):
                            print(f"    {k}: ndarray, shape={v.shape}, dtype={v.dtype}")
                        else:
                            print(f"    {k}: {type(v).__name__}")
                else:
                    print(f"  [0] type: {type(item).__name__}")
        elif isinstance(val, dict):
            print(f"  Type: dict, keys={sorted(val.keys())}")
        elif hasattr(val, 'shape'):
            print(f"  Type: ndarray, shape={val.shape}, dtype={val.dtype}")
        else:
            print(f"  Type: {type(val).__name__}")

    # Check for expected fields
    expected_fields = {'PPG_Record', 'PPG_F', 'ECG_F', 'ABP_Raw', 'SegSBP', 'SegDBP', 'Age', 'Gender'}
    if 'Subj_Wins' in data:
        wins = data['Subj_Wins']
        if isinstance(wins, list) and len(wins) > 0 and isinstance(wins[0], dict):
            found = set(wins[0].keys())
            missing = expected_fields - found
            extra = found - expected_fields
            print(f"\n--- GATE G5 Check ---")
            print(f"  Expected fields: {sorted(expected_fields)}")
            print(f"  Found fields: {sorted(found)}")
            if missing:
                print(f"  MISSING: {sorted(missing)}")
            if extra:
                print(f"  Extra (not expected): {sorted(extra)}")
            if not missing:
                print(f"  GATE G5: PASS -- all expected fields present")
            else:
                print(f"  GATE G5: FAIL -- missing fields detected")
                print(f"  ACTION: STOP and review before proceeding")

    # Count segments
    if 'Subj_Wins' in data and isinstance(data['Subj_Wins'], list):
        n_segments = len(data['Subj_Wins'])
        print(f"\n  Segments in this subject: {n_segments}")

        # Check PPG_Record shape
        if n_segments > 0 and isinstance(data['Subj_Wins'][0], dict):
            seg = data['Subj_Wins'][0]
            for signal_key in ['PPG_Record', 'PPG_F', 'ECG_F', 'ABP_Raw']:
                if signal_key in seg:
                    sig = seg[signal_key]
                    if hasattr(sig, 'shape'):
                        print(f"  {signal_key} shape: {sig.shape}")
                    elif isinstance(sig, list):
                        print(f"  {signal_key} length: {len(sig)}")


def main():
    import os

    data_dir = os.environ.get('DATA_DIR', 'data/raw')

    # Find one MIMIC and one VitalDB file
    mimic_dir = Path(data_dir) / "MIMIC" / "PulseDB_MIMIC"
    vital_dir = Path(data_dir) / "VitalDB" / "PulseDB_Vital"

    mimic_files = sorted(mimic_dir.glob("*.mat"))[:1] if mimic_dir.exists() else []
    vital_files = sorted(vital_dir.glob("*.mat"))[:1] if vital_dir.exists() else []

    if not mimic_files and not vital_files:
        print("ERROR: No .mat files found")
        print(f"  Checked: {mimic_dir}")
        print(f"  Checked: {vital_dir}")
        sys.exit(1)

    for f in mimic_files:
        inspect_file(str(f), "MIMIC")

    for f in vital_files:
        inspect_file(str(f), "VitalDB")

    # Also check Info files
    info_dir = Path(data_dir) / "Info_Files"
    if info_dir.exists():
        print(f"\n{'='*60}")
        print(f"Info Files:")
        print(f"{'='*60}")
        for f in sorted(info_dir.glob("*.mat")):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name}: {size_mb:.1f} MB")

    print(f"\n{'='*60}")
    print("Inspection complete.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
