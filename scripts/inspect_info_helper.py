#!/usr/bin/env python3
"""Inspect PulseDB Info .mat file structure to debug parsing."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def inspect_info(path: str):
    try:
        import mat73
        data = mat73.loadmat(path)
    except Exception:
        from scipy.io import loadmat
        data = loadmat(path, squeeze_me=True)

    print(f"\n{'='*60}")
    print(f"File: {path}")
    print(f"{'='*60}")
    print(f"Top-level keys: {[k for k in data.keys() if not k.startswith('_')]}")

    for key in data:
        if key.startswith('_'):
            continue
        val = data[key]
        print(f"\n--- {key} ---")
        print(f"  type: {type(val).__name__}")

        if isinstance(val, dict):
            print(f"  dict keys: {sorted(val.keys())[:15]}")
            for sk, sv in sorted(val.items())[:5]:
                if isinstance(sv, np.ndarray):
                    print(f"    {sk}: ndarray shape={sv.shape} dtype={sv.dtype}")
                    if sv.ndim == 1 and len(sv) > 0:
                        print(f"      first 3: {sv[:3]}")
                elif isinstance(sv, list):
                    print(f"    {sk}: list len={len(sv)}")
                    if len(sv) > 0:
                        print(f"      first 3: {sv[:3]}")
                        print(f"      [0] type: {type(sv[0]).__name__}")
                else:
                    print(f"    {sk}: {type(sv).__name__} = {sv}")

        elif isinstance(val, np.ndarray):
            print(f"  ndarray shape={val.shape} dtype={val.dtype}")
            if val.ndim == 1 and len(val) > 0:
                print(f"  first element type: {type(val[0]).__name__}")
                if isinstance(val[0], dict):
                    print(f"  [0] keys: {list(val[0].keys())}")

        elif isinstance(val, list):
            print(f"  list len={len(val)}")
            if len(val) > 0:
                print(f"  [0] type: {type(val[0]).__name__}")
                if isinstance(val[0], dict):
                    print(f"  [0] keys: {list(val[0].keys())}")

    print(f"\n{'='*60}")


def main():
    import os
    data_dir = os.environ.get('DATA_DIR', 'data/raw')
    info_dir = Path(data_dir) / "Info_Files"

    # Use smallest info file for speed
    candidates = ["AAMI_Test_Info.mat", "CalFree_Test_Info.mat", "CalBased_Test_Info.mat"]
    for name in candidates:
        path = info_dir / name
        if path.exists():
            inspect_info(str(path))
            break
    else:
        print(f"No info files found in {info_dir}")


if __name__ == "__main__":
    main()
