"""
TDD tests for h5py-based targeted HDF5 field loader.

Synthetic fixtures replicate the exact MATLAB v7.3 struct array layout
confirmed by probe_hdf5_helper.py on the cluster:
  - /Subj_Wins/<field> is a Dataset of shape (1, N), dtype=h5py.ref_dtype
  - Each reference resolves to a Dataset of shape (1, 1250), dtype=float64
  - Actual data lives in /#refs#/ group
"""

import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── Synthetic HDF5 fixture builder ───


def create_synthetic_subject_hdf5(
    path: Path,
    n_segments: int = 5,
    signal_length: int = 1250,
    fields: dict = None,
    seed: int = 42,
):
    """
    Create a synthetic HDF5 file matching PulseDB MATLAB v7.3 layout.

    Layout (confirmed by cluster probe):
      /Subj_Wins/<field>  -- Dataset, shape (1, N), dtype=h5py.ref_dtype
      /#refs#/<key>        -- Dataset, shape (1, signal_length), dtype=float64
    """
    rng = np.random.RandomState(seed)

    if fields is None:
        fields = {
            'PPG_Record': [rng.randn(signal_length).astype(np.float64) for _ in range(n_segments)],
            'ECG_F': [rng.randn(signal_length).astype(np.float64) for _ in range(n_segments)],
            'ABP_Raw': [rng.randn(signal_length).astype(np.float64) for _ in range(n_segments)],
        }

    with h5py.File(str(path), 'w') as f:
        refs_group = f.create_group('#refs#')
        sw_group = f.create_group('Subj_Wins')

        ref_counter = 0
        for field_name, segments_data in fields.items():
            n = len(segments_data)
            ref_array = np.empty((1, n), dtype=h5py.ref_dtype)

            for i, data in enumerate(segments_data):
                # Store as (1, signal_length) matching MATLAB convention
                ds_name = f"ref_{ref_counter}"
                ref_counter += 1
                ds = refs_group.create_dataset(ds_name, data=data.reshape(1, -1))
                ref_array[0, i] = ds.ref

            sw_group.create_dataset(field_name, data=ref_array)

    return path


# ─── Fixtures ───


@pytest.fixture
def synthetic_subject_file(tmp_path):
    """Create a synthetic HDF5 subject file with known data."""
    path = tmp_path / "p000001.mat"
    rng = np.random.RandomState(42)
    signals = {
        'PPG_Record': [rng.randn(1250).astype(np.float64) for _ in range(5)],
        'ECG_F': [rng.randn(1250).astype(np.float64) for _ in range(5)],
        'ABP_Raw': [rng.randn(1250).astype(np.float64) for _ in range(5)],
    }
    create_synthetic_subject_hdf5(path, fields=signals)
    return path, signals


# ─── Tests ───


def test_h5py_loader_returns_three_signals(synthetic_subject_file):
    """h5py loader returns PPG, ECG, ABP for a requested segment."""
    path, _ = synthetic_subject_file
    from scripts.generate_subsets import load_subject_signals_h5py

    result = load_subject_signals_h5py(path, seg_indices=[1], fields=['PPG_Record', 'ECG_F', 'ABP_Raw'])
    assert 1 in result
    assert 'PPG_Record' in result[1]
    assert 'ECG_F' in result[1]
    assert 'ABP_Raw' in result[1]


def test_h5py_loader_correct_data(synthetic_subject_file):
    """h5py loader returns exact signal values (not a different segment)."""
    path, expected_signals = synthetic_subject_file
    from scripts.generate_subsets import load_subject_signals_h5py

    result = load_subject_signals_h5py(path, seg_indices=[1], fields=['PPG_Record'])
    # MATLAB index 1 -> Python index 0
    np.testing.assert_array_almost_equal(
        result[1]['PPG_Record'],
        expected_signals['PPG_Record'][0],
    )


def test_h5py_loader_multiple_segments(synthetic_subject_file):
    """h5py loader handles multiple segment indices in one call."""
    path, expected_signals = synthetic_subject_file
    from scripts.generate_subsets import load_subject_signals_h5py

    result = load_subject_signals_h5py(path, seg_indices=[1, 3, 5], fields=['PPG_Record'])
    assert len(result) == 3
    # MATLAB index 3 -> Python index 2
    np.testing.assert_array_almost_equal(
        result[3]['PPG_Record'],
        expected_signals['PPG_Record'][2],
    )
    # MATLAB index 5 -> Python index 4
    np.testing.assert_array_almost_equal(
        result[5]['PPG_Record'],
        expected_signals['PPG_Record'][4],
    )


def test_h5py_loader_out_of_range_skipped(synthetic_subject_file):
    """Segment indices beyond actual count are silently omitted."""
    path, _ = synthetic_subject_file
    from scripts.generate_subsets import load_subject_signals_h5py

    result = load_subject_signals_h5py(path, seg_indices=[1, 999], fields=['PPG_Record'])
    assert 1 in result
    assert 999 not in result


def test_h5py_loader_no_subj_wins_raises(tmp_path):
    """File without Subj_Wins raises ValueError."""
    path = tmp_path / "bad.mat"
    with h5py.File(str(path), 'w') as f:
        f.create_dataset('garbage', data=[1, 2, 3])

    from scripts.generate_subsets import load_subject_signals_h5py
    with pytest.raises(ValueError, match="Subj_Wins"):
        load_subject_signals_h5py(path, seg_indices=[1], fields=['PPG_Record'])


def test_h5py_loader_returns_flat_1d_float64(synthetic_subject_file):
    """Returned signals are flattened to 1D float64 arrays."""
    path, _ = synthetic_subject_file
    from scripts.generate_subsets import load_subject_signals_h5py

    result = load_subject_signals_h5py(path, seg_indices=[1], fields=['PPG_Record', 'ECG_F', 'ABP_Raw'])
    for field in ['PPG_Record', 'ECG_F', 'ABP_Raw']:
        sig = result[1][field]
        assert sig.ndim == 1, f"{field} should be 1D, got {sig.ndim}D"
        assert sig.dtype == np.float64, f"{field} should be float64, got {sig.dtype}"
        assert len(sig) == 1250


def test_h5py_loader_missing_field_returns_empty(tmp_path):
    """Missing field in Subj_Wins returns empty array."""
    path = tmp_path / "partial.mat"
    rng = np.random.RandomState(42)
    # Only PPG_Record, no ECG_F
    create_synthetic_subject_hdf5(path, fields={
        'PPG_Record': [rng.randn(1250).astype(np.float64)],
    })

    from scripts.generate_subsets import load_subject_signals_h5py

    result = load_subject_signals_h5py(path, seg_indices=[1], fields=['PPG_Record', 'ECG_F'])
    assert len(result[1]['PPG_Record']) == 1250
    assert len(result[1]['ECG_F']) == 0


def test_probe_hdf5_sbatch_syntax():
    """Verify probe_hdf5.sbatch has no bash syntax errors."""
    script = Path(__file__).parent.parent / "scripts" / "probe_hdf5.sbatch"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"Syntax error: {result.stderr}"
