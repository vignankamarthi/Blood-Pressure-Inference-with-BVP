"""
Tests for the key-based feature-label merge pipeline.
Verifies that Rust output (file_name, segment_id) correctly maps to
.npz labels via parsed (subject, within-subject seg_idx).
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── parse_segment_key tests ───


def test_parse_segment_key_basic():
    from scripts.merge_features_labels import parse_segment_key
    subject, seg_idx = parse_segment_key("p072634_0.csv", "p072634_0_seg0003")
    assert subject == "p072634_0"
    assert seg_idx == 3


def test_parse_segment_key_zero():
    from scripts.merge_features_labels import parse_segment_key
    subject, seg_idx = parse_segment_key("p001_1.csv", "p001_1_seg0000")
    assert subject == "p001_1"
    assert seg_idx == 0


def test_parse_segment_key_large_index():
    from scripts.merge_features_labels import parse_segment_key
    subject, seg_idx = parse_segment_key("p000160_0.csv", "p000160_0_seg0157")
    assert subject == "p000160_0"
    assert seg_idx == 157


# ─── build_label_lookup tests ───


def test_build_label_lookup_correct_seg_idx(tmp_path):
    from scripts.merge_features_labels import build_label_lookup
    npz_path = tmp_path / "test.npz"
    np.savez(
        str(npz_path),
        subjects=np.array(["A", "A", "A", "B", "B"]),
        sbp=np.array([100.0, 110.0, 120.0, 130.0, 140.0]),
        dbp=np.array([60.0, 65.0, 70.0, 75.0, 80.0]),
    )

    df = build_label_lookup(npz_path)
    assert len(df) == 5
    assert list(df['seg_idx']) == [0, 1, 2, 0, 1]

    # Verify specific lookups
    a2 = df[(df['subject'] == 'A') & (df['seg_idx'] == 2)]
    assert float(a2['sbp'].iloc[0]) == 120.0
    assert float(a2['dbp'].iloc[0]) == 70.0

    b0 = df[(df['subject'] == 'B') & (df['seg_idx'] == 0)]
    assert float(b0['sbp'].iloc[0]) == 130.0
    assert float(b0['dbp'].iloc[0]) == 75.0


# ─── keyed merge tests ───


def test_keyed_merge_correct_alignment(tmp_path):
    """Critical test: features in alphabetical order, .npz in info-record order.
    Positional merge would assign wrong labels. Key-based merge must get it right."""
    import logging
    import scripts.merge_features_labels as mfl
    mfl.logger = logging.getLogger("test")

    from scripts.merge_features_labels import add_labels_keyed

    # Features in alphabetical file order (as Rust would produce)
    features_df = pd.DataFrame({
        'file_name': ['A.csv', 'A.csv', 'B.csv'],
        'segment_id': ['A_seg0000', 'A_seg0001', 'B_seg0000'],
        'signal_length': [1250, 1250, 1250],
        'nan_percentage': [0.0, 0.0, 0.0],
        'feature_1': [1.0, 2.0, 3.0],
    })

    # .npz with subjects in DIFFERENT order than alphabetical
    npz_path = tmp_path / "test.npz"
    np.savez(
        str(npz_path),
        subjects=np.array(["A", "A", "B"]),
        sbp=np.array([100.0, 110.0, 130.0]),
        dbp=np.array([60.0, 65.0, 75.0]),
    )

    result = add_labels_keyed(features_df, npz_path)

    # A_seg0000 should get sbp=100 (A's first segment)
    a0 = result[result['segment_id'] == 'A_seg0000']
    assert float(a0['sbp'].iloc[0]) == 100.0

    # A_seg0001 should get sbp=110 (A's second segment)
    a1 = result[result['segment_id'] == 'A_seg0001']
    assert float(a1['sbp'].iloc[0]) == 110.0

    # B_seg0000 should get sbp=130 (B's first segment)
    b0 = result[result['segment_id'] == 'B_seg0000']
    assert float(b0['sbp'].iloc[0]) == 130.0


def test_keyed_merge_unmatched_rows_logged(tmp_path):
    """Feature rows with no matching label should be dropped."""
    import logging
    import scripts.merge_features_labels as mfl
    mfl.logger = logging.getLogger("test")

    from scripts.merge_features_labels import add_labels_keyed

    features_df = pd.DataFrame({
        'file_name': ['A.csv', 'MISSING.csv'],
        'segment_id': ['A_seg0000', 'MISSING_seg0000'],
        'signal_length': [1250, 1250],
        'nan_percentage': [0.0, 0.0],
        'feature_1': [1.0, 2.0],
    })

    npz_path = tmp_path / "test.npz"
    np.savez(
        str(npz_path),
        subjects=np.array(["A"]),
        sbp=np.array([100.0]),
        dbp=np.array([60.0]),
    )

    result = add_labels_keyed(features_df, npz_path)
    assert len(result) == 1  # Only A matched
    assert result.iloc[0]['segment_id'] == 'A_seg0000'


def test_merge_multi_signal_on_segment_id(tmp_path):
    """Merging PPG + ECG features should produce correct column count."""
    import logging
    import scripts.merge_features_labels as mfl
    mfl.logger = logging.getLogger("test")

    from scripts.merge_features_labels import merge_signals

    # Create per-signal feature CSVs
    for signal in ['ppg', 'ecg']:
        signal_dir = tmp_path / signal / 'Train_Subset'
        signal_dir.mkdir(parents=True)
        df = pd.DataFrame({
            'file_name': ['A.csv', 'B.csv'],
            'segment_id': ['A_seg0000', 'B_seg0000'],
            'signal_length': [1250, 1250],
            'nan_percentage': [0.0, 0.0],
            'feat_1': [1.0, 2.0],
            'feat_2': [3.0, 4.0],
        })
        df.to_csv(signal_dir / 'features.csv', index=False)

    result = merge_signals(tmp_path, ['ppg', 'ecg'], 'Train_Subset')
    assert result is not None
    assert len(result) == 2
    # 4 metadata + 2 ppg features + 2 ecg features = 8 columns
    assert 'ppg_feat_1' in result.columns
    assert 'ecg_feat_1' in result.columns
    assert len(result.columns) == 8


def test_output_has_split_column(tmp_path):
    """Final output must have split column with train/test values."""
    import logging
    import scripts.merge_features_labels as mfl
    mfl.logger = logging.getLogger("test")

    from scripts.merge_features_labels import add_labels_keyed

    features_df = pd.DataFrame({
        'file_name': ['A.csv'],
        'segment_id': ['A_seg0000'],
        'signal_length': [1250],
        'nan_percentage': [0.0],
        'feature_1': [1.0],
    })

    npz_path = tmp_path / "test.npz"
    np.savez(
        str(npz_path),
        subjects=np.array(["A"]),
        sbp=np.array([100.0]),
        dbp=np.array([60.0]),
    )

    result = add_labels_keyed(features_df, npz_path)
    result['split'] = 'train'

    assert 'split' in result.columns
    assert result.iloc[0]['split'] == 'train'


def test_calfree_is_default_output():
    """CalFree should produce features_labeled.csv (no suffix)."""
    from scripts.merge_features_labels import TEST_SETS
    # CalFree is in the test sets
    assert 'CalFree_Test_Subset' in TEST_SETS


def test_column_order_independence(tmp_path):
    """Rust processes files alphabetically; .npz may have different order.
    Key-based merge must still produce correct label-feature alignment.
    This test would FAIL with the old positional merge."""
    import logging
    import scripts.merge_features_labels as mfl
    mfl.logger = logging.getLogger("test")

    from scripts.merge_features_labels import add_labels_keyed

    # Rust alphabetical order: B before Z
    features_df = pd.DataFrame({
        'file_name': ['B.csv', 'B.csv', 'Z.csv'],
        'segment_id': ['B_seg0000', 'B_seg0001', 'Z_seg0000'],
        'signal_length': [1250, 1250, 1250],
        'nan_percentage': [0.0, 0.0, 0.0],
        'feature_1': [10.0, 20.0, 30.0],
    })

    # .npz has Z first, then B (different from alphabetical)
    npz_path = tmp_path / "test.npz"
    np.savez(
        str(npz_path),
        subjects=np.array(["Z", "B", "B"]),
        sbp=np.array([999.0, 200.0, 210.0]),
        dbp=np.array([88.0, 70.0, 72.0]),
    )

    result = add_labels_keyed(features_df, npz_path)

    # B_seg0000 must get sbp=200 (B's first segment), NOT sbp=999 (positional match)
    b0 = result[result['segment_id'] == 'B_seg0000']
    assert float(b0['sbp'].iloc[0]) == 200.0

    # B_seg0001 must get sbp=210
    b1 = result[result['segment_id'] == 'B_seg0001']
    assert float(b1['sbp'].iloc[0]) == 210.0

    # Z_seg0000 must get sbp=999
    z0 = result[result['segment_id'] == 'Z_seg0000']
    assert float(z0['sbp'].iloc[0]) == 999.0
