"""Tests for src/utils.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import convert_to_serializable, atomic_json_write, load_json, timer


def test_convert_numpy_int():
    assert isinstance(convert_to_serializable(np.int64(42)), int)
    assert convert_to_serializable(np.int64(42)) == 42


def test_convert_numpy_float():
    assert isinstance(convert_to_serializable(np.float64(3.14)), float)


def test_convert_numpy_array():
    arr = np.array([1, 2, 3])
    result = convert_to_serializable(arr)
    assert isinstance(result, list)
    assert result == [1, 2, 3]


def test_convert_nested_dict():
    data = {"key": np.int64(1), "nested": {"val": np.float64(2.5)}}
    result = convert_to_serializable(data)
    assert result == {"key": 1, "nested": {"val": 2.5}}


def test_atomic_json_write():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        atomic_json_write(path, {"key": "value"})
        assert path.exists()
        assert not path.with_suffix(".tmp").exists()
        with open(path) as f:
            data = json.load(f)
        assert data == {"key": "value"}


def test_load_json_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"
        with open(path, "w") as f:
            json.dump({"a": 1}, f)
        result = load_json(path)
        assert result == {"a": 1}


def test_load_json_missing():
    result = load_json(Path("/nonexistent/file.json"))
    assert result is None


def test_timer():
    import io
    with timer("test_op"):
        x = sum(range(1000))
    # Just verify it doesn't crash
