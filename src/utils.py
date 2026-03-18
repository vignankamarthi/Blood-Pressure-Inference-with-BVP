"""Utility functions: checkpointing, logging, serialization, memory management."""

import gc
import json
import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


def setup_logging(log_dir: str = "logs", name: str = "bp_pipeline") -> logging.Logger:
    """Set up file + console logging."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{name}_{timestamp}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # File handler (DEBUG level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

    # Console handler (INFO level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {log_file}")
    return logger


def convert_to_serializable(obj: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(v) for v in obj]
    return obj


def atomic_json_write(path: Path, data: Dict) -> None:
    """Write JSON atomically: write to .tmp then rename."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(convert_to_serializable(data), f, indent=2)
    os.replace(str(tmp_path), str(path))


def load_json(path: Path) -> Optional[Dict]:
    """Load JSON file if it exists."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def clear_memory():
    """Force garbage collection."""
    gc.collect()


@contextmanager
def timer(label: str, logger: Optional[logging.Logger] = None):
    """Context manager for timing code blocks."""
    start = time.time()
    yield
    elapsed = time.time() - start
    msg = f"{label}: {elapsed:.2f}s"
    if logger:
        logger.info(msg)
    else:
        print(msg)
