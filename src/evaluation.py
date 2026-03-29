"""
Clinical evaluation suite for blood pressure estimation.
Metrics: ME, SD, MAE, RMSE, R^2, Median AE, Max Error, AAMI, BHS, Bland-Altman.
"""

import logging
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error, max_error

from .utils import atomic_json_write

logger = logging.getLogger("bp_pipeline")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Compute all accuracy metrics for BP estimation."""
    errors = y_true - y_pred

    me = float(np.mean(errors))
    sd = float(np.std(errors, ddof=1))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    med_ae = float(median_absolute_error(y_true, y_pred))
    max_err = float(max_error(y_true, y_pred))

    return {
        "ME": me,
        "SD": sd,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MedianAE": med_ae,
        "MaxError": max_err,
        "n_samples": len(y_true),
    }


def aami_compliance(metrics: Dict) -> Dict:
    """
    Check AAMI (Association for the Advancement of Medical Instrumentation) compliance.
    Standard: ME < 5 mmHg AND SD < 8 mmHg.
    """
    me_pass = abs(metrics["ME"]) < 5.0
    sd_pass = metrics["SD"] < 8.0
    grade = "PASS" if (me_pass and sd_pass) else "FAIL"

    return {
        "grade": grade,
        "ME_pass": me_pass,
        "SD_pass": sd_pass,
        "ME_value": metrics["ME"],
        "SD_value": metrics["SD"],
        "ME_threshold": 5.0,
        "SD_threshold": 8.0,
    }


def bhs_grading(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    British Hypertension Society grading.
    Grade A: >= 60% within 5, >= 85% within 10, >= 95% within 15 mmHg
    Grade B: >= 50% within 5, >= 75% within 10, >= 90% within 15 mmHg
    Grade C: >= 40% within 5, >= 65% within 10, >= 85% within 15 mmHg
    Grade D: below Grade C
    """
    abs_errors = np.abs(y_true - y_pred)
    n = len(abs_errors)

    pct_5 = float(np.sum(abs_errors <= 5) / n * 100)
    pct_10 = float(np.sum(abs_errors <= 10) / n * 100)
    pct_15 = float(np.sum(abs_errors <= 15) / n * 100)

    if pct_5 >= 60 and pct_10 >= 85 and pct_15 >= 95:
        grade = "A"
    elif pct_5 >= 50 and pct_10 >= 75 and pct_15 >= 90:
        grade = "B"
    elif pct_5 >= 40 and pct_10 >= 65 and pct_15 >= 85:
        grade = "C"
    else:
        grade = "D"

    return {
        "grade": grade,
        "pct_within_5": pct_5,
        "pct_within_10": pct_10,
        "pct_within_15": pct_15,
    }


def bland_altman(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Bland-Altman agreement analysis.
    Returns bias, limits of agreement, and data for plotting.
    """
    diff = y_true - y_pred
    mean_vals = (y_true + y_pred) / 2.0

    bias = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    lower_limit = bias - 1.96 * sd
    upper_limit = bias + 1.96 * sd

    return {
        "bias": bias,
        "sd": sd,
        "lower_limit": float(lower_limit),
        "upper_limit": float(upper_limit),
        "mean_values": mean_vals.tolist(),
        "differences": diff.tolist(),
    }


def bp_category(sbp: float, dbp: float) -> str:
    """Classify BP into clinical categories (AHA guidelines)."""
    if sbp < 120 and dbp < 80:
        return "normal"
    elif sbp < 130 and dbp < 80:
        return "elevated"
    elif sbp < 140 or dbp < 90:
        return "stage1"
    else:
        return "stage2"


def stratified_evaluation(
    y_true_sbp: np.ndarray, y_pred_sbp: np.ndarray,
    y_true_dbp: np.ndarray, y_pred_dbp: np.ndarray,
) -> Dict:
    """Evaluate error stratified by BP category."""
    categories = [bp_category(s, d) for s, d in zip(y_true_sbp, y_true_dbp)]
    results = {}

    for cat in ["normal", "elevated", "stage1", "stage2"]:
        mask = np.array([c == cat for c in categories])
        if mask.sum() == 0:
            continue

        results[cat] = {
            "n_samples": int(mask.sum()),
            "sbp": compute_metrics(y_true_sbp[mask], y_pred_sbp[mask]),
            "dbp": compute_metrics(y_true_dbp[mask], y_pred_dbp[mask]),
        }

    return results


def evaluate_model(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target: str,
    results_dir: Path,
) -> Dict:
    """Full evaluation of a single model on a single target."""
    metrics = compute_metrics(y_true, y_pred)
    aami = aami_compliance(metrics)
    bhs = bhs_grading(y_true, y_pred)
    ba = bland_altman(y_true, y_pred)

    result = {
        "model": model_name,
        "target": target,
        "metrics": metrics,
        "aami": aami,
        "bhs": bhs,
        "bland_altman": {k: v for k, v in ba.items() if k not in ("mean_values", "differences")},
    }

    # Save
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"{model_name}_{target}_metrics.json"
    atomic_json_write(result_path, result)

    logger.info(f"  {model_name} ({target}): MAE={metrics['MAE']:.2f}, "
                f"RMSE={metrics['RMSE']:.2f}, R2={metrics['R2']:.4f}, "
                f"AAMI={aami['grade']}, BHS={bhs['grade']}")

    return result


def generate_leaderboard(results_dir: Path) -> Dict:
    """Generate model comparison leaderboard from saved results."""
    import json

    entries = []
    for json_file in sorted(results_dir.glob("*_metrics.json")):
        with open(json_file) as f:
            result = json.load(f)
        entries.append({
            "model": result["model"],
            "target": result["target"],
            "MAE": result["metrics"]["MAE"],
            "RMSE": result["metrics"]["RMSE"],
            "R2": result["metrics"]["R2"],
            "ME": result["metrics"]["ME"],
            "SD": result["metrics"]["SD"],
            "AAMI": result["aami"]["grade"],
            "BHS": result["bhs"]["grade"],
        })

    # Sort by MAE
    entries.sort(key=lambda x: x["MAE"])

    leaderboard = {"entries": entries, "best_sbp": None, "best_dbp": None}

    sbp_entries = [e for e in entries if e["target"] == "sbp"]
    dbp_entries = [e for e in entries if e["target"] == "dbp"]
    if sbp_entries:
        leaderboard["best_sbp"] = sbp_entries[0]["model"]
    if dbp_entries:
        leaderboard["best_dbp"] = dbp_entries[0]["model"]

    atomic_json_write(results_dir / "leaderboard.json", leaderboard)
    logger.info(f"Leaderboard saved with {len(entries)} entries")
    return leaderboard


def generate_ablation_leaderboard(results_base_dir: Path) -> Dict:
    """
    Generate cross-config comparison leaderboard.
    Scans config subdirectories (ppg/, ppg_ecg/, etc.) and produces
    a unified comparison showing how each model performs across configs.
    """
    import json

    all_entries = []
    configs_found = []

    for config_dir in sorted(results_base_dir.iterdir()):
        if not config_dir.is_dir():
            continue
        config_name = config_dir.name
        leaderboard_path = config_dir / "leaderboard.json"

        if not leaderboard_path.exists():
            continue

        configs_found.append(config_name)
        with open(leaderboard_path) as f:
            lb = json.load(f)

        for entry in lb.get("entries", []):
            entry["config"] = config_name
            all_entries.append(entry)

    if not all_entries:
        logger.warning("No config leaderboards found for ablation comparison")
        return {}

    # Sort by config, then MAE
    all_entries.sort(key=lambda x: (x["config"], x["MAE"]))

    # Build comparison: for each model+target, show MAE across configs
    comparison = {}
    for entry in all_entries:
        key = f"{entry['model']}_{entry['target']}"
        if key not in comparison:
            comparison[key] = {"model": entry["model"], "target": entry["target"]}
        comparison[key][f"{entry['config']}_MAE"] = entry["MAE"]
        comparison[key][f"{entry['config']}_AAMI"] = entry["AAMI"]

    ablation_lb = {
        "configs": configs_found,
        "entries": all_entries,
        "comparison": list(comparison.values()),
    }

    atomic_json_write(results_base_dir / "ablation_leaderboard.json", ablation_lb)
    logger.info(f"Ablation leaderboard: {len(configs_found)} configs, {len(all_entries)} total entries")
    return ablation_lb
