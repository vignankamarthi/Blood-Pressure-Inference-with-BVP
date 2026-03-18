#!/usr/bin/env python3
"""
Standalone evaluation script.
Loads trained models and runs full evaluation suite on test set.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer
from src.data_loader import load_cached_features, prepare_train_test
from src.evaluation import evaluate_model, generate_leaderboard, stratified_evaluation, bland_altman

import joblib
import json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--features", type=str, default="data/features/features.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    logger = setup_logging(name="bp_evaluate")
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir)

    # Load features
    cache_path = checkpoint_dir / "features_cache.parquet"
    df = load_cached_features(Path(args.features), cache_path)

    # Prepare test data
    X_train_sbp, y_train_sbp, X_test_sbp, y_test_sbp, _ = prepare_train_test(df, "sbp", checkpoint_dir)
    X_train_dbp, y_train_dbp, X_test_dbp, y_test_dbp, _ = prepare_train_test(df, "dbp", checkpoint_dir)
    X_test = X_test_sbp

    # Find all trained models
    models_dir = checkpoint_dir / "models"
    if not models_dir.exists():
        logger.error(f"No models found in {models_dir}")
        sys.exit(1)

    all_predictions_sbp = {}
    all_predictions_dbp = {}

    for model_file in sorted(models_dir.glob("*.pkl")):
        name = model_file.stem
        # Skip intermediate checkpoints (rf_sbp_n100, etc.)
        if "_n" in name and name.split("_n")[-1].isdigit():
            continue

        parts = name.rsplit("_", 1)
        if len(parts) != 2:
            continue
        model_name, target = parts

        logger.info(f"Evaluating {model_name} ({target})...")
        model = joblib.load(model_file)
        y_pred = model.predict(X_test)

        y_test = y_test_sbp if target == "sbp" else y_test_dbp
        evaluate_model(model_name, y_test, y_pred, target, results_dir)

        if target == "sbp":
            all_predictions_sbp[model_name] = y_pred
        else:
            all_predictions_dbp[model_name] = y_pred

    # Stratified evaluation (for models that have both SBP and DBP)
    common_models = set(all_predictions_sbp.keys()) & set(all_predictions_dbp.keys())
    for model_name in common_models:
        strat = stratified_evaluation(
            y_test_sbp, all_predictions_sbp[model_name],
            y_test_dbp, all_predictions_dbp[model_name])
        strat_path = results_dir / f"{model_name}_stratified.json"
        with open(strat_path, "w") as f:
            json.dump(strat, f, indent=2, default=str)
        logger.info(f"  Stratified evaluation saved to {strat_path}")

    # Leaderboard
    generate_leaderboard(results_dir)
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
