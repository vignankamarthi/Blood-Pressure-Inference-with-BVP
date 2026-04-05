#!/usr/bin/env python3
"""
Training entry point for Blood Pressure Inference pipeline.
Orchestrates: load data -> tune (optional) -> train -> evaluate -> save.

Usage:
    python scripts/train_models.py --resume --tune
    python scripts/train_models.py --fresh
    python scripts/train_models.py --resume --models ridge,rf --targets sbp
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer
from src.data_loader import load_cached_features, prepare_train_test, get_feature_columns
from src.models import train_all, load_status
from src.tuning import tune_all
from src.evaluation import evaluate_model, generate_leaderboard


def parse_args():
    parser = argparse.ArgumentParser(description="Train BP estimation models")
    parser.add_argument("--config", type=str, default="ppg",
                        choices=["ppg", "ppg_ecg"],
                        help="Ablation config (signal combination). ABP configs dropped 2026-04-05 (feature-source leakage audit).")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to feature CSV (auto-derived from --config if not set)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Base directory for checkpoints (namespaced by config)")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Base directory for results (namespaced by config)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--fresh", action="store_true",
                        help="Start fresh (delete checkpoints)")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning before training")
    parser.add_argument("--models", type=str, default="ridge,dt,rf,xgb,lgbm",
                        help="Comma-separated model names")
    parser.add_argument("--targets", type=str, default="sbp,dbp",
                        help="Comma-separated targets")
    parser.add_argument("--n-trials", type=int, default=100,
                        help="Number of Optuna trials per model")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(name="bp_pipeline")

    # Auto-derive feature path from config if not explicitly set
    if args.features is None:
        feature_path = Path(f"data/features/{args.config}/features_labeled.csv")
    else:
        feature_path = Path(args.features)

    # Namespace checkpoints and results by config
    checkpoint_dir = Path(args.checkpoint_dir) / args.config
    results_dir = Path(args.results_dir) / args.config
    models_to_train = args.models.split(",")
    targets = args.targets.split(",")

    logger.info(f"Config: {args.config}")
    logger.info(f"Features: {feature_path}")

    # Safety check: --resume and --fresh are mutually exclusive
    if args.resume and args.fresh:
        logger.error("Cannot use both --resume and --fresh")
        sys.exit(1)

    # Default: error if checkpoints exist without explicit flag
    if not args.resume and not args.fresh and checkpoint_dir.exists():
        status = load_status(checkpoint_dir)
        if status.get("completed"):
            logger.error("Checkpoints exist. Use --resume to continue or --fresh to restart.")
            sys.exit(1)

    if args.fresh and checkpoint_dir.exists():
        import shutil
        logger.warning("Deleting all checkpoints (--fresh)")
        shutil.rmtree(checkpoint_dir)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load features
    cache_path = checkpoint_dir / "features_cache.parquet"
    with timer("Loading features", logger):
        df = load_cached_features(feature_path, cache_path)

    logger.info(f"Feature matrix: {df.shape}")
    logger.info(f"Feature columns: {len(get_feature_columns(df))}")

    # Prepare data for SBP and DBP
    with timer("Preparing train/test splits", logger):
        X_train_sbp, y_train_sbp, X_test_sbp, y_test_sbp, scaler_sbp = prepare_train_test(
            df, "sbp", checkpoint_dir, force_refit=args.fresh)
        X_train_dbp, y_train_dbp, X_test_dbp, y_test_dbp, scaler_dbp = prepare_train_test(
            df, "dbp", checkpoint_dir, force_refit=args.fresh)

    # Use same X (features are identical, only target differs)
    X_train = X_train_sbp
    X_test = X_test_sbp

    # Hyperparameter tuning
    if args.tune:
        with timer("Hyperparameter tuning", logger):
            best_params = tune_all(
                X_train, y_train_sbp, y_train_dbp,
                checkpoint_dir, models_to_tune=models_to_train,
                n_trials=args.n_trials)
    else:
        # Load saved params or use defaults
        best_params = {}
        params_dir = checkpoint_dir / "best_params"
        if params_dir.exists():
            import json
            for f in params_dir.glob("*.json"):
                with open(f) as fp:
                    data = json.load(fp)
                best_params[f"{data['model']}_{data['target']}"] = data["best_params"]
            logger.info(f"Loaded {len(best_params)} saved parameter sets")
        else:
            logger.info("No saved params, using defaults")
            for m in models_to_train:
                for t in targets:
                    best_params[f"{m}_{t}"] = {}

    # Train
    with timer("Model training", logger):
        trained_models = train_all(
            X_train, y_train_sbp, y_train_dbp,
            best_params, checkpoint_dir,
            models_to_train=models_to_train,
            resume=args.resume)

    # Evaluate
    with timer("Evaluation", logger):
        for key, model in trained_models.items():
            model_name, target = key.rsplit("_", 1)
            y_test = y_test_sbp if target == "sbp" else y_test_dbp
            y_pred = model.predict(X_test)
            evaluate_model(model_name, y_test, y_pred, target, results_dir)

    # Leaderboard
    leaderboard = generate_leaderboard(results_dir)
    logger.info("=" * 60)
    logger.info("LEADERBOARD")
    logger.info("=" * 60)
    for entry in leaderboard["entries"]:
        logger.info(f"  {entry['model']:8s} ({entry['target']:3s}): "
                     f"MAE={entry['MAE']:.2f}, RMSE={entry['RMSE']:.2f}, "
                     f"R2={entry['R2']:.4f}, AAMI={entry['AAMI']}, BHS={entry['BHS']}")

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
