#!/usr/bin/env python3
"""
Deep learning training entry point.
Trains DL models on raw signal windows for BP estimation.

Usage:
    python scripts/train_dl_models.py --config ppg --target sbp --model cnn --resume
    python scripts/train_dl_models.py --config ppg_ecg_abp --target dbp --model transformer
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer
from src.dl_models import create_dl_model, DL_MODELS
from src.dl_data import BPSignalDataset, SIGNAL_CONFIGS
from src.dl_training import train_dl_model, predict_dl
from src.evaluation import evaluate_model

import json
import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description="Train DL models for BP estimation")
    parser.add_argument("--config", type=str, required=True, choices=list(SIGNAL_CONFIGS.keys()))
    parser.add_argument("--target", type=str, required=True, choices=["sbp", "dbp"])
    parser.add_argument("--model", type=str, required=True, choices=list(DL_MODELS.keys()))
    parser.add_argument("--subset-dir", type=str, default="data/subsets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(name="bp_dl_train")

    subset_dir = Path(args.subset_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_dir = Path(args.results_dir) / args.config

    n_channels = len(SIGNAL_CONFIGS[args.config])
    logger.info(f"Config: {args.config} ({n_channels} channels)")
    logger.info(f"Model: {args.model}, Target: {args.target}")

    # Load datasets
    train_path = subset_dir / "Train_Subset.npz"
    test_path = subset_dir / "CalFree_Test_Subset.npz"

    with timer("Loading training data", logger):
        train_dataset = BPSignalDataset(str(train_path), args.config, args.target, normalize=False)
        scaler_stats = train_dataset.compute_scaler_stats()
        train_dataset.scaler_stats = scaler_stats
        train_dataset.normalize = True

    with timer("Loading test data", logger):
        test_dataset = BPSignalDataset(str(test_path), args.config, args.target,
                                       normalize=True, scaler_stats=scaler_stats)

    # Save scaler stats
    scaler_path = checkpoint_dir / f"dl_scaler_{args.config}_{args.target}.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model = create_dl_model(args.model, in_channels=n_channels)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    with timer(f"Training {args.model}", logger):
        result = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            model_name=args.model,
            target=args.target,
            config=args.config,
            checkpoint_dir=checkpoint_dir,
            max_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            resume=args.resume,
        )

    # Evaluate
    with timer("Evaluating", logger):
        y_pred = predict_dl(result["model"], test_loader)
        y_true = test_dataset.targets
        results_dir.mkdir(parents=True, exist_ok=True)
        evaluate_model(f"dl_{args.model}", y_true, y_pred, args.target, results_dir)

    logger.info(f"Done. Best val loss: {result['best_val_loss']:.4f}, Epochs: {result['epochs']}")


if __name__ == "__main__":
    main()
