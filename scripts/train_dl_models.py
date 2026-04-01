#!/usr/bin/env python3
"""
Deep learning training entry point.
Trains ResNet-BiGRU or ResNet-1D on raw PPG waveforms for BP estimation.
Evaluates on multiple test sets (CalFree, CalBased, AAMI).

Usage:
    python scripts/train_dl_models.py --target sbp --model resnet_bigru --resume
    python scripts/train_dl_models.py --target dbp --model resnet --resume
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, timer
from src.dl_models import create_dl_model, DL_MODELS
from src.dl_data import BPSignalDataset
from src.dl_training import train_dl_model, predict_dl
from src.evaluation import evaluate_model

import json
import torch
from torch.utils.data import DataLoader

TEST_SET_FILES = {
    "CalFree": "CalFree_Test_Subset.npz",
    "CalBased": "CalBased_Test_Subset.npz",
    "AAMI": "AAMI_Test_Subset.npz",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train DL models for BP estimation")
    parser.add_argument("--config", type=str, default="ppg",
                        help="Signal config (default: ppg -- DL trains on raw PPG only)")
    parser.add_argument("--target", type=str, required=True, choices=["sbp", "dbp"])
    parser.add_argument("--model", type=str, required=True, choices=list(DL_MODELS.keys()))
    parser.add_argument("--subset-dir", type=str, default="data/subsets")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--test-sets", type=str, default="CalFree,CalBased,AAMI",
                        help="Comma-separated test sets to evaluate on")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    logger = setup_logging(name="bp_dl_train")

    subset_dir = Path(args.subset_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    results_base = Path(args.results_dir) / "dl"

    n_channels = 1  # DL trains on PPG only
    if args.config != "ppg":
        logger.warning(f"DL track uses PPG only. Ignoring config '{args.config}', using 'ppg'.")
    config = "ppg"

    logger.info(f"Model: {args.model}, Target: {args.target}, Config: {config}")

    # Load training data
    train_path = subset_dir / "Train_Subset.npz"
    val_path = subset_dir / "CalFree_Test_Subset.npz"

    with timer("Loading training data", logger):
        train_dataset = BPSignalDataset(str(train_path), config, args.target, normalize=False)
        scaler_stats = train_dataset.compute_scaler_stats()
        train_dataset.scaler_stats = scaler_stats
        train_dataset.normalize = True

    with timer("Loading validation data (CalFree)", logger):
        val_dataset = BPSignalDataset(str(val_path), config, args.target,
                                      normalize=True, scaler_stats=scaler_stats)

    # Save scaler stats for reproducibility
    scaler_path = checkpoint_dir / f"dl_scaler_{config}_{args.target}.json"
    scaler_path.parent.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "w") as f:
        json.dump(scaler_stats, f)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Create model
    model = create_dl_model(args.model, in_channels=n_channels)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train (uses CalFree as validation for early stopping)
    with timer(f"Training {args.model}", logger):
        result = train_dl_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            model_name=args.model,
            target=args.target,
            config=config,
            checkpoint_dir=checkpoint_dir,
            max_epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            resume=args.resume,
        )

    logger.info(f"Training done. Best val loss: {result['best_val_loss']:.4f}, Epochs: {result['epochs']}")

    # Evaluate on all requested test sets
    best_model = result["model"]
    for test_name in args.test_sets.split(","):
        test_name = test_name.strip()
        if test_name not in TEST_SET_FILES:
            logger.warning(f"Unknown test set '{test_name}', skipping")
            continue

        test_path = subset_dir / TEST_SET_FILES[test_name]
        if not test_path.exists():
            logger.warning(f"Test set {test_name} not found at {test_path}, skipping")
            continue

        with timer(f"Evaluating on {test_name}", logger):
            test_dataset = BPSignalDataset(str(test_path), config, args.target,
                                           normalize=True, scaler_stats=scaler_stats)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                                     shuffle=False, num_workers=4, pin_memory=True)
            y_pred = predict_dl(best_model, test_loader)
            y_true = test_dataset.targets

            test_results_dir = results_base / test_name
            test_results_dir.mkdir(parents=True, exist_ok=True)
            evaluate_model(f"dl_{args.model}", y_true, y_pred, args.target, test_results_dir)
            logger.info(f"  {test_name}: saved to {test_results_dir}")

    logger.info("All evaluations complete.")


if __name__ == "__main__":
    main()
