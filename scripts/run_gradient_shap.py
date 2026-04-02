#!/usr/bin/env python3
"""
GradientSHAP temporal interpretability for DL models on raw PPG.

Computes per-timestep attribution values showing which regions of the
10-second PPG waveform drive the model's BP prediction. Uses captum's
GradientShap with zero baselines.

Novel contribution: no prior work applies GradientSHAP directly to raw
PPG temporal maps for blood pressure estimation.

Usage:
    python scripts/run_gradient_shap.py --model-name resnet_bigru --target sbp
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging
from src.dl_models import create_dl_model
from src.dl_data import BPSignalDataset
from src.dl_training import DEVICE

logger = None


def load_trained_model(model_name: str, checkpoint_dir: Path, config: str, target: str):
    """Load best.pt checkpoint for a trained model."""
    model = create_dl_model(model_name, in_channels=1)
    best_path = checkpoint_dir / f"dl_{model_name}_{config}_{target}" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"No checkpoint at {best_path}")

    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(DEVICE)
    # Keep model in train mode for GradientSHAP -- cuDNN RNN backward requires it.
    # Disable dropout manually instead of using eval().
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d)):
            module.eval()
        if isinstance(module, torch.nn.BatchNorm1d):
            module.eval()
    logger.info(f"Loaded model from {best_path}")
    return model


def compute_attributions(model, test_dataset, n_samples: int = 200, batch_size: int = 10):
    """Compute GradientSHAP attributions for test samples."""
    from captum.attr import GradientShap

    gs = GradientShap(model)

    # Zero baseline (represents "no signal")
    baseline = torch.zeros(1, 1, 1250).to(DEVICE)

    all_attributions = []
    all_predictions = []
    n = min(n_samples, len(test_dataset))

    logger.info(f"Computing GradientSHAP for {n} samples (batch_size={batch_size})...")

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        signals = []
        for i in range(start, end):
            sig, _ = test_dataset[i]
            signals.append(sig)

        input_batch = torch.stack(signals).to(DEVICE)
        input_batch.requires_grad = True

        # Expand baseline to match batch size
        baselines = baseline.expand(input_batch.shape[0], -1, -1)

        attributions = gs.attribute(
            input_batch,
            baselines=baselines,
            n_samples=50,
            stdevs=0.0001,
        )

        # Get predictions for context
        with torch.no_grad():
            preds = model(input_batch)

        all_attributions.append(attributions.detach().cpu().numpy())
        all_predictions.append(preds.detach().cpu().numpy().flatten())

        if (start + batch_size) % 50 == 0 or end == n:
            logger.info(f"  Processed {end}/{n} samples")

    attributions_array = np.concatenate(all_attributions, axis=0)  # (n, 1, 1250)
    predictions_array = np.concatenate(all_predictions, axis=0)  # (n,)
    return attributions_array, predictions_array


def generate_temporal_plot(attributions: np.ndarray, output_path: Path,
                          model_name: str, target: str):
    """Generate temporal importance plot from mean absolute attributions."""
    # attributions shape: (n_samples, 1, 1250) -> squeeze channel dim
    attr_2d = attributions.squeeze(1)  # (n_samples, 1250)
    mean_abs_attr = np.mean(np.abs(attr_2d), axis=0)  # (1250,)

    # Time axis: 1250 samples at 125 Hz = 10 seconds
    time_s = np.arange(1250) / 125.0

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(time_s, mean_abs_attr, alpha=0.3, color='steelblue')
    ax.plot(time_s, mean_abs_attr, color='steelblue', linewidth=1)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Mean |Attribution|', fontsize=12)
    ax.set_title(f'GradientSHAP Temporal Importance -- {model_name} ({target.upper()})', fontsize=14)
    ax.set_xlim(0, 10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved temporal importance plot to {output_path}")


def main():
    global logger

    parser = argparse.ArgumentParser(description="GradientSHAP analysis on trained DL model")
    parser.add_argument("--model-name", type=str, default="resnet_bigru")
    parser.add_argument("--target", type=str, required=True, choices=["sbp", "dbp"])
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--subset-dir", type=str, default="data/subsets")
    parser.add_argument("--test-set", type=str, default="CalFree",
                        help="Which test set to sample from")
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="results/dl/gradient_shap")
    args = parser.parse_args()

    logger = setup_logging(name="gradient_shap")

    checkpoint_dir = Path(args.checkpoint_dir)
    subset_dir = Path(args.subset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_trained_model(args.model_name, checkpoint_dir, "ppg", args.target)

    # Load scaler stats from training
    scaler_path = checkpoint_dir / f"dl_scaler_ppg_{args.target}.json"
    with open(scaler_path) as f:
        scaler_stats = json.load(f)

    # Load test dataset
    test_set_files = {
        "CalFree": "CalFree_Test_Subset.npz",
        "CalBased": "CalBased_Test_Subset.npz",
        "AAMI": "AAMI_Test_Subset.npz",
    }
    test_path = subset_dir / test_set_files[args.test_set]
    test_dataset = BPSignalDataset(str(test_path), "ppg", args.target,
                                   normalize=True, scaler_stats=scaler_stats)

    logger.info(f"Test set: {args.test_set} ({len(test_dataset)} samples)")

    # Compute attributions
    attributions, predictions = compute_attributions(
        model, test_dataset, n_samples=args.n_samples
    )

    # Save raw attributions
    attr_path = output_dir / f"{args.model_name}_{args.target}_attributions.npy"
    np.save(str(attr_path), attributions)
    logger.info(f"Saved attributions: {attr_path} (shape: {attributions.shape})")

    # Generate temporal importance plot
    plot_path = output_dir / f"{args.model_name}_{args.target}_temporal_importance.png"
    generate_temporal_plot(attributions, plot_path, args.model_name, args.target)

    # Summary statistics
    attr_2d = attributions.squeeze(1)
    mean_abs = np.mean(np.abs(attr_2d), axis=0)
    top_indices = np.argsort(mean_abs)[-10:][::-1]

    summary = {
        "model": args.model_name,
        "target": args.target,
        "test_set": args.test_set,
        "n_samples": len(attributions),
        "mean_prediction": float(np.mean(predictions)),
        "std_prediction": float(np.std(predictions)),
        "top_10_timesteps": [int(i) for i in top_indices],
        "top_10_times_seconds": [float(i / 125.0) for i in top_indices],
        "top_10_mean_abs_attribution": [float(mean_abs[i]) for i in top_indices],
        "global_mean_abs_attribution": float(np.mean(mean_abs)),
    }

    summary_path = output_dir / f"{args.model_name}_{args.target}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary: {summary_path}")

    logger.info("GradientSHAP analysis complete.")


if __name__ == "__main__":
    main()
