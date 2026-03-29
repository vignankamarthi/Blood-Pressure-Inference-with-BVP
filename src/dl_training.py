"""
PyTorch training loop with exhaustive checkpointing.
Saves model + optimizer state every epoch. Early stopping. LR scheduling.
Resume from any checkpoint.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import atomic_json_write

logger = logging.getLogger("bp_pipeline")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_dl_checkpoint(
    model: nn.Module, optimizer, epoch: int, loss: float,
    best_loss: float, path: Path
):
    """Save model + optimizer state for resuming."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'best_loss': best_loss,
    }, path)


def load_dl_checkpoint(model: nn.Module, optimizer, path: Path):
    """Load checkpoint and return (epoch, best_loss)."""
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    return ckpt['epoch'], ckpt.get('best_loss', float('inf'))


def train_dl_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    model_name: str,
    target: str,
    config: str,
    checkpoint_dir: Path,
    max_epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 20,
    resume: bool = False,
) -> Dict:
    """
    Train a DL model with full checkpointing.

    Returns dict with training history and best metrics.
    """
    model = model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    ckpt_dir = checkpoint_dir / f"dl_{model_name}_{config}_{target}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "latest.pt"
    best_path = ckpt_dir / "best.pt"

    start_epoch = 0
    best_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}

    # Resume
    if resume and ckpt_path.exists():
        start_epoch, best_loss = load_dl_checkpoint(model, optimizer, ckpt_path)
        logger.info(f"Resuming {model_name} ({config}/{target}) from epoch {start_epoch}")

        # Load history
        hist_path = ckpt_dir / "history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text())

    logger.info(f"Training {model_name} ({config}/{target}) on {DEVICE}")
    logger.info(f"  Epochs: {start_epoch}/{max_epochs}, LR: {lr}, Patience: {patience}")

    for epoch in range(start_epoch, max_epochs):
        # Train
        model.train()
        train_losses = []
        for signals, targets in train_loader:
            signals = signals.to(DEVICE)
            targets = targets.to(DEVICE).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)

        # Validate
        avg_val_loss = float('inf')
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for signals, targets in val_loader:
                    signals = signals.to(DEVICE)
                    targets = targets.to(DEVICE).unsqueeze(1)
                    outputs = model(signals)
                    loss = criterion(outputs, targets)
                    val_losses.append(loss.item())
            avg_val_loss = np.mean(val_losses)
        history['val_loss'].append(avg_val_loss)

        # LR scheduling
        scheduler.step(avg_val_loss)

        # Checkpoint every epoch
        save_dl_checkpoint(model, optimizer, epoch + 1, avg_val_loss, best_loss, ckpt_path)
        atomic_json_write(ckpt_dir / "history.json", history)

        # Best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_dl_checkpoint(model, optimizer, epoch + 1, avg_val_loss, best_loss, best_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"  Epoch {epoch+1}/{max_epochs}: "
                f"train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, "
                f"best={best_loss:.4f}, lr={current_lr:.2e}, patience={patience_counter}/{patience}"
            )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_path.exists():
        best_ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])

    # Save progress status
    atomic_json_write(ckpt_dir / "status.json", {
        "model": model_name,
        "config": config,
        "target": target,
        "epochs_completed": len(history['train_loss']),
        "best_val_loss": best_loss,
        "status": "completed",
    })

    return {
        "model": model,
        "history": history,
        "best_val_loss": best_loss,
        "epochs": len(history['train_loss']),
    }


def predict_dl(model: nn.Module, data_loader: DataLoader) -> np.ndarray:
    """Generate predictions from a trained DL model."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for signals, _ in data_loader:
            signals = signals.to(DEVICE)
            outputs = model(signals)
            all_preds.append(outputs.cpu().numpy().flatten())
    return np.concatenate(all_preds)
