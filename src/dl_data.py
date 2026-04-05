"""
PyTorch Dataset for raw signal windows from PulseDB.
Loads segments and returns (signal_tensor, target) pairs.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional


# Ablation config -> which signal keys to include.
# ABP configs (ppg_abp, ppg_ecg_abp) dropped 2026-04-05 due to feature-source leakage:
# SBP/DBP labels are peaks/troughs of the ABP waveform, so using ABP as a predictive
# input leaks the target. See CLEANUP_PLAN.md at repo root for full audit.
SIGNAL_CONFIGS = {
    "ppg": ["ppg"],
    "ppg_ecg": ["ppg", "ecg"],
}


class BPSignalDataset(Dataset):
    """
    PyTorch Dataset for blood pressure estimation from raw signals.

    Each item returns:
        signal: (channels, 1250) float tensor
        target: scalar float (SBP or DBP)
    """

    def __init__(
        self,
        subset_path: str,
        config: str,
        target: str,
        normalize: bool = True,
        scaler_stats: Optional[Dict] = None,
    ):
        """
        Args:
            subset_path: Path to .npz subset file
            config: Ablation config ('ppg' or 'ppg_ecg')
            target: 'sbp' or 'dbp'
            normalize: Whether to normalize signals per-channel
            scaler_stats: Dict with 'mean' and 'std' per channel (fit on training)
        """
        if config not in SIGNAL_CONFIGS:
            raise ValueError(f"Unknown config: {config}. Available: {list(SIGNAL_CONFIGS.keys())}")

        data = np.load(subset_path, allow_pickle=True)
        self.target_key = target
        self.config = config
        self.signal_keys = SIGNAL_CONFIGS[config]

        # Load targets
        self.targets = data[target].astype(np.float32)

        # Load signals -- each is an array of variable-length arrays
        # For raw signal DL, we need the original waveforms, not extracted features
        # The subset .npz should contain ppg_signals, ecg_signals, abp_signals
        self.signals = []
        for key in self.signal_keys:
            signal_key = f"{key}_signals"
            if signal_key in data:
                self.signals.append(data[signal_key])
            else:
                raise KeyError(f"Signal key '{signal_key}' not found in {subset_path}. "
                               f"Available: {list(data.keys())}")

        self.n_samples = len(self.targets)
        self.n_channels = len(self.signal_keys)
        self.normalize = normalize
        self.scaler_stats = scaler_stats

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Stack channels: (channels, timesteps)
        channels = []
        for ch_signals in self.signals:
            sig = np.array(ch_signals[idx], dtype=np.float32)
            # Pad or truncate to 1250
            if len(sig) < 1250:
                sig = np.pad(sig, (0, 1250 - len(sig)))
            elif len(sig) > 1250:
                sig = sig[:1250]
            channels.append(sig)

        signal = np.stack(channels, axis=0)  # (channels, 1250)

        if self.normalize and self.scaler_stats is not None:
            for ch in range(signal.shape[0]):
                mean = self.scaler_stats['mean'][ch]
                std = self.scaler_stats['std'][ch]
                if std > 1e-8:
                    signal[ch] = (signal[ch] - mean) / std

        target = self.targets[idx]

        return torch.from_numpy(signal), torch.tensor(target, dtype=torch.float32)

    def compute_scaler_stats(self) -> Dict:
        """Compute per-channel mean and std for normalization. Call on training set only."""
        all_means = []
        all_stds = []
        for ch_idx in range(self.n_channels):
            ch_data = []
            for i in range(min(self.n_samples, 10000)):  # Sample for efficiency
                sig = np.array(self.signals[ch_idx][i], dtype=np.float64)
                ch_data.extend(sig.tolist())
            ch_data = np.array(ch_data)
            all_means.append(float(np.mean(ch_data)))
            all_stds.append(float(np.std(ch_data)))
        return {'mean': all_means, 'std': all_stds}
