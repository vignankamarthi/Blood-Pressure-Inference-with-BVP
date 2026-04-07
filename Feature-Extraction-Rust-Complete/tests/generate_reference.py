#!/usr/bin/env python3
"""
Generate reference values for TDD validation of Rust feature extraction.
Computes all 41 features using Python reference implementations:
  - pycatch22 for Catch22 features
  - ordpy/antropy for entropy features
  - numpy/scipy for statistical features

Saves to reference_values.json for Rust test loading.
"""

import json
import numpy as np

# Generate test signals
np.random.seed(42)

signals = {
    "sinusoid": np.sin(np.arange(256) * 0.1).tolist(),
    "random_walk": np.cumsum(np.random.randn(300)).tolist(),
    "white_noise": np.random.randn(400).tolist(),
    "chirp": np.sin(np.arange(500) * np.arange(500) * 0.0001).tolist(),
    "ppg_like": (
        np.sin(np.arange(1250) * 2 * np.pi / 125)  # ~1 Hz fundamental
        + 0.3 * np.sin(np.arange(1250) * 4 * np.pi / 125)  # 2nd harmonic
        + 0.1 * np.random.randn(1250)  # noise
    ).tolist(),
}


def compute_stats(signal):
    """Compute statistical features using numpy/scipy."""
    from scipy import stats as sp_stats
    x = np.array(signal)
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x, ddof=1)),
        "skewness": float(sp_stats.skew(x, bias=False)),
        "kurtosis": float(sp_stats.kurtosis(x, fisher=True, bias=False)),
        "rms": float(np.sqrt(np.mean(x**2))),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def compute_catch22(signal):
    """Compute Catch22 features using pycatch22."""
    try:
        import pycatch22
        result = pycatch22.catch22_all(signal)
        return dict(zip(result["names"], result["values"]))
    except ImportError:
        print("WARNING: pycatch22 not installed, skipping Catch22 reference values")
        return None


def compute_entropy(signal, dimension=7, tau=2):
    """Compute entropy features using ordpy."""
    try:
        import ordpy

        # Permutation entropy (normalized)
        pe = ordpy.permutation_entropy(signal, dx=dimension, taux=tau, base=2, normalized=True)

        # Statistical complexity
        hc = ordpy.complexity_entropy(signal, dx=dimension, taux=tau, base=2)
        # hc returns [H, C] where H=PE, C=complexity

        # Fisher information
        fi_result = ordpy.fisher_shannon(signal, dx=dimension, taux=tau)
        # Returns [PE, Fisher_I] or (PE, Fisher_I)

        # Renyi (q=1 -> Shannon limit)
        renyi_result = ordpy.renyi_complexity_entropy(signal, dx=dimension, taux=tau, q=1.0)

        # Tsallis (q=1 -> Shannon limit)
        tsallis_result = ordpy.tsallis_complexity_entropy(signal, dx=dimension, taux=tau, q=1.0)

        return {
            "permutation_entropy": float(pe) if not hasattr(pe, '__len__') else float(pe),
            "statistical_complexity": float(hc[1]) if hasattr(hc, '__len__') else float(hc),
            "fisher_shannon": float(fi_result[0]) if hasattr(fi_result, '__len__') else float(fi_result),
            "fisher_information": float(fi_result[1]) if hasattr(fi_result, '__len__') and len(fi_result) > 1 else 0.0,
            "renyi_pe": float(renyi_result[0]) if hasattr(renyi_result, '__len__') else float(renyi_result),
            "renyi_complexity": float(renyi_result[1]) if hasattr(renyi_result, '__len__') and len(renyi_result) > 1 else 0.0,
            "tsallis_pe": float(tsallis_result[0]) if hasattr(tsallis_result, '__len__') else float(tsallis_result),
            "tsallis_complexity": float(tsallis_result[1]) if hasattr(tsallis_result, '__len__') and len(tsallis_result) > 1 else 0.0,
        }
    except ImportError:
        print("WARNING: ordpy not installed, skipping entropy reference values")
        return None


def compute_sample_approximate_entropy(signal, m=2):
    """Compute sample and approximate entropy."""
    try:
        import antropy
        x = np.array(signal)
        se = antropy.sample_entropy(x, order=m)
        ae = antropy.app_entropy(x, order=m)
        return {
            "sample_entropy": float(se),
            "approximate_entropy": float(ae),
        }
    except ImportError:
        try:
            # Fallback to nolds
            import nolds
            x = np.array(signal)
            return {
                "sample_entropy": float(nolds.sampen(x, emb_dim=m)),
                "approximate_entropy": 0.0,  # nolds doesn't have ApEn
            }
        except ImportError:
            print("WARNING: antropy/nolds not installed, skipping sample/approx entropy")
            return None


def main():
    reference = {"signals": signals, "features": {}}

    for name, signal in signals.items():
        print(f"Computing features for '{name}' (length={len(signal)})...")
        features = {}

        # Stats
        stats = compute_stats(signal)
        features["stats"] = stats
        print(f"  Stats: mean={stats['mean']:.6f}, std={stats['std']:.6f}")

        # Catch22
        c22 = compute_catch22(signal)
        if c22 is not None:
            features["catch22"] = c22
            print(f"  Catch22: {len(c22)} features computed")

        # Entropy
        ent = compute_entropy(signal, dimension=7, tau=2)
        if ent is not None:
            features["entropy"] = ent
            print(f"  Entropy: PE={ent['permutation_entropy']:.6f}")

        # Sample/Approximate entropy
        sa_ent = compute_sample_approximate_entropy(signal)
        if sa_ent is not None:
            features["sample_approximate"] = sa_ent
            print(f"  SampEn={sa_ent['sample_entropy']:.6f}")

        reference["features"][name] = features

    # Save
    output_path = "tests/reference_values.json"
    with open(output_path, "w") as f:
        json.dump(reference, f, indent=2)
    print(f"\nSaved reference values to {output_path}")
    print(f"Signals: {list(signals.keys())}")


if __name__ == "__main__":
    main()
