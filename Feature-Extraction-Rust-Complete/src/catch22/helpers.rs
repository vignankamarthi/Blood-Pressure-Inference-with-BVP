/// Shared helper functions for Catch22 features.

/// Compute the full autocorrelation function of a signal (normalized).
/// Returns ACF for lags 0..N-1 (ACF[0] = 1.0).
pub fn autocorrelation_fn(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let mean = signal.iter().sum::<f64>() / n as f64;
    let var: f64 = signal.iter().map(|v| (v - mean).powi(2)).sum();

    if var < 1e-15 {
        return vec![1.0; n];
    }

    let mut acf = Vec::with_capacity(n);
    for lag in 0..n {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += (signal[i] - mean) * (signal[i + lag] - mean);
        }
        acf.push(sum / var);
    }
    acf
}

/// Find the first minimum of the autocorrelation function.
/// Returns the lag at first local minimum (or N-1 if none found).
pub fn first_min_ac(acf: &[f64]) -> usize {
    if acf.len() < 3 {
        return 1;
    }
    for i in 1..acf.len() - 1 {
        if acf[i] < acf[i - 1] && acf[i] <= acf[i + 1] {
            return i;
        }
    }
    acf.len() - 1
}

/// Z-score normalize a signal to mean=0, std=1.
pub fn zscore(signal: &[f64]) -> Vec<f64> {
    let n = signal.len() as f64;
    let mean = signal.iter().sum::<f64>() / n;
    let std = (signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
    if std < 1e-15 {
        return vec![0.0; signal.len()];
    }
    signal.iter().map(|v| (v - mean) / std).collect()
}

/// Successive differences of a signal.
pub fn successive_diffs(signal: &[f64]) -> Vec<f64> {
    signal.windows(2).map(|w| w[1] - w[0]).collect()
}

/// Histogram bin assignment (equal-width bins).
/// Returns (bin_edges, bin_counts, bin_assignments).
pub fn histogram(data: &[f64], n_bins: usize) -> (Vec<f64>, Vec<usize>) {
    if data.is_empty() || n_bins == 0 {
        return (vec![], vec![]);
    }
    let min_val = data.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-15 {
        let mut counts = vec![0; n_bins];
        counts[n_bins / 2] = data.len();
        let edges: Vec<f64> = (0..=n_bins).map(|i| min_val - 0.5 + i as f64 / n_bins as f64).collect();
        return (edges, counts);
    }

    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut counts = vec![0usize; n_bins];

    for &v in data {
        let mut bin = ((v - min_val) / bin_width) as usize;
        if bin >= n_bins {
            bin = n_bins - 1;
        }
        counts[bin] += 1;
    }

    let edges: Vec<f64> = (0..=n_bins).map(|i| min_val + i as f64 * bin_width).collect();
    (edges, counts)
}
