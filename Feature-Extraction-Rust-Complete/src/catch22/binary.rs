/// Binary/transition-based Catch22 features.
use super::helpers;

/// SB_BinaryStats_mean_longstretch1: Longest stretch of 1s in binary signal (above mean).
pub fn sb_binarystats_mean_longstretch1(signal: &[f64]) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let mean = signal.iter().sum::<f64>() / signal.len() as f64;
    let binary: Vec<bool> = signal.iter().map(|&v| v > mean).collect();

    let mut max_stretch = 0;
    let mut current = 0;
    for &b in &binary {
        if b {
            current += 1;
            max_stretch = max_stretch.max(current);
        } else {
            current = 0;
        }
    }
    max_stretch as f64
}

/// SB_TransitionMatrix_3ac_sumdiagcov: Trace of the covariance of the transition
/// matrix on a 3-letter alphabet.
pub fn sb_transitionmatrix_3ac_sumdiagcov(signal: &[f64]) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }
    let z = helpers::zscore(signal);

    // Quantize to 3 states: 0 (below -1/3 quantile), 1 (middle), 2 (above)
    let mut sorted = z.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q1 = sorted[n / 3];
    let q2 = sorted[2 * n / 3];

    let quantized: Vec<usize> = z.iter().map(|&v| {
        if v < q1 { 0 } else if v < q2 { 1 } else { 2 }
    }).collect();

    // Build 3x3 transition matrix
    let mut trans = [[0.0_f64; 3]; 3];
    let mut row_sums = [0.0_f64; 3];
    for w in quantized.windows(2) {
        trans[w[0]][w[1]] += 1.0;
        row_sums[w[0]] += 1.0;
    }

    // Normalize rows
    for i in 0..3 {
        if row_sums[i] > 0.0 {
            for j in 0..3 {
                trans[i][j] /= row_sums[i];
            }
        }
    }

    // Sum of diagonal of covariance matrix of transition matrix columns
    // For each column j, compute variance across rows i
    let mut sum_diag_cov = 0.0;
    for j in 0..3 {
        let col: Vec<f64> = (0..3).map(|i| trans[i][j]).collect();
        let mean = col.iter().sum::<f64>() / 3.0;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / 3.0;
        sum_diag_cov += var;
    }
    sum_diag_cov
}

/// SB_MotifThree_quantile_hh: Proportion of "high-high" motifs in 3-letter alphabet.
pub fn sb_motifthree_quantile_hh(signal: &[f64]) -> f64 {
    if signal.len() < 3 {
        return 0.0;
    }
    let z = helpers::zscore(signal);
    let mut sorted = z.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let q1 = sorted[n / 3];
    let q2 = sorted[2 * n / 3];

    let quantized: Vec<usize> = z.iter().map(|&v| {
        if v < q1 { 0 } else if v < q2 { 1 } else { 2 }
    }).collect();

    // Count "hh" motifs (two consecutive highs = state 2)
    let mut hh_count = 0;
    let total = quantized.len() - 1;
    for w in quantized.windows(2) {
        if w[0] == 2 && w[1] == 2 {
            hh_count += 1;
        }
    }
    hh_count as f64 / total as f64
}
