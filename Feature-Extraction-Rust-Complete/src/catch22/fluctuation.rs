/// Fluctuation analysis-based Catch22 features (DFA and RS range).

/// SC_FluctAnal_2_rsrangefit_50_1_logi_prop_r1:
/// Rescaled range fluctuation analysis. Proportion of segments
/// where the RS statistic indicates long-range dependence.
pub fn sc_fluctanal_rsrangefit(signal: &[f64]) -> f64 {
    fluctuation_analysis(signal, false)
}

/// SC_FluctAnal_2_dfa_50_1_2_logi_prop_r1:
/// Detrended Fluctuation Analysis. Returns the proportion of
/// segments with DFA exponent indicating long-range correlation.
pub fn sc_fluctanal_dfa(signal: &[f64]) -> f64 {
    fluctuation_analysis(signal, true)
}

/// Core fluctuation analysis implementation.
/// dfa=true for DFA, dfa=false for RS range.
fn fluctuation_analysis(signal: &[f64], dfa: bool) -> f64 {
    let n = signal.len();
    if n < 50 {
        return 0.0;
    }

    // Z-score
    let mean = signal.iter().sum::<f64>() / n as f64;
    let std = (signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    if std < 1e-15 {
        return 0.0;
    }
    let z: Vec<f64> = signal.iter().map(|v| (v - mean) / std).collect();

    // Cumulative sum (profile)
    let mut profile = vec![0.0; n];
    let mut cum = 0.0;
    for i in 0..n {
        cum += z[i];
        profile[i] = cum;
    }

    // Generate segment sizes (log-spaced from 4 to n/4)
    let min_seg = 4;
    let max_seg = n / 4;
    if max_seg < min_seg {
        return 0.0;
    }

    let n_sizes = 20.min((max_seg - min_seg + 1) as usize);
    let log_min = (min_seg as f64).ln();
    let log_max = (max_seg as f64).ln();
    let mut seg_sizes: Vec<usize> = (0..n_sizes)
        .map(|i| {
            let log_s = log_min + i as f64 * (log_max - log_min) / (n_sizes - 1).max(1) as f64;
            log_s.exp().round() as usize
        })
        .collect();
    seg_sizes.dedup();
    seg_sizes.retain(|&s| s >= min_seg && s <= max_seg);

    if seg_sizes.len() < 3 {
        return 0.0;
    }

    let mut log_sizes = Vec::new();
    let mut log_flucts = Vec::new();

    for &seg_size in &seg_sizes {
        let n_segs = n / seg_size;
        if n_segs == 0 {
            continue;
        }

        let mut fluct_sum = 0.0;
        let mut count = 0;

        for s in 0..n_segs {
            let start = s * seg_size;
            let end = start + seg_size;
            let segment = &profile[start..end];

            let f = if dfa {
                // DFA: detrend with linear fit, compute RMS of residuals
                let (slope, intercept) = linear_fit(segment);
                let rms: f64 = segment.iter().enumerate()
                    .map(|(i, &v)| (v - (slope * i as f64 + intercept)).powi(2))
                    .sum::<f64>() / seg_size as f64;
                rms.sqrt()
            } else {
                // RS range: (max - min) of segment / std of original segment
                let seg_z = &z[start..end];
                let seg_mean = seg_z.iter().sum::<f64>() / seg_size as f64;
                let seg_std = (seg_z.iter().map(|v| (v - seg_mean).powi(2)).sum::<f64>() / seg_size as f64).sqrt();
                if seg_std < 1e-15 { 0.0 } else {
                    let seg_min = segment.iter().copied().fold(f64::INFINITY, f64::min);
                    let seg_max = segment.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    (seg_max - seg_min) / seg_std
                }
            };

            if f > 0.0 {
                fluct_sum += f;
                count += 1;
            }
        }

        if count > 0 {
            let avg_fluct = fluct_sum / count as f64;
            log_sizes.push((seg_size as f64).ln());
            log_flucts.push(avg_fluct.ln());
        }
    }

    if log_sizes.len() < 3 {
        return 0.0;
    }

    // Linear regression on log-log plot
    let (slope, _) = linear_fit_xy(&log_sizes, &log_flucts);

    // Proportion of positive residuals from the fit
    let n_fit = log_sizes.len() as f64;
    let mean_x = log_sizes.iter().sum::<f64>() / n_fit;
    let mean_y = log_flucts.iter().sum::<f64>() / n_fit;

    let positive_residuals = log_sizes.iter().zip(log_flucts.iter())
        .filter(|(&x, &y)| y > slope * (x - mean_x) + mean_y)
        .count();

    positive_residuals as f64 / log_sizes.len() as f64
}

/// Simple linear regression: y = slope * x + intercept
fn linear_fit(y: &[f64]) -> (f64, f64) {
    let n = y.len() as f64;
    let x_mean = (n - 1.0) / 2.0;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (i, &yi) in y.iter().enumerate() {
        let xi = i as f64;
        num += (xi - x_mean) * (yi - y_mean);
        den += (xi - x_mean).powi(2);
    }

    if den.abs() < 1e-15 {
        return (0.0, y_mean);
    }

    let slope = num / den;
    let intercept = y_mean - slope * x_mean;
    (slope, intercept)
}

/// Linear regression with explicit x and y vectors.
fn linear_fit_xy(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let x_mean = x.iter().sum::<f64>() / n;
    let y_mean = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        num += (xi - x_mean) * (yi - y_mean);
        den += (xi - x_mean).powi(2);
    }

    if den.abs() < 1e-15 {
        return (0.0, y_mean);
    }

    let slope = num / den;
    let intercept = y_mean - slope * x_mean;
    (slope, intercept)
}
