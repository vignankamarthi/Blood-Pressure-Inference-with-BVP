/// Local prediction-based Catch22 features.

/// FC_LocalSimple_mean1_tauresrat: Ratio of first zero-crossing of residual ACF
/// to the ACF of the original, using 1-step mean predictor.
pub fn fc_localsimple_mean1_tauresrat(signal: &[f64], tau: usize) -> f64 {
    if signal.len() < 3 {
        return 0.0;
    }

    // 1-step mean predictor: predict x_t from x_{t-1}
    let residuals: Vec<f64> = signal.windows(2)
        .map(|w| w[1] - w[0])
        .collect();

    if residuals.is_empty() {
        return 0.0;
    }

    // ACF of residuals
    let res_acf = super::helpers::autocorrelation_fn(&residuals);
    let res_tau = super::helpers::first_min_ac(&res_acf);

    // Ratio
    let tau = tau.max(1);
    res_tau as f64 / tau as f64
}

/// FC_LocalSimple_mean3_stderr: Standard error of 3-step mean predictor residuals.
pub fn fc_localsimple_mean3_stderr(signal: &[f64]) -> f64 {
    if signal.len() < 4 {
        return 0.0;
    }

    // 3-step mean predictor: predict x_t from mean(x_{t-1}, x_{t-2}, x_{t-3})
    let mut residuals = Vec::with_capacity(signal.len() - 3);
    for i in 3..signal.len() {
        let pred = (signal[i - 1] + signal[i - 2] + signal[i - 3]) / 3.0;
        residuals.push(signal[i] - pred);
    }

    if residuals.is_empty() {
        return 0.0;
    }

    // Standard error
    let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
    let var = residuals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (residuals.len() as f64 - 1.0);
    var.sqrt()
}

/// IN_AutoMutualInfoStats_40_gaussian_fmmi: First minimum of automutual information
/// computed using Gaussian kernel estimation, up to lag 40.
pub fn in_automutualinfostats_40_gaussian_fmmi(signal: &[f64]) -> f64 {
    let z = super::helpers::zscore(signal);
    let n = z.len();
    let max_lag = 40.min(n / 2);

    if max_lag < 2 {
        return 1.0;
    }

    // Compute AMI for each lag using histogram method (5-bin approximation)
    let n_bins = 5;
    let min_val = z.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-15 {
        return 1.0;
    }
    let bin_width = range / n_bins as f64;

    let bin_of = |v: f64| -> usize {
        ((v - min_val) / bin_width).min(n_bins as f64 - 1.0) as usize
    };

    let mut ami_values = Vec::with_capacity(max_lag);

    for lag in 1..=max_lag {
        let n_pairs = n - lag;
        let mut joint = vec![vec![0usize; n_bins]; n_bins];
        for i in 0..n_pairs {
            joint[bin_of(z[i])][bin_of(z[i + lag])] += 1;
        }

        let mut marg1 = vec![0usize; n_bins];
        let mut marg2 = vec![0usize; n_bins];
        for i in 0..n_bins {
            for j in 0..n_bins {
                marg1[i] += joint[i][j];
                marg2[j] += joint[i][j];
            }
        }

        let total = n_pairs as f64;
        let mut mi = 0.0;
        for i in 0..n_bins {
            for j in 0..n_bins {
                let pj = joint[i][j] as f64 / total;
                let p1 = marg1[i] as f64 / total;
                let p2 = marg2[j] as f64 / total;
                if pj > 0.0 && p1 > 0.0 && p2 > 0.0 {
                    mi += pj * (pj / (p1 * p2)).ln();
                }
            }
        }
        ami_values.push(mi);
    }

    // Find first minimum
    for i in 1..ami_values.len() - 1 {
        if ami_values[i] < ami_values[i - 1] && ami_values[i] <= ami_values[i + 1] {
            return (i + 1) as f64; // +1 because lag starts at 1
        }
    }
    max_lag as f64
}
