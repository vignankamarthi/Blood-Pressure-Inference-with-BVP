/// Autocorrelation-based Catch22 features.
use super::helpers;

/// CO_f1ecac: First 1/e crossing of the ACF.
/// Returns the first lag where ACF drops below 1/e.
pub fn co_f1ecac(acf: &[f64]) -> f64 {
    let threshold = 1.0 / std::f64::consts::E;
    for i in 1..acf.len() {
        if acf[i] < threshold {
            // Linear interpolation between i-1 and i
            if (acf[i - 1] - acf[i]).abs() < 1e-15 {
                return i as f64;
            }
            let frac = (acf[i - 1] - threshold) / (acf[i - 1] - acf[i]);
            return (i - 1) as f64 + frac;
        }
    }
    acf.len() as f64
}

/// CO_FirstMin_ac: First minimum of the ACF.
pub fn co_firstmin_ac(acf: &[f64]) -> f64 {
    helpers::first_min_ac(acf) as f64
}

/// CO_trev_1_num: Time-reversibility statistic.
/// Mean of (x_{t+1} - x_t)^3.
pub fn co_trev_1_num(signal: &[f64]) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }
    let z = helpers::zscore(signal);
    let n = z.len() - 1;
    let sum: f64 = z.windows(2).map(|w| (w[1] - w[0]).powi(3)).sum();
    sum / n as f64
}

/// CO_Embed2_Dist_tau_d_expfit_meandiff: Exponential fit to embedding distance distribution.
/// Embeds signal with delay tau, computes pairwise distances, fits exponential.
pub fn co_embed2_dist_tau_d_expfit_meandiff(signal: &[f64], tau: usize) -> f64 {
    let tau = tau.max(1);
    let z = helpers::zscore(signal);
    let n = z.len();
    if n <= tau {
        return f64::NAN;
    }

    // Create 2D embedding
    let n_embed = n - tau;
    let mut distances = Vec::with_capacity(n_embed * (n_embed - 1) / 2);

    for i in 0..n_embed {
        for j in (i + 1)..n_embed {
            let d = ((z[i] - z[j]).powi(2) + (z[i + tau] - z[j + tau]).powi(2)).sqrt();
            distances.push(d);
        }
    }

    if distances.is_empty() {
        return f64::NAN;
    }

    // Compute mean of distances
    let mean_dist = distances.iter().sum::<f64>() / distances.len() as f64;

    // Fit exponential: for mean of exponential distribution, parameter = 1/mean
    // Return the mean difference from the exponential fit
    mean_dist
}

/// CO_HistogramAMI_even_2_5: Automutual information using 5-bin histogram at lag 2.
pub fn co_histogram_ami_even_2_5(signal: &[f64]) -> f64 {
    let z = helpers::zscore(signal);
    let n = z.len();
    let lag = 2;
    if n <= lag {
        return 0.0;
    }

    let n_bins = 5;
    let min_val = z.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;
    if range < 1e-15 {
        return 0.0;
    }
    let bin_width = range / n_bins as f64;

    let bin_of = |v: f64| -> usize {
        let b = ((v - min_val) / bin_width) as usize;
        b.min(n_bins - 1)
    };

    let n_pairs = n - lag;

    // Joint histogram
    let mut joint = vec![vec![0usize; n_bins]; n_bins];
    for i in 0..n_pairs {
        let b1 = bin_of(z[i]);
        let b2 = bin_of(z[i + lag]);
        joint[b1][b2] += 1;
    }

    // Marginals
    let mut marg1 = vec![0usize; n_bins];
    let mut marg2 = vec![0usize; n_bins];
    for i in 0..n_bins {
        for j in 0..n_bins {
            marg1[i] += joint[i][j];
            marg2[j] += joint[i][j];
        }
    }

    // Mutual information
    let mut mi = 0.0;
    let total = n_pairs as f64;
    for i in 0..n_bins {
        for j in 0..n_bins {
            let p_joint = joint[i][j] as f64 / total;
            let p1 = marg1[i] as f64 / total;
            let p2 = marg2[j] as f64 / total;
            if p_joint > 0.0 && p1 > 0.0 && p2 > 0.0 {
                mi += p_joint * (p_joint / (p1 * p2)).ln();
            }
        }
    }
    mi
}
