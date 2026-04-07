/// Successive-difference and spectral Catch22 features.
use super::helpers;
use rustfft::{FftPlanner, num_complex::Complex};

/// MD_hrv_classic_pnn40: Proportion of successive differences exceeding 40% of std.
pub fn md_hrv_classic_pnn40(signal: &[f64]) -> f64 {
    if signal.len() < 2 {
        return 0.0;
    }
    let diffs = helpers::successive_diffs(signal);
    let n = diffs.len() as f64;
    let mean = diffs.iter().sum::<f64>() / n;
    let std = (diffs.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();

    if std < 1e-15 {
        return 0.0;
    }

    let threshold = 0.4 * std;
    let count = diffs.iter().filter(|&&d| d.abs() > threshold).count();
    count as f64 / n
}

/// PD_PeriodicityWang_th0_01: Periodicity measure using Wang's method.
/// Finds the period of the dominant ACF peak.
pub fn pd_periodicitywang_th0_01(signal: &[f64]) -> f64 {
    let acf = helpers::autocorrelation_fn(signal);
    let n = acf.len();
    if n < 4 {
        return 0.0;
    }

    // Find the first peak in ACF after the first zero crossing
    let threshold = 0.01;
    let mut found_zero = false;

    for i in 1..n {
        if acf[i] < 0.0 {
            found_zero = true;
        }
        if found_zero && i > 1 && acf[i] > threshold {
            // Check if this is a local max
            if acf[i] > acf[i - 1] && (i + 1 >= n || acf[i] >= acf[i + 1]) {
                return i as f64;
            }
        }
    }
    0.0
}

/// SP_Summaries_welch_rect_area_5_1: Area under the first 1/5 of the power spectrum.
pub fn sp_summaries_welch_rect_area_5_1(signal: &[f64]) -> f64 {
    let psd = compute_power_spectrum(signal);
    if psd.is_empty() {
        return 0.0;
    }
    let fifth = psd.len() / 5;
    if fifth == 0 {
        return psd.iter().sum();
    }
    let total: f64 = psd.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }
    let area: f64 = psd[..fifth].iter().sum();
    area / total
}

/// SP_Summaries_welch_rect_centroid: Spectral centroid of the power spectrum.
pub fn sp_summaries_welch_rect_centroid(signal: &[f64]) -> f64 {
    let psd = compute_power_spectrum(signal);
    if psd.is_empty() {
        return 0.0;
    }
    let total: f64 = psd.iter().sum();
    if total < 1e-15 {
        return 0.0;
    }
    let weighted_sum: f64 = psd.iter().enumerate()
        .map(|(i, &p)| (i + 1) as f64 * p)
        .sum();
    weighted_sum / total
}

/// Compute the one-sided power spectral density using FFT (rectangular window).
fn compute_power_spectrum(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }

    // Zero-mean
    let mean = signal.iter().sum::<f64>() / n as f64;

    // FFT
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = signal.iter()
        .map(|&v| Complex::new(v - mean, 0.0))
        .collect();

    fft.process(&mut buffer);

    // One-sided PSD (magnitude squared, normalized)
    let n_onesided = n / 2 + 1;
    let norm = 1.0 / (n as f64);

    buffer[..n_onesided]
        .iter()
        .map(|c| (c.re * c.re + c.im * c.im) * norm)
        .collect()
}
