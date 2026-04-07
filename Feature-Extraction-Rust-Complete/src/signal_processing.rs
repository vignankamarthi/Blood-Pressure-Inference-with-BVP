/// Signal preprocessing for PPG waveforms.
/// Adapted from AI4Pain Paper 2 (ai4pain-rust/src/signal_processing.rs).
/// NO pre-extraction z-score normalization -- locked decision.

/// Remove NaN values from a signal, returning the clean signal
/// and the percentage of NaN values found.
pub fn remove_nans(signal: &[f64]) -> (Vec<f64>, f64) {
    let total = signal.len() as f64;
    let clean: Vec<f64> = signal.iter().copied().filter(|v| v.is_finite()).collect();
    let nan_count = total - clean.len() as f64;
    let nan_pct = if total > 0.0 { nan_count / total } else { 0.0 };
    (clean, nan_pct)
}

/// Validate that a signal meets minimum quality requirements.
pub fn validate_signal(signal: &[f64], min_length: usize, max_nan_pct: f64) -> Result<(), String> {
    if signal.is_empty() {
        return Err("Signal is empty".to_string());
    }

    let (clean, nan_pct) = remove_nans(signal);

    if nan_pct > max_nan_pct {
        return Err(format!(
            "NaN percentage {:.2}% exceeds threshold {:.2}%",
            nan_pct * 100.0,
            max_nan_pct * 100.0
        ));
    }

    if clean.len() < min_length {
        return Err(format!(
            "Clean signal length {} is below minimum {}",
            clean.len(),
            min_length
        ));
    }

    // Check for constant signal
    let first = clean[0];
    if clean.iter().all(|&v| (v - first).abs() < 1e-15) {
        return Err("Signal is constant (zero variance)".to_string());
    }

    Ok(())
}

/// Compute signal quality metrics.
pub struct SignalQuality {
    pub mean: f64,
    pub std_dev: f64,
    pub nan_percentage: f64,
    pub valid_length: usize,
}

impl SignalQuality {
    pub fn compute(signal: &[f64]) -> Self {
        let (clean, nan_pct) = remove_nans(signal);
        let n = clean.len() as f64;

        let mean = if n > 0.0 {
            clean.iter().sum::<f64>() / n
        } else {
            0.0
        };

        let std_dev = if n > 1.0 {
            let variance = clean.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
            variance.sqrt()
        } else {
            0.0
        };

        Self {
            mean,
            std_dev,
            nan_percentage: nan_pct,
            valid_length: clean.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_remove_nans_clean() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (clean, pct) = remove_nans(&signal);
        assert_eq!(clean.len(), 5);
        assert!((pct - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_remove_nans_with_nans() {
        let signal = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let (clean, pct) = remove_nans(&signal);
        assert_eq!(clean.len(), 3);
        assert!((pct - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_remove_nans_with_inf() {
        let signal = vec![1.0, f64::INFINITY, 3.0];
        let (clean, pct) = remove_nans(&signal);
        assert_eq!(clean.len(), 2);
    }

    #[test]
    fn test_validate_signal_ok() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(validate_signal(&signal, 3, 0.5).is_ok());
    }

    #[test]
    fn test_validate_signal_empty() {
        let signal: Vec<f64> = vec![];
        assert!(validate_signal(&signal, 3, 0.5).is_err());
    }

    #[test]
    fn test_validate_signal_too_short() {
        let signal = vec![1.0, 2.0];
        assert!(validate_signal(&signal, 3, 0.5).is_err());
    }

    #[test]
    fn test_validate_signal_too_many_nans() {
        let signal = vec![1.0, f64::NAN, f64::NAN, f64::NAN, 5.0];
        assert!(validate_signal(&signal, 1, 0.5).is_err());
    }

    #[test]
    fn test_validate_signal_constant() {
        let signal = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        assert!(validate_signal(&signal, 3, 0.5).is_err());
    }

    #[test]
    fn test_signal_quality() {
        let signal = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let quality = SignalQuality::compute(&signal);
        assert_eq!(quality.valid_length, 8);
        assert!((quality.mean - 5.0).abs() < 1e-10);
        assert!((quality.nan_percentage - 0.0).abs() < 1e-10);
    }
}
