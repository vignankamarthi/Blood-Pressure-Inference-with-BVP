/// Statistical descriptive features for time-series signals.
/// All functions expect a clean (no NaN) f64 slice.

/// Arithmetic mean.
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Median (middle value of sorted data).
pub fn median(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}

/// Sample standard deviation (ddof=1, matching numpy default for ddof=1).
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return f64::NAN;
    }
    let m = mean(data);
    let n = data.len() as f64;
    let variance = data.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

/// Skewness (Fisher's definition, bias=False equivalent).
/// Uses the adjusted formula: [n/((n-1)(n-2))] * sum[((x-mean)/std)^3]
pub fn skewness(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 3 {
        return f64::NAN;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 {
        return 0.0;
    }
    let n_f = n as f64;
    let m3 = data.iter().map(|v| ((v - m) / s).powi(3)).sum::<f64>();
    let adjustment = n_f / ((n_f - 1.0) * (n_f - 2.0));
    adjustment * m3
}

/// Excess kurtosis (Fisher's definition, bias=False equivalent).
/// Uses the adjusted formula matching scipy.stats.kurtosis(fisher=True, bias=False).
pub fn kurtosis(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 4 {
        return f64::NAN;
    }
    let m = mean(data);
    let s = std_dev(data);
    if s < 1e-15 {
        return 0.0;
    }
    let n_f = n as f64;
    let m4 = data.iter().map(|v| ((v - m) / s).powi(4)).sum::<f64>();
    let term1 = (n_f * (n_f + 1.0)) / ((n_f - 1.0) * (n_f - 2.0) * (n_f - 3.0)) * m4;
    let term2 = (3.0 * (n_f - 1.0).powi(2)) / ((n_f - 2.0) * (n_f - 3.0));
    term1 - term2
}

/// Root mean square.
pub fn rms(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }
    let sum_sq: f64 = data.iter().map(|v| v * v).sum();
    (sum_sq / data.len() as f64).sqrt()
}

/// Minimum value.
pub fn min(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
}

/// Maximum value.
pub fn max(data: &[f64]) -> f64 {
    data.iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Compute all 8 statistical features at once.
pub struct StatFeatures {
    pub mean: f64,
    pub median: f64,
    pub std: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub rms: f64,
    pub min: f64,
    pub max: f64,
}

impl StatFeatures {
    pub fn compute(data: &[f64]) -> Self {
        Self {
            mean: mean(data),
            median: median(data),
            std: std_dev(data),
            skewness: skewness(data),
            kurtosis: kurtosis(data),
            rms: rms(data),
            min: min(data),
            max: max(data),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-10);
        assert!((mean(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_odd() {
        assert!((median(&[1.0, 3.0, 5.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        assert!((median(&[1.0, 2.0, 3.0, 4.0]) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_std_dev() {
        // numpy: np.std([2,4,4,4,5,5,7,9], ddof=1) = 2.1380899352993952
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert!((std_dev(&data) - 2.1380899352993952).abs() < 1e-10);
    }

    #[test]
    fn test_rms() {
        // rms([1,2,3]) = sqrt((1+4+9)/3) = sqrt(14/3)
        let expected = (14.0_f64 / 3.0).sqrt();
        assert!((rms(&[1.0, 2.0, 3.0]) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        assert!((min(&data) - 1.0).abs() < 1e-10);
        assert!((max(&data) - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_empty_data() {
        assert!(mean(&[]).is_nan());
        assert!(median(&[]).is_nan());
        assert!(std_dev(&[]).is_nan());
        assert!(rms(&[]).is_nan());
    }

    #[test]
    fn test_stat_features_all() {
        let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let features = StatFeatures::compute(&data);
        assert!((features.mean - 5.0).abs() < 1e-10);
        assert!((features.std - 2.1380899352993952).abs() < 1e-10);
        assert!((features.min - 2.0).abs() < 1e-10);
        assert!((features.max - 9.0).abs() < 1e-10);
    }
}
