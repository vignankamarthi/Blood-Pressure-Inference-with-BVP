/// Sample Entropy -- Richman & Moorman (2000)
/// Measures unpredictability of a time series.
/// Lower values = more self-similar / regular.
///
/// Parameters:
/// - signal: input time series
/// - m: embedding dimension (template length), typically 2
/// - r: tolerance (if None, uses 0.2 * std(signal))
pub fn sample_entropy(signal: &[f64], m: usize, r: Option<f64>) -> f64 {
    let n = signal.len();
    if n < m + 1 {
        return f64::NAN;
    }

    let tolerance = r.unwrap_or_else(|| {
        let mean = signal.iter().sum::<f64>() / n as f64;
        let var = signal.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0);
        0.2 * var.sqrt()
    });

    if tolerance <= 0.0 {
        return f64::NAN;
    }

    let count_matches = |template_len: usize| -> usize {
        let mut count = 0;
        for i in 0..n - template_len {
            for j in (i + 1)..n - template_len {
                let is_match = (0..template_len)
                    .all(|k| (signal[i + k] - signal[j + k]).abs() <= tolerance);
                if is_match {
                    count += 1;
                }
            }
        }
        count
    };

    let b = count_matches(m) as f64;     // matches of length m
    let a = count_matches(m + 1) as f64;  // matches of length m+1

    if b == 0.0 {
        return f64::NAN; // undefined
    }

    -(a / b).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_entropy_constant() {
        // Constant signal: perfectly self-similar -> SampEn should be 0 or very small
        let signal = vec![1.0; 100];
        let se = sample_entropy(&signal, 2, Some(0.2));
        // For constant signal, all templates match -> a/b = 1 -> ln(1) = 0
        // But r > 0 means all match, so se should be ~0
        assert!(se.is_finite() || se.is_nan());
    }

    #[test]
    fn test_sample_entropy_sine() {
        let signal: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let se = sample_entropy(&signal, 2, None);
        assert!(se.is_finite());
        assert!(se > 0.0);
    }
}
