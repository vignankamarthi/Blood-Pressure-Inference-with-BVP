/// Approximate Entropy -- Pincus (1991)
/// Similar to sample entropy but includes self-matches.
/// Lower values = more regular.
///
/// Parameters:
/// - signal: input time series
/// - m: embedding dimension, typically 2
/// - r: tolerance (if None, uses 0.2 * std(signal))
pub fn approximate_entropy(signal: &[f64], m: usize, r: Option<f64>) -> f64 {
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

    let phi = |template_len: usize| -> f64 {
        let nn = n - template_len + 1;
        let mut counts = vec![0usize; nn];

        for i in 0..nn {
            for j in 0..nn {
                let is_match = (0..template_len)
                    .all(|k| (signal[i + k] - signal[j + k]).abs() <= tolerance);
                if is_match {
                    counts[i] += 1;
                }
            }
        }

        let log_sum: f64 = counts.iter()
            .map(|&c| (c as f64 / nn as f64).ln())
            .sum();
        log_sum / nn as f64
    };

    let phi_m = phi(m);
    let phi_m1 = phi(m + 1);

    phi_m - phi_m1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_approximate_entropy_sine() {
        let signal: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
        let ae = approximate_entropy(&signal, 2, None);
        assert!(ae.is_finite());
        assert!(ae >= 0.0);
    }

    #[test]
    fn test_approximate_entropy_random() {
        // Pseudo-random via simple formula (no rand dep needed)
        let signal: Vec<f64> = (0..200).map(|i| {
            let x = (i as f64 * 1.618033988749895) % 1.0;
            x
        }).collect();
        let ae = approximate_entropy(&signal, 2, None);
        assert!(ae.is_finite());
    }
}
