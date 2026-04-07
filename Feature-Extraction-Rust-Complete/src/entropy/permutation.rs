use anyhow::Result;
use std::collections::HashMap;
use super::{extract_ordinal_patterns, factorial, Pattern};

/// Permutation Entropy -- Bandt & Pompe (2002)
/// Returns normalized PE in [0, 1].
pub fn permutation_entropy(signal: &[f64], dimension: usize, tau: usize) -> Result<f64> {
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;

    let mut counts: HashMap<Pattern, usize> = HashMap::new();
    for p in patterns {
        *counts.entry(p).or_insert(0) += 1;
    }

    let total = counts.values().sum::<usize>() as f64;
    let mut entropy = 0.0;
    for &count in counts.values() {
        let p = count as f64 / total;
        if p > 0.0 {
            entropy -= p * p.log2();
        }
    }

    let max_entropy = (factorial(dimension) as f64).log2();
    if max_entropy > 0.0 {
        Ok(entropy / max_entropy)
    } else {
        Ok(0.0)
    }
}
