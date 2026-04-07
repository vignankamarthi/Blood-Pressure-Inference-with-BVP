use anyhow::Result;
use std::collections::HashMap;
use super::{extract_ordinal_patterns, calculate_probability_distribution, factorial, pattern_to_index, Pattern};

/// Statistical Complexity -- Rosso et al. (2007)
pub fn statistical_complexity(signal: &[f64], dimension: usize, tau: usize) -> Result<f64> {
    let pe = super::permutation::permutation_entropy(signal, dimension, tau)?;
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let diseq = calculate_disequilibrium(&patterns, dimension)?;
    Ok(pe * diseq)
}

/// Fisher Information -- Martin et al. (2003)
/// Returns (PE, Fisher Information)
pub fn fisher_information(signal: &[f64], dimension: usize, tau: usize) -> Result<(f64, f64)> {
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let n_possible = factorial(dimension);
    let mut full_prob = vec![0.0; n_possible];

    let mut counts: HashMap<Pattern, usize> = HashMap::new();
    for p in &patterns {
        *counts.entry(p.clone()).or_insert(0) += 1;
    }

    let total = patterns.len() as f64;
    for (pattern, &count) in &counts {
        let idx = pattern_to_index(pattern);
        if idx < n_possible {
            full_prob[idx] = count as f64 / total;
        }
    }

    let pe = super::permutation::permutation_entropy(signal, dimension, tau)?;

    // ordpy special case: single pattern -> Fisher_I = 1.0
    if full_prob.iter().filter(|&&p| (p - 1.0).abs() < 1e-10).count() == 1 {
        return Ok((0.0, 1.0));
    }

    let mut prob_rev = full_prob;
    prob_rev.reverse();
    let sqrt_probs: Vec<f64> = prob_rev.iter().map(|&p| p.sqrt()).collect();

    let mut fi = 0.0;
    for i in 1..sqrt_probs.len() {
        let diff = sqrt_probs[i] - sqrt_probs[i - 1];
        fi += diff * diff;
    }
    fi /= 2.0;

    Ok((pe, fi))
}

fn calculate_disequilibrium(patterns: &[Pattern], dimension: usize) -> Result<f64> {
    let prob_dist = calculate_probability_distribution(patterns, dimension);
    let n = factorial(dimension) as f64;
    let u = 1.0 / n;
    let n_missing = n - prob_dist.len() as f64;

    let mut s_mix = 0.0;
    for &p in &prob_dist {
        let mix = (u + p) / 2.0;
        s_mix -= mix * mix.ln();
    }
    let half_u = 0.5 * u;
    if half_u > 0.0 {
        s_mix -= half_u * half_u.ln() * n_missing;
    }

    let s_p: f64 = prob_dist.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>() / 2.0;

    let s_u = n.ln() / 2.0;
    let js = s_mix - s_p - s_u;
    let js_max = -0.5 * (((n + 1.0) / n) * (n + 1.0).ln() + n.ln() - 2.0 * (2.0 * n).ln());

    if js_max.abs() < 1e-15 {
        Ok(0.0)
    } else {
        Ok(js / js_max)
    }
}
