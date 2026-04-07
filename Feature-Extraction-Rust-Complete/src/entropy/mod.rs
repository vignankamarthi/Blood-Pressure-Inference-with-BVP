pub mod permutation;
pub mod complexity;
pub mod generalized;
pub mod sample;
pub mod approximate;

use anyhow::Result;
use std::collections::HashMap;

type Pattern = Vec<usize>;

/// All entropy features computed for a single signal segment.
#[derive(Debug, Clone)]
pub struct EntropyFeatures {
    pub permutation_entropy: f64,
    pub statistical_complexity: f64,
    pub fisher_shannon: f64,
    pub fisher_information: f64,
    pub renyi_pe: f64,
    pub renyi_complexity: f64,
    pub tsallis_pe: f64,
    pub tsallis_complexity: f64,
    pub sample_entropy: f64,
    pub approximate_entropy: f64,
}

impl EntropyFeatures {
    pub fn calculate(signal: &[f64], dimension: usize, tau: usize) -> Result<Self> {
        let pe = permutation::permutation_entropy(signal, dimension, tau)?;
        let sc = complexity::statistical_complexity(signal, dimension, tau)?;
        let (fs, fi) = complexity::fisher_information(signal, dimension, tau)?;
        let (rpe, rc) = generalized::renyi_entropy(signal, dimension, tau, 1.0)?;
        let (tpe, tc) = generalized::tsallis_entropy(signal, dimension, tau, 1.0)?;

        // Sample and approximate entropy with standard parameters
        // r = 0.2 * std(signal), m = 2
        let se = sample::sample_entropy(signal, 2, None);
        let ae = approximate::approximate_entropy(signal, 2, None);

        Ok(Self {
            permutation_entropy: pe,
            statistical_complexity: sc,
            fisher_shannon: fs,
            fisher_information: fi,
            renyi_pe: rpe,
            renyi_complexity: rc,
            tsallis_pe: tpe,
            tsallis_complexity: tc,
            sample_entropy: se,
            approximate_entropy: ae,
        })
    }
}

// Shared helper functions used by submodules

pub(crate) fn extract_ordinal_patterns(signal: &[f64], dimension: usize, tau: usize) -> Result<Vec<Pattern>> {
    if signal.len() < dimension * tau {
        anyhow::bail!("Signal too short for given dimension={} and tau={}", dimension, tau);
    }
    let n_patterns = signal.len() - (dimension - 1) * tau;
    let mut patterns = Vec::with_capacity(n_patterns);

    for i in 0..n_patterns {
        let mut embedded: Vec<(f64, usize)> = (0..dimension)
            .map(|j| (signal[i + j * tau], j))
            .collect();
        embedded.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        patterns.push(embedded.iter().map(|&(_, idx)| idx).collect());
    }
    Ok(patterns)
}

pub(crate) fn calculate_probability_distribution(patterns: &[Pattern], _dimension: usize) -> Vec<f64> {
    let mut counts: HashMap<Pattern, usize> = HashMap::new();
    for p in patterns {
        *counts.entry(p.clone()).or_insert(0) += 1;
    }
    let total = patterns.len() as f64;
    let mut dist: Vec<f64> = counts.values().map(|&c| c as f64 / total).collect();
    dist.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dist
}

pub(crate) fn factorial(n: usize) -> usize {
    match n {
        0 | 1 => 1,
        _ => (2..=n).product(),
    }
}

pub(crate) fn pattern_to_index(pattern: &[usize]) -> usize {
    let mut index = 0;
    let n = pattern.len();
    for i in 0..n {
        let smaller_count = (i + 1..n).filter(|&j| pattern[j] < pattern[i]).count();
        index = index * (n - i) + smaller_count;
    }
    index
}
