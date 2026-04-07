use anyhow::Result;
use super::{extract_ordinal_patterns, calculate_probability_distribution, factorial};

/// Renyi Entropy -- generalization of Shannon entropy.
/// Returns (renyi_pe, renyi_complexity).
pub fn renyi_entropy(signal: &[f64], dimension: usize, tau: usize, q: f64) -> Result<(f64, f64)> {
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let prob_dist = calculate_probability_distribution(&patterns, dimension);
    let n = factorial(dimension) as f64;
    let u = 1.0 / n;
    let n_missing = n - prob_dist.len() as f64;

    let renyi_pe = if (q - 1.0).abs() < 1e-10 {
        let h: f64 = prob_dist.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
        h / n.ln()
    } else {
        let sum: f64 = prob_dist.iter().map(|&p| p.powf(q)).sum();
        ((1.0 / (1.0 - q)) * sum.ln()) / n.ln()
    };

    let jr_div = if (q - 1.0).abs() < 1e-10 {
        let mut s_mix = 0.0;
        for &p in &prob_dist {
            let mix = (u + p) / 2.0;
            if mix > 0.0 { s_mix -= mix * mix.ln(); }
        }
        let half_u = 0.5 * u;
        if half_u > 0.0 { s_mix -= half_u * half_u.ln() * n_missing; }
        let s_p: f64 = prob_dist.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum::<f64>() / 2.0;
        let s_u = n.ln() / 2.0;
        s_mix - s_p - s_u
    } else {
        let t1: f64 = prob_dist.iter().map(|&p| ((p + u) / 2.0).powf(1.0 - q) * p.powf(q)).sum();
        let t2: f64 = prob_dist.iter().map(|&p| (1.0 / n.powf(q)) * ((p + u) / 2.0).powf(1.0 - q)).sum();
        let t3 = n_missing * (1.0 / n.powf(q)) * (1.0 / (2.0 * n)).powf(1.0 - q);
        (1.0 / (2.0 * (q - 1.0))) * (t1.ln() + (t2 + t3).ln())
    };

    let jr_max = if (q - 1.0).abs() < 1e-10 {
        -0.5 * (((n + 1.0) / n) * (n + 1.0).ln() + n.ln() - 2.0 * (2.0 * n).ln())
    } else {
        let t1 = ((n + 1.0).powf(1.0 - q) + n - 1.0) / (2.0_f64.powf(1.0 - q) * n);
        let t2 = (1.0 - q) * ((n + 1.0) / (2.0 * n)).ln();
        (t1.ln() + t2) / (2.0 * (q - 1.0))
    };

    let complexity = if jr_max.abs() < 1e-15 { 0.0 } else { renyi_pe * jr_div / jr_max };
    Ok((renyi_pe, complexity))
}

/// Tsallis Entropy -- non-extensive entropy.
/// Returns (tsallis_pe, tsallis_complexity).
pub fn tsallis_entropy(signal: &[f64], dimension: usize, tau: usize, q: f64) -> Result<(f64, f64)> {
    let patterns = extract_ordinal_patterns(signal, dimension, tau)?;
    let prob_dist = calculate_probability_distribution(&patterns, dimension);
    let n = factorial(dimension) as f64;
    let u = 1.0 / n;
    let n_missing = n - prob_dist.len() as f64;

    let tsallis_pe = if (q - 1.0).abs() < 1e-10 {
        let h: f64 = prob_dist.iter().filter(|&&p| p > 0.0).map(|&p| -p * p.ln()).sum();
        h / n.ln()
    } else {
        let sum: f64 = prob_dist.iter().map(|&p| p.powf(q)).sum();
        let norm = (n.powf(1.0 - q) - 1.0) / (1.0 - q);
        if norm.abs() < 1e-15 { 0.0 } else { ((1.0 - sum) / (q - 1.0)) / norm }
    };

    fn logq(x: f64, q: f64) -> f64 {
        if (q - 1.0).abs() < 1e-10 { x.ln() } else { (x.powf(1.0 - q) - 1.0) / (1.0 - q) }
    }

    let t1: f64 = prob_dist.iter().map(|&p| p * logq((u + p) / (2.0 * p), q)).sum();
    let t1 = -0.5 * t1;

    let t2: f64 = prob_dist.iter().map(|&p| logq(n * (u + p) / 2.0, q)).sum();
    let t2_missing = logq(0.5, q) * n_missing;
    let t2 = -(0.5 / n) * (t2 + t2_missing);

    let jt_div = t1 + t2;

    let jt_max = if (q - 1.0).abs() < 1e-10 {
        -0.5 * (((n + 1.0) / n) * (n + 1.0).ln() + n.ln() - 2.0 * (2.0 * n).ln())
    } else {
        let num = 2.0_f64.powf(2.0 - q) * n - (1.0 + n).powf(1.0 - q) - n * (1.0 + 1.0 / n).powf(1.0 - q) - n + 1.0;
        let den = (1.0 - q) * 2.0_f64.powf(2.0 - q) * n;
        num / den
    };

    let complexity = if jt_max.abs() < 1e-15 { 0.0 } else { tsallis_pe * jt_div / jt_max };
    Ok((tsallis_pe, complexity))
}
