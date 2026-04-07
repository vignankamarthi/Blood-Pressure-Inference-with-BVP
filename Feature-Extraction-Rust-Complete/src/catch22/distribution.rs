/// Distribution-based Catch22 features.
use super::helpers;

/// DN_HistogramMode_N: Mode of the data distribution using N-bin histogram.
/// Returns the bin center with the highest count.
pub fn dn_histogram_mode(signal: &[f64], n_bins: usize) -> f64 {
    if signal.is_empty() {
        return f64::NAN;
    }
    let z = helpers::zscore(signal);
    let (edges, counts) = helpers::histogram(&z, n_bins);

    if counts.is_empty() {
        return f64::NAN;
    }

    let max_idx = counts.iter().enumerate()
        .max_by_key(|&(_, &c)| c)
        .map(|(i, _)| i)
        .unwrap_or(0);

    if edges.len() > max_idx + 1 {
        (edges[max_idx] + edges[max_idx + 1]) / 2.0
    } else {
        f64::NAN
    }
}

/// DN_OutlierInclude_p/n_001_mdrmd: Median of rolling median ratios for outliers.
/// positive=true for positive outliers, false for negative.
pub fn dn_outlierinclude_mdrmd(signal: &[f64], positive: bool) -> f64 {
    if signal.is_empty() {
        return 0.0;
    }
    let z = helpers::zscore(signal);
    let n = z.len();

    // Find threshold increments
    let mut inc_values = Vec::new();
    let start = if positive { 0.01 } else { -0.01 };
    let step = if positive { 0.01 } else { -0.01 };

    let max_iter = 1000;
    for i in 0..max_iter {
        let threshold = start + i as f64 * step;
        if positive && threshold > 10.0 { break; }
        if !positive && threshold < -10.0 { break; }

        let count = if positive {
            z.iter().filter(|&&v| v >= threshold).count()
        } else {
            z.iter().filter(|&&v| v <= threshold).count()
        };

        if count == 0 { break; }

        // median of indices where outliers occur
        let indices: Vec<f64> = z.iter().enumerate()
            .filter(|&(_, &v)| if positive { v >= threshold } else { v <= threshold })
            .map(|(i, _)| i as f64)
            .collect();

        if !indices.is_empty() {
            let med = sorted_median(&indices);
            let ratio = med / (n as f64 / 2.0);
            inc_values.push(ratio);
        }
    }

    if inc_values.is_empty() {
        0.0
    } else {
        sorted_median(&inc_values)
    }
}

fn sorted_median(data: &[f64]) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    }
}
