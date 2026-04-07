use anyhow::Result;
use csv::ReaderBuilder;
use std::fs;
use std::path::{Path, PathBuf};

/// Load a CSV file where each column is a signal segment.
/// Returns a Vec of (column_name, signal_data) pairs.
pub fn load_csv_signals(path: &Path) -> Result<Vec<(String, Vec<f64>)>> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)?;

    let headers: Vec<String> = reader.headers()?
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Read all records
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); headers.len()];
    for result in reader.records() {
        let record = result?;
        for (i, field) in record.iter().enumerate() {
            if i < columns.len() {
                let val: f64 = field.parse().unwrap_or(f64::NAN);
                columns[i].push(val);
            }
        }
    }

    Ok(headers.into_iter().zip(columns.into_iter()).collect())
}

/// List all CSV files in a directory.
pub fn list_csv_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("csv") {
                files.push(path);
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Load a single-column CSV as a signal vector.
pub fn load_single_signal(path: &Path) -> Result<Vec<f64>> {
    let content = fs::read_to_string(path)?;
    let values: Vec<f64> = content.lines()
        .filter(|line| !line.is_empty())
        .filter_map(|line| line.trim().parse::<f64>().ok())
        .collect();
    Ok(values)
}
