use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use std::fs;
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::{Arc, Mutex};

use feature_extraction_rust_complete::catch22::Catch22Features;
use feature_extraction_rust_complete::entropy::EntropyFeatures;
use feature_extraction_rust_complete::stats::StatFeatures;
use feature_extraction_rust_complete::signal_processing;
use feature_extraction_rust_complete::types::{ExtractionCheckpoint, FeatureFramework};
use feature_extraction_rust_complete::utils;
use feature_extraction_rust_complete::data_loader;

#[derive(Parser)]
#[command(name = "feature-extraction")]
#[command(about = "Pure Rust feature extraction: Catch22 + entropy + statistical features")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Verbosity level (0=error, 1=info, 2=debug)
    #[arg(short, long, default_value = "1")]
    verbose: u8,
}

#[derive(Subcommand)]
enum Commands {
    /// Extract features from signal data
    Extract {
        /// Input directory containing CSV signal files
        #[arg(short, long)]
        input_dir: PathBuf,

        /// Output directory for feature CSVs
        #[arg(short, long)]
        output_dir: PathBuf,

        /// Feature frameworks to extract (comma-separated: catch22,entropy,stats)
        #[arg(short, long, default_value = "catch22,entropy,stats")]
        frameworks: String,

        /// Number of worker threads (default: all CPUs)
        #[arg(short, long)]
        workers: Option<usize>,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Entropy embedding dimensions (comma-separated)
        #[arg(long, default_value = "7")]
        dimensions: String,

        /// Entropy time delays (comma-separated)
        #[arg(long, default_value = "2")]
        delays: String,
    },
    /// Validate features against Python reference values
    Validate {
        /// Path to reference_values.json
        #[arg(short, long, default_value = "tests/reference_values.json")]
        reference: PathBuf,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let log_level = match cli.verbose {
        0 => "error",
        1 => "info",
        _ => "debug",
    };
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or(log_level)).init();

    match cli.command {
        Commands::Extract {
            input_dir, output_dir, frameworks, workers, resume, dimensions, delays,
        } => {
            let fw: Vec<FeatureFramework> = frameworks
                .split(',')
                .filter_map(|s| FeatureFramework::from_str(s.trim()))
                .collect();

            let dims: Vec<usize> = dimensions.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let taus: Vec<usize> = delays.split(',').filter_map(|s| s.trim().parse().ok()).collect();
            let n_workers = workers.unwrap_or_else(num_cpus::get);

            log::info!("Feature extraction starting");
            log::info!("  Input: {:?}", input_dir);
            log::info!("  Output: {:?}", output_dir);
            log::info!("  Frameworks: {:?}", fw.iter().map(|f| f.as_str()).collect::<Vec<_>>());
            log::info!("  Workers: {}", n_workers);

            rayon::ThreadPoolBuilder::new().num_threads(n_workers).build_global().ok();
            fs::create_dir_all(&output_dir)?;

            run_extraction(&input_dir, &output_dir, &fw, &dims, &taus, resume)?;
        }
        Commands::Validate { reference } => {
            log::info!("Validating against reference: {:?}", reference);
            println!("Use `cargo test` for validation");
        }
    }

    Ok(())
}

fn run_extraction(
    input_dir: &PathBuf, output_dir: &PathBuf,
    frameworks: &[FeatureFramework], dimensions: &[usize], delays: &[usize],
    resume: bool,
) -> Result<()> {
    let files = data_loader::list_csv_files(input_dir)?;
    log::info!("Found {} CSV files", files.len());

    if files.is_empty() {
        log::warn!("No CSV files found in {:?}", input_dir);
        return Ok(());
    }

    let checkpoint_path = output_dir.join("checkpoint.json");
    let checkpoint = if resume {
        utils::load_checkpoint(&checkpoint_path)?.unwrap_or_else(|| ExtractionCheckpoint::new(files.len()))
    } else {
        ExtractionCheckpoint::new(files.len())
    };
    let checkpoint = Arc::new(Mutex::new(checkpoint));

    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("[{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ({eta}) {msg}").unwrap());

    let results: Arc<Mutex<Vec<Vec<String>>>> = Arc::new(Mutex::new(Vec::new()));
    let output_dir_clone = output_dir.clone();

    files.par_iter().for_each(|file| {
        let file_name = file.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();

        {
            let ckpt = checkpoint.lock().unwrap();
            if ckpt.is_completed(&file_name) {
                pb.inc(1);
                return;
            }
        }

        pb.set_message(file_name.clone());

        match process_file(file, frameworks, dimensions, delays) {
            Ok(rows) => {
                results.lock().unwrap().extend(rows);
                checkpoint.lock().unwrap().mark_completed(file_name);
            }
            Err(e) => {
                log::error!("Failed {}: {}", file_name, e);
                checkpoint.lock().unwrap().mark_failed(file_name, e.to_string());
            }
        }

        if let Ok(ckpt) = checkpoint.lock() {
            if ckpt.completed_subjects.len() % 10 == 0 {
                let _ = utils::save_checkpoint(&output_dir_clone.join("checkpoint.json"), &ckpt);
            }
        }
        pb.inc(1);
    });

    pb.finish_with_message("Done");

    let results = results.lock().unwrap();
    if !results.is_empty() {
        write_results_csv(&output_dir.join("features.csv"), &results)?;
    }

    let ckpt = checkpoint.lock().unwrap();
    utils::save_checkpoint(&checkpoint_path, &ckpt)?;
    log::info!("Complete: {} processed, {} failed", ckpt.completed_subjects.len(), ckpt.failed_subjects.len());
    Ok(())
}

fn process_file(
    path: &PathBuf, frameworks: &[FeatureFramework],
    dimensions: &[usize], delays: &[usize],
) -> Result<Vec<Vec<String>>> {
    let signals = data_loader::load_csv_signals(path)?;
    let mut all_rows = Vec::new();
    let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("unknown");

    for (col_name, raw_signal) in &signals {
        let (clean, nan_pct) = signal_processing::remove_nans(raw_signal);
        if clean.len() < 50 || nan_pct > 0.2 { continue; }

        let mut row = vec![
            file_name.to_string(), col_name.clone(),
            clean.len().to_string(), format!("{:.6}", nan_pct),
        ];

        if frameworks.contains(&FeatureFramework::Catch22) {
            match Catch22Features::compute(&clean) {
                Ok(c22) => row.extend(catch22_to_strings(&c22)),
                Err(e) => {
                    log::debug!("Catch22 error: {}", e);
                    row.extend(vec!["NaN".to_string(); 22]);
                }
            }
        }

        if frameworks.contains(&FeatureFramework::Entropy) {
            for &d in dimensions {
                for &tau in delays {
                    match EntropyFeatures::calculate(&clean, d, tau) {
                        Ok(ent) => row.extend(entropy_to_strings(&ent, d, tau)),
                        Err(e) => {
                            log::debug!("Entropy error d={} tau={}: {}", d, tau, e);
                            row.extend(vec!["NaN".to_string(); 10]);
                        }
                    }
                }
            }
        }

        if frameworks.contains(&FeatureFramework::Stats) {
            let stats = StatFeatures::compute(&clean);
            row.extend(stats_to_strings(&stats));
        }

        all_rows.push(row);
    }
    Ok(all_rows)
}

fn catch22_to_strings(c22: &Catch22Features) -> Vec<String> {
    vec![
        c22.dn_histogram_mode_5, c22.dn_histogram_mode_10, c22.co_f1ecac,
        c22.co_firstmin_ac, c22.co_histogram_ami_even_2_5, c22.co_trev_1_num,
        c22.md_hrv_classic_pnn40, c22.sb_binarystats_mean_longstretch1,
        c22.sb_transitionmatrix_3ac_sumdiagcov, c22.pd_periodicitywang_th0_01,
        c22.co_embed2_dist_tau_d_expfit_meandiff, c22.in_automutualinfostats_40_gaussian_fmmi,
        c22.fc_localsimple_mean1_tauresrat, c22.dn_outlierinclude_p_001_mdrmd,
        c22.dn_outlierinclude_n_001_mdrmd, c22.sp_summaries_welch_rect_area_5_1,
        c22.sb_motifthree_quantile_hh, c22.sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1,
        c22.sc_fluctanal_2_dfa_50_1_2_logi_prop_r1, c22.sp_summaries_welch_rect_centroid,
        c22.fc_localsimple_mean3_stderr, c22.co_embed2_dist_tau_d_expfit_meandiff_2,
    ].iter().map(|v| format!("{:.10}", v)).collect()
}

fn entropy_to_strings(ent: &EntropyFeatures, _d: usize, _tau: usize) -> Vec<String> {
    vec![
        format!("{:.10}", ent.permutation_entropy),
        format!("{:.10}", ent.statistical_complexity),
        format!("{:.10}", ent.fisher_shannon),
        format!("{:.10}", ent.fisher_information),
        format!("{:.10}", ent.renyi_pe),
        format!("{:.10}", ent.renyi_complexity),
        format!("{:.10}", ent.tsallis_pe),
        format!("{:.10}", ent.tsallis_complexity),
        format!("{:.10}", ent.sample_entropy),
        format!("{:.10}", ent.approximate_entropy),
    ]
}

fn stats_to_strings(stats: &StatFeatures) -> Vec<String> {
    vec![
        stats.mean, stats.median, stats.std, stats.skewness,
        stats.kurtosis, stats.rms, stats.min, stats.max,
    ].iter().map(|v| format!("{:.10}", v)).collect()
}

fn write_results_csv(path: &PathBuf, rows: &[Vec<String>]) -> Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    let header = vec![
        "file_name", "segment_id", "signal_length", "nan_percentage",
        "dn_histogram_mode_5", "dn_histogram_mode_10", "co_f1ecac", "co_firstmin_ac",
        "co_histogram_ami_even_2_5", "co_trev_1_num", "md_hrv_classic_pnn40",
        "sb_binarystats_mean_longstretch1", "sb_transitionmatrix_3ac_sumdiagcov",
        "pd_periodicitywang_th0_01", "co_embed2_dist_tau_d_expfit_meandiff",
        "in_automutualinfostats_40_gaussian_fmmi", "fc_localsimple_mean1_tauresrat",
        "dn_outlierinclude_p_001_mdrmd", "dn_outlierinclude_n_001_mdrmd",
        "sp_summaries_welch_rect_area_5_1", "sb_motifthree_quantile_hh",
        "sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1", "sc_fluctanal_2_dfa_50_1_2_logi_prop_r1",
        "sp_summaries_welch_rect_centroid", "fc_localsimple_mean3_stderr",
        "co_embed2_dist_tau_d_expfit_meandiff_2",
        "permutation_entropy", "statistical_complexity", "fisher_shannon",
        "fisher_information", "renyi_pe", "renyi_complexity", "tsallis_pe",
        "tsallis_complexity", "sample_entropy", "approximate_entropy",
        "stat_mean", "stat_median", "stat_std", "stat_skewness",
        "stat_kurtosis", "stat_rms", "stat_min", "stat_max",
    ];
    wtr.write_record(&header)?;
    for row in rows { wtr.write_record(row)?; }
    wtr.flush()?;
    log::info!("Wrote {} rows to {:?}", rows.len(), path);
    Ok(())
}
