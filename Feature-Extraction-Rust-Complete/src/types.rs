use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

/// Which feature frameworks to extract
#[derive(Debug, Clone, PartialEq)]
pub enum FeatureFramework {
    Catch22,
    Entropy,
    Stats,
}

impl FeatureFramework {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "catch22" => Some(Self::Catch22),
            "entropy" => Some(Self::Entropy),
            "stats" => Some(Self::Stats),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Catch22 => "catch22",
            Self::Entropy => "entropy",
            Self::Stats => "stats",
        }
    }
}

/// Configuration for a feature extraction run
#[derive(Debug, Clone)]
pub struct ExtractionConfig {
    pub input_dir: PathBuf,
    pub output_dir: PathBuf,
    pub frameworks: Vec<FeatureFramework>,
    pub workers: usize,
    pub resume: bool,
    /// Entropy-specific: embedding dimensions to compute
    pub dimensions: Vec<usize>,
    /// Entropy-specific: time delays to compute
    pub delays: Vec<usize>,
    /// Sampling frequency in Hz
    pub sampling_frequency: f64,
    /// Maximum NaN percentage before rejecting a segment
    pub nan_threshold: f64,
}

impl Default for ExtractionConfig {
    fn default() -> Self {
        Self {
            input_dir: PathBuf::from("data/raw"),
            output_dir: PathBuf::from("data/features"),
            frameworks: vec![
                FeatureFramework::Catch22,
                FeatureFramework::Entropy,
                FeatureFramework::Stats,
            ],
            workers: num_cpus::get(),
            resume: false,
            dimensions: vec![3, 4, 5, 6, 7],
            delays: vec![1, 2, 3],
            sampling_frequency: 125.0,
            nan_threshold: 0.2,
        }
    }
}

/// Result of extracting all features from a single signal segment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureRow {
    pub file_name: String,
    pub segment_id: String,
    pub signal_length: usize,
    pub nan_percentage: f64,

    // Catch22 features (22)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dn_histogram_mode_5: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dn_histogram_mode_10: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub co_f1ecac: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub co_firstmin_ac: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub co_histogram_ami_even_2_5: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub co_trev_1_num: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub md_hrv_classic_pnn40: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sb_binarystats_mean_longstretch1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sb_transitionmatrix_3ac_sumdiagcov: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pd_periodicitywang_th0_01: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub co_embed2_dist_tau_d_expfit_meandiff: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub in_automutualinfostats_40_gaussian_fmmi: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fc_localsimple_mean1_tauresrat: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dn_outlierinclude_p_001_mdrmd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dn_outlierinclude_n_001_mdrmd: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sp_summaries_welch_rect_area_5_1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sb_motifthree_quantile_hh: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sc_fluctanal_2_dfa_50_1_2_logi_prop_r1: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sp_summaries_welch_rect_centroid: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fc_localsimple_mean3_stderr: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub co_embed2_dist_tau_d_expfit_meandiff_2: Option<f64>,

    // Entropy features (10 features)
    // 8 ordinal (from Paper 2) + 2 amplitude-based (sample, approximate)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub permutation_entropy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub statistical_complexity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fisher_shannon: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fisher_information: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renyi_pe: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub renyi_complexity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tsallis_pe: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tsallis_complexity: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sample_entropy: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub approximate_entropy: Option<f64>,

    // Statistical features (8)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_mean: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_median: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_std: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_skewness: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_kurtosis: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_rms: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_min: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat_max: Option<f64>,
}

impl FeatureRow {
    pub fn new(file_name: String, segment_id: String, signal_length: usize, nan_percentage: f64) -> Self {
        Self {
            file_name,
            segment_id,
            signal_length,
            nan_percentage,
            dn_histogram_mode_5: None,
            dn_histogram_mode_10: None,
            co_f1ecac: None,
            co_firstmin_ac: None,
            co_histogram_ami_even_2_5: None,
            co_trev_1_num: None,
            md_hrv_classic_pnn40: None,
            sb_binarystats_mean_longstretch1: None,
            sb_transitionmatrix_3ac_sumdiagcov: None,
            pd_periodicitywang_th0_01: None,
            co_embed2_dist_tau_d_expfit_meandiff: None,
            in_automutualinfostats_40_gaussian_fmmi: None,
            fc_localsimple_mean1_tauresrat: None,
            dn_outlierinclude_p_001_mdrmd: None,
            dn_outlierinclude_n_001_mdrmd: None,
            sp_summaries_welch_rect_area_5_1: None,
            sb_motifthree_quantile_hh: None,
            sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1: None,
            sc_fluctanal_2_dfa_50_1_2_logi_prop_r1: None,
            sp_summaries_welch_rect_centroid: None,
            fc_localsimple_mean3_stderr: None,
            co_embed2_dist_tau_d_expfit_meandiff_2: None,
            permutation_entropy: None,
            statistical_complexity: None,
            fisher_shannon: None,
            fisher_information: None,
            renyi_pe: None,
            renyi_complexity: None,
            tsallis_pe: None,
            tsallis_complexity: None,
            sample_entropy: None,
            approximate_entropy: None,
            stat_mean: None,
            stat_median: None,
            stat_std: None,
            stat_skewness: None,
            stat_kurtosis: None,
            stat_rms: None,
            stat_min: None,
            stat_max: None,
        }
    }
}

/// Checkpoint state for resumable extraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionCheckpoint {
    pub completed_subjects: Vec<String>,
    pub failed_subjects: Vec<FailedSubject>,
    pub total_subjects: usize,
    pub last_batch: usize,
    pub start_time: String,
    pub last_update: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedSubject {
    pub subject_id: String,
    pub error: String,
    pub timestamp: String,
}

impl ExtractionCheckpoint {
    pub fn new(total_subjects: usize) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            completed_subjects: Vec::new(),
            failed_subjects: Vec::new(),
            total_subjects,
            last_batch: 0,
            start_time: now.clone(),
            last_update: now,
        }
    }

    pub fn is_completed(&self, subject_id: &str) -> bool {
        self.completed_subjects.iter().any(|s| s == subject_id)
    }

    pub fn mark_completed(&mut self, subject_id: String) {
        self.completed_subjects.push(subject_id);
        self.last_update = chrono::Utc::now().to_rfc3339();
    }

    pub fn mark_failed(&mut self, subject_id: String, error: String) {
        self.failed_subjects.push(FailedSubject {
            subject_id,
            error,
            timestamp: chrono::Utc::now().to_rfc3339(),
        });
        self.last_update = chrono::Utc::now().to_rfc3339();
    }
}
