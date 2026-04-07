pub mod autocorrelation;
pub mod distribution;
pub mod fluctuation;
pub mod local;
pub mod binary;
pub mod successive;
pub mod helpers;

use anyhow::Result;

/// All 22 Catch22 features computed for a single signal.
#[derive(Debug, Clone)]
pub struct Catch22Features {
    pub dn_histogram_mode_5: f64,
    pub dn_histogram_mode_10: f64,
    pub co_f1ecac: f64,
    pub co_firstmin_ac: f64,
    pub co_histogram_ami_even_2_5: f64,
    pub co_trev_1_num: f64,
    pub md_hrv_classic_pnn40: f64,
    pub sb_binarystats_mean_longstretch1: f64,
    pub sb_transitionmatrix_3ac_sumdiagcov: f64,
    pub pd_periodicitywang_th0_01: f64,
    pub co_embed2_dist_tau_d_expfit_meandiff: f64,
    pub in_automutualinfostats_40_gaussian_fmmi: f64,
    pub fc_localsimple_mean1_tauresrat: f64,
    pub dn_outlierinclude_p_001_mdrmd: f64,
    pub dn_outlierinclude_n_001_mdrmd: f64,
    pub sp_summaries_welch_rect_area_5_1: f64,
    pub sb_motifthree_quantile_hh: f64,
    pub sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1: f64,
    pub sc_fluctanal_2_dfa_50_1_2_logi_prop_r1: f64,
    pub sp_summaries_welch_rect_centroid: f64,
    pub fc_localsimple_mean3_stderr: f64,
    pub co_embed2_dist_tau_d_expfit_meandiff_2: f64,
}

impl Catch22Features {
    /// Compute all 22 features from a raw signal.
    pub fn compute(signal: &[f64]) -> Result<Self> {
        let ac = helpers::autocorrelation_fn(signal);
        let tau = helpers::first_min_ac(&ac);

        Ok(Self {
            dn_histogram_mode_5: distribution::dn_histogram_mode(signal, 5),
            dn_histogram_mode_10: distribution::dn_histogram_mode(signal, 10),
            co_f1ecac: autocorrelation::co_f1ecac(&ac),
            co_firstmin_ac: autocorrelation::co_firstmin_ac(&ac),
            co_histogram_ami_even_2_5: autocorrelation::co_histogram_ami_even_2_5(signal),
            co_trev_1_num: autocorrelation::co_trev_1_num(signal),
            md_hrv_classic_pnn40: successive::md_hrv_classic_pnn40(signal),
            sb_binarystats_mean_longstretch1: binary::sb_binarystats_mean_longstretch1(signal),
            sb_transitionmatrix_3ac_sumdiagcov: binary::sb_transitionmatrix_3ac_sumdiagcov(signal),
            pd_periodicitywang_th0_01: successive::pd_periodicitywang_th0_01(signal),
            co_embed2_dist_tau_d_expfit_meandiff: autocorrelation::co_embed2_dist_tau_d_expfit_meandiff(signal, tau),
            in_automutualinfostats_40_gaussian_fmmi: local::in_automutualinfostats_40_gaussian_fmmi(signal),
            fc_localsimple_mean1_tauresrat: local::fc_localsimple_mean1_tauresrat(signal, tau),
            dn_outlierinclude_p_001_mdrmd: distribution::dn_outlierinclude_mdrmd(signal, true),
            dn_outlierinclude_n_001_mdrmd: distribution::dn_outlierinclude_mdrmd(signal, false),
            sp_summaries_welch_rect_area_5_1: successive::sp_summaries_welch_rect_area_5_1(signal),
            sb_motifthree_quantile_hh: binary::sb_motifthree_quantile_hh(signal),
            sc_fluctanal_2_rsrangefit_50_1_logi_prop_r1: fluctuation::sc_fluctanal_rsrangefit(signal),
            sc_fluctanal_2_dfa_50_1_2_logi_prop_r1: fluctuation::sc_fluctanal_dfa(signal),
            sp_summaries_welch_rect_centroid: successive::sp_summaries_welch_rect_centroid(signal),
            fc_localsimple_mean3_stderr: local::fc_localsimple_mean3_stderr(signal),
            co_embed2_dist_tau_d_expfit_meandiff_2: autocorrelation::co_embed2_dist_tau_d_expfit_meandiff(signal, tau.max(1)),
        })
    }
}
