//! Comprehensive and systematic comparison of different fitting strategies on real AR data.
//!
//! This script exhaustively tests a wide range of parameter combinations across all
//! major fitting strategies offered by the `wc_fir` library:
//! - Manual OLS with various lag and regularization settings.
//! - Automatic lag selection using Lasso, Information Criteria (IC), Correlation Screening,
//!   and Prefix CV.
//!
//! It iterates through different settings for:
//! - Lag structures (manual and automatic caps)
//! - OLS options (intercept, ridge lambda, non-negativity, scale constraints)
//! - Strategy-specific parameters (e.g., Lasso folds, IC kind, Screening thresholds)
//! - Guardrails and Truncation settings
//!
//! All settings and corresponding results (RMSE, R², selected coefficients, errors, etc.)
//! are systematically logged to a timestamped CSV file for later analysis.
//!
//! ---
//!
//! BEFORE RUNNING: Add the following to your Cargo.toml:
//!
//! [dependencies]
//! csv = "1.3"
//! serde = { version = "1.0", features = ["derive"] }
//! chrono = "0.4"
//!
//! ---
//!
//! Run with: cargo run --release --example ar_comparison

use chrono::Local;
use serde::Serialize;
use std::fs::File;
use wc_fir::{
    fit_ols, fit_ols_auto_lags, AutoLagResult, Caps, FirError, Guardrails, IcKind, IcSearchKind,
    LagSelect, LassoSettings, OlsFit, OlsOptions, Truncation,
};

/// A unified struct to hold all settings and results for a single experiment run.
/// This struct is serialized directly into a CSV row.
#[derive(Debug, Serialize)]
struct ExperimentResult {
    // === IDENTIFICATION ===
    id: usize,
    strategy_group: String,

    // === STATUS ===
    status: String,
    error_message: String,

    // === INPUT SETTINGS ===
    // --- Strategy Specific ---
    manual_lags: String,
    caps_per_driver_max: String,
    lasso_folds: String,
    lasso_one_se_rule: String,
    ic_kind: String,
    ic_search: String,
    screen_top_k: String,
    screen_min_corr: String,
    screen_prune_bic: String,
    prefix_cv_folds: String,
    prefix_cv_shared: String,

    // --- Common Settings ---
    ols_intercept: bool,
    ols_ridge_lambda: f64,
    ols_nonnegative: bool,
    ols_constrain_scale_0_1: bool,
    guardrails_max_total_lag: usize,
    truncation_pct_epsilon: f64,

    // === OUTPUT METRICS ===
    rmse: String,
    r2: String,
    cv_rmse: String,
    intercept: String,
    selected_lags: String,
    num_coeffs: usize,
    driver1_scale: String,
    driver1_taps: String,
    driver2_scale: String,
    driver2_taps: String,
}

impl ExperimentResult {
    /// Creates a new, empty result struct for a given strategy group and ID.
    fn new(id: usize, strategy_group: &str) -> Self {
        Self {
            id,
            strategy_group: strategy_group.to_string(),
            status: "Pending".to_string(),
            error_message: "N/A".to_string(),
            manual_lags: "N/A".to_string(),
            caps_per_driver_max: "N/A".to_string(),
            lasso_folds: "N/A".to_string(),
            lasso_one_se_rule: "N/A".to_string(),
            ic_kind: "N/A".to_string(),
            ic_search: "N/A".to_string(),
            screen_top_k: "N/A".to_string(),
            screen_min_corr: "N/A".to_string(),
            screen_prune_bic: "N/A".to_string(),
            prefix_cv_folds: "N/A".to_string(),
            prefix_cv_shared: "N/A".to_string(),
            ols_intercept: false,
            ols_ridge_lambda: 0.0,
            ols_nonnegative: false,
            ols_constrain_scale_0_1: false,
            guardrails_max_total_lag: 0,
            truncation_pct_epsilon: 0.0,
            rmse: "N/A".to_string(),
            r2: "N/A".to_string(),
            cv_rmse: "N/A".to_string(),
            intercept: "N/A".to_string(),
            selected_lags: "N/A".to_string(),
            num_coeffs: 0,
            driver1_scale: "N/A".to_string(),
            driver1_taps: "N/A".to_string(),
            driver2_scale: "N/A".to_string(),
            driver2_taps: "N/A".to_string(),
        }
    }
}

/// A wrapper for the two possible successful fit results from the library.
enum FitResult {
    Ols(OlsFit),
    Auto(AutoLagResult),
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // --- DATA SETUP ---
    let ar = vec![
        165532.91, 483476.76, 470199.48, 539707.66, 366605.04, 359052.43, 433683.29, 474414.57,
        375906.87, 294573.02, 426566.36, 312889.95, 341971.52, 472090.89, 370363.33, 513245.55,
        464515.38, 471388.09, 571130.96, 487430.63, 319406.78, 355640.65, 418705.33, 389474.63,
        323051.77, 342541.41,
    ];
    let driver1 = vec![
        510108.13, 547818.63, 583058.03, 392269.01, 419556.41, 437038.35, 426095.15, 327307.61,
        366352.33, 295984.64, 431445.37, 439684.68, 446277.07, 586428.82, 564331.46, 565251.95,
        633946.64, 526030.39, 518168.85, 471000.37, 458404.53, 409470.74, 533905.26, 452251.71,
        431818.70, 0.0,
    ];
    let driver2 = vec![
        0.0, 0.0, 0.0, 0.0, 2666.66, 2666.66, 9905.66, 0.0, 0.0, 0.0, 8500.0, 6000.0, 23000.0, 0.0,
        4000.0, 0.0, 18000.0, 10163.7, 0.0, 5000.0, 0.0, 12000.0, 0.0, 0.0, 0.0, 0.0,
    ];
    let driver1 = &driver1[..ar.len()];
    let driver2 = &driver2[..ar.len()];
    let drivers = [driver1.to_vec(), driver2.to_vec()];

    // --- CSV SETUP ---
    let timestamp = Local::now().format("%Y%m%d-%H%M%S");
    let filename = format!("ar_comparison_results_{}.csv", timestamp);
    let file = File::create(&filename)?;
    let mut wtr = csv::Writer::from_writer(file);
    let mut experiment_counter: usize = 0;

    println!("Starting comprehensive model comparison...");
    println!("Results will be saved to: {}", filename);

    // --- PARAMETER SPACE DEFINITION ---
    let ols_options_combinations = generate_ols_options();
    let caps_options = vec![vec![4, 2], vec![6, 4], vec![8, 5]];
    let truncation_epsilons = vec![0.01, 0.05];
    let guardrail_lags = vec![8, 12];

    // --- RUN EXPERIMENTS ---
    run_manual_ols_experiments(
        &mut wtr,
        &mut experiment_counter,
        &drivers,
        &ar,
        &ols_options_combinations,
    )?;

    run_auto_experiments(
        &mut wtr,
        &mut experiment_counter,
        &drivers,
        &ar,
        &ols_options_combinations,
        &caps_options,
        &truncation_epsilons,
        &guardrail_lags,
    )?;

    wtr.flush()?;
    println!("\nFinished! Ran a total of {} experiments.", experiment_counter);
    println!("Results saved in '{}'.", filename);

    Ok(())
}

/// Generates a vector of `OlsOptions` to iterate over.
fn generate_ols_options() -> Vec<OlsOptions> {
    let mut combinations = Vec::new();
    let ridge_lambdas = vec![0.0, 0.1, 100.0, 10000.0];
    let nonnegatives = vec![true, false];
    let constrain_scales = vec![true, false];

    for &ridge_lambda in &ridge_lambdas {
        for &nonnegative in &nonnegatives {
            for &constrain_scale_0_1 in &constrain_scales {
                combinations.push(OlsOptions {
                    intercept: true, // Keep intercept fixed for this study
                    ridge_lambda,
                    nonnegative,
                    constrain_scale_0_1,
                });
            }
        }
    }
    combinations
}

/// Runs all experiments for the manual `fit_ols` strategy.
fn run_manual_ols_experiments(
    wtr: &mut csv::Writer<File>,
    counter: &mut usize,
    drivers: &[Vec<f64>],
    target: &[f64],
    ols_options_list: &[OlsOptions],
) -> Result<(), Box<dyn std::error::Error>> {
    println!("\n--- Running Manual OLS Experiments ---");
    let manual_lags_options = vec![vec![2, 1], vec![3, 2], vec![5, 3]];

    for lags in &manual_lags_options {
        for ols_opts in ols_options_list {
            *counter += 1;
            print!("\r  > Running experiment #{}", *counter);

            let mut result = ExperimentResult::new(*counter, "Manual");
            result.manual_lags = format!("{:?}", lags);
            result.ols_intercept = ols_opts.intercept;
            result.ols_ridge_lambda = ols_opts.ridge_lambda;
            result.ols_nonnegative = ols_opts.nonnegative;
            result.ols_constrain_scale_0_1 = ols_opts.constrain_scale_0_1;

            let fit_result = fit_ols(drivers, target, lags, ols_opts).map(FitResult::Ols);
            populate_outputs(&mut result, fit_result);

            wtr.serialize(&result)?;
        }
    }
    Ok(())
}

/// Runs all experiments for the automatic strategies (`fit_ols_auto_lags`).
fn run_auto_experiments(
    wtr: &mut csv::Writer<File>,
    counter: &mut usize,
    drivers: &[Vec<f64>],
    target: &[f64],
    ols_options_list: &[OlsOptions],
    caps_options: &[Vec<usize>],
    trunc_eps_options: &[f64],
    guard_lag_options: &[usize],
) -> Result<(), Box<dyn std::error::Error>> {
    let strategies_to_run = [
        "Lasso",
        "IC",
        "Screen",
        "PrefixCV",
    ];

    for strategy_name in &strategies_to_run {
        println!("\n--- Running {} Experiments ---", strategy_name);
        for caps_val in caps_options {
            for &trunc_eps in trunc_eps_options {
                for &guard_lag in guard_lag_options {
                    for ols_opts in ols_options_list {
                        // This inner part generates the specific LagSelect variants
                        let lag_select_variants =
                            generate_lag_select_variants(strategy_name, caps_val.clone());

                        for lag_select in lag_select_variants {
                            *counter += 1;
                            print!("\r  > Running experiment #{}", *counter);

                            let mut result = ExperimentResult::new(*counter, strategy_name);
                            
                            // Populate settings
                            result.caps_per_driver_max = format!("{:?}", caps_val);
                            result.truncation_pct_epsilon = trunc_eps;
                            result.guardrails_max_total_lag = guard_lag;
                            result.ols_intercept = ols_opts.intercept;
                            result.ols_ridge_lambda = ols_opts.ridge_lambda;
                            result.ols_nonnegative = ols_opts.nonnegative;
                            result.ols_constrain_scale_0_1 = ols_opts.constrain_scale_0_1;
                            populate_strategy_settings(&mut result, &lag_select);

                            let guardrails = Guardrails {
                                max_total_lag: guard_lag,
                                ..Default::default()
                            };
                            let truncation = Truncation { pct_epsilon: trunc_eps };

                            let fit_result = fit_ols_auto_lags(
                                drivers,
                                target,
                                &lag_select,
                                ols_opts,
                                &guardrails,
                                &truncation,
                            )
                            .map(FitResult::Auto);

                            populate_outputs(&mut result, fit_result);
                            wtr.serialize(&result)?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Generates the specific `LagSelect` enum variants to test for a given strategy group.
fn generate_lag_select_variants(strategy_name: &str, caps_vec: Vec<usize>) -> Vec<LagSelect> {
    let mut variants = Vec::new();
    let caps = Caps { per_driver_max: caps_vec, default_cap: 5 };

    match strategy_name {
        "Lasso" => {
            for &folds in &[3, 5] {
                for &one_se_rule in &[true, false] {
                    variants.push(LagSelect::Lasso {
                        caps: caps.clone(),
                        lasso: LassoSettings { folds, one_se_rule, ..Default::default() },
                    });
                }
            }
        }
        "IC" => {
            for &criterion in &[IcKind::Bic, IcKind::Aic] {
                for &search in &[IcSearchKind::Grid, IcSearchKind::GreedyForward] {
                    variants.push(LagSelect::Ic { caps: caps.clone(), criterion, search });
                }
            }
        }
        "Screen" => {
            for &top_k in &[5, 10] {
                for &min_abs_corr in &[0.1, 0.3] {
                    for &prune_bic in &[true, false] {
                        variants.push(LagSelect::Screen {
                            caps: caps.clone(), top_k, min_abs_corr, prune_bic,
                        });
                    }
                }
            }
        }
        "PrefixCV" => {
            for &folds in &[3, 5] {
                for &shared in &[true, false] {
                    variants.push(LagSelect::PrefixCv { caps: caps.clone(), folds, shared });
                }
            }
        }
        _ => {}
    }
    variants
}

/// Populates the strategy-specific settings in the result struct.
fn populate_strategy_settings(result: &mut ExperimentResult, strategy: &LagSelect) {
    match strategy {
        LagSelect::Lasso { lasso, .. } => {
            result.lasso_folds = lasso.folds.to_string();
            result.lasso_one_se_rule = lasso.one_se_rule.to_string();
        }
        LagSelect::Ic { criterion, search, .. } => {
            result.ic_kind = format!("{:?}", criterion);
            result.ic_search = format!("{:?}", search);
        }
        LagSelect::Screen { top_k, min_abs_corr, prune_bic, .. } => {
            result.screen_top_k = top_k.to_string();
            result.screen_min_corr = min_abs_corr.to_string();
            result.screen_prune_bic = prune_bic.to_string();
        }
        LagSelect::PrefixCv { folds, shared, .. } => {
            result.prefix_cv_folds = folds.to_string();
            result.prefix_cv_shared = shared.to_string();
        }
    }
}

/// Populates the output fields of the result struct from a fit result or an error.
fn populate_outputs(result: &mut ExperimentResult, fit_result: Result<FitResult, FirError>) {
    match fit_result {
        Ok(fit) => {
            result.status = "Success".to_string();
            match fit {
                FitResult::Ols(ols_fit) => {
                    result.rmse = format!("{:.2}", ols_fit.rmse);
                    result.r2 = format!("{:.4}", ols_fit.r2);
                    result.intercept = format!("{:.2}", ols_fit.intercept);
                    result.num_coeffs = ols_fit.coeffs.len();
                    if !ols_fit.per_driver.is_empty() {
                        result.driver1_scale = format!("{:.4}", ols_fit.per_driver[0].0);
                        result.driver1_taps = format!("{:?}", ols_fit.per_driver[0].1);
                        result.driver2_scale = format!("{:.4}", ols_fit.per_driver[1].0);
                        result.driver2_taps = format!("{:?}", ols_fit.per_driver[1].1);
                    }
                }
                FitResult::Auto(auto_fit) => {
                    result.rmse = format!("{:.2}", auto_fit.rmse_fit);
                    result.r2 = format!("{:.4}", auto_fit.r2_fit);
                    result.cv_rmse = auto_fit.cv_rmse.map(|v| format!("{:.2}", v)).unwrap_or("N/A".to_string());
                    result.intercept = format!("{:.2}", auto_fit.intercept);
                    result.selected_lags = format!("{:?}", auto_fit.per_driver_l);
                    result.num_coeffs = auto_fit.coeffs.len();
                    if !auto_fit.per_driver.is_empty() {
                        result.driver1_scale = format!("{:.4}", auto_fit.per_driver[0].0);
                        result.driver1_taps = format!("{:?}", auto_fit.per_driver[0].1);
                        result.driver2_scale = format!("{:.4}", auto_fit.per_driver[1].0);
                        result.driver2_taps = format!("{:?}", auto_fit.per_driver[1].1);
                    }
                }
            }
        }
        Err(e) => {
            result.status = "Error".to_string();
            result.error_message = e.to_string().replace(',', ";"); // Avoid breaking CSV format
        }
    }
}