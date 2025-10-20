//! Comprehensive comparison of different fitting strategies on real AR data.
//!
//! This example demonstrates how to evaluate and compare various approaches:
//! - Manual lag specifications with OLS
//! - Automatic lag selection (Lasso + TS-CV)
//! - Information criteria (BIC)
//! - Correlation screening
//! - Different regularization settings
//!
//! Run with: cargo run --example ar_comparison

use wc_fir::{
    fit_auto, fit_ols, fit_ols_auto_lags, Caps, Guardrails, IcKind, IcSearchKind, LagSelect,
    OlsOptions, Truncation,
};

fn main() {
    // Real accounts receivable data (26 periods)
    let ar = vec![
        165532.91, 483476.76, 470199.48, 539707.66, 366605.04, 359052.43, 433683.29, 474414.57,
        375906.87, 294573.02, 426566.36, 312889.95, 341971.52, 472090.89, 370363.33, 513245.55,
        464515.38, 471388.09, 571130.96, 487430.63, 319406.78, 355640.65, 418705.33, 389474.63,
        323051.77, 342541.41,
    ];

    // Driver 1: Revenue or sales (26 periods - note: one more than AR, offset by 1)
    let driver1 = vec![
        510108.13, 547818.63, 583058.03, 392269.01, 419556.41, 437038.35, 426095.15, 327307.61,
        366352.33, 295984.64, 431445.37, 439684.68, 446277.07, 586428.82, 564331.46, 565251.95,
        633946.64, 526030.39, 518168.85, 471000.37, 458404.53, 409470.74, 533905.26, 452251.71,
        431818.70, 0.0, // Extra period for lagged predictions
    ];

    // Driver 2: Special charges or adjustments (mostly zeros)
    let driver2 = vec![
        0.0, 0.0, 0.0, 0.0, 2666.66, 2666.66, 9905.66, 0.0, 0.0, 0.0, 8500.0, 6000.0, 23000.0,
        0.0, 4000.0, 0.0, 18000.0, 10163.7, 0.0, 5000.0, 0.0, 12000.0, 0.0, 0.0, 0.0, 0.0,
    ];

    // Trim drivers to match AR length (remove last element)
    let driver1 = &driver1[..ar.len()];
    let driver2 = &driver2[..ar.len()];

    println!("=== AR Forecasting: Strategy Comparison ===\n");
    println!("Dataset: {} periods, 2 drivers", ar.len());
    println!(
        "AR range: ${:.0} - ${:.0}\n",
        ar.iter().copied().fold(f64::INFINITY, f64::min),
        ar.iter().copied().fold(f64::NEG_INFINITY, f64::max)
    );

    // ========================================================================
    // Strategy 1: Manual Lag Selection - Conservative (short lags)
    // ========================================================================
    println!("--- Strategy 1: Manual OLS (lags=[2,1], intercept, no ridge) ---");
    let fit1 = fit_ols(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &[2, 1],
        &OlsOptions {
            intercept: true,
            ridge_lambda: 0.0,
            nonnegative: true,
        },
    );

    match fit1 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse);
            println!("  R²:         {:.4}", result.r2);
            println!("  Intercept:  ${:.2}", result.coeffs.last().unwrap_or(&0.0));
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 2: Manual Lag Selection - Moderate lags
    // ========================================================================
    println!("--- Strategy 2: Manual OLS (lags=[3,2], intercept, light ridge) ---");
    let fit2 = fit_ols(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &[3, 2],
        &OlsOptions {
            intercept: true,
            ridge_lambda: 1000.0, // Light regularization
            nonnegative: true,
        },
    );

    match fit2 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse);
            println!("  R²:         {:.4}", result.r2);
            println!("  Intercept:  ${:.2}", result.coeffs.last().unwrap_or(&0.0));
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 3: Manual Lag Selection - Aggressive (longer lags)
    // ========================================================================
    println!("--- Strategy 3: Manual OLS (lags=[5,3], intercept, heavier ridge) ---");
    let fit3 = fit_ols(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &[5, 3],
        &OlsOptions {
            intercept: true,
            ridge_lambda: 5000.0, // More regularization for stability
            nonnegative: true,
        },
    );

    match fit3 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse);
            println!("  R²:         {:.4}", result.r2);
            println!("  Intercept:  ${:.2}", result.coeffs.last().unwrap_or(&0.0));
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 4: Automatic Lag Selection (Lasso + Time-Series CV)
    // ========================================================================
    println!("--- Strategy 4: Auto Lasso (default settings) ---");
    let fit4 = fit_auto(&[driver1.to_vec(), driver2.to_vec()], &ar);

    match fit4 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse);
            println!("  R²:         {:.4}", result.r2);
            println!("  Intercept:  ${:.2}", result.intercept);
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 5: Automatic with Custom Lasso Settings
    // ========================================================================
    println!("--- Strategy 5: Auto Lasso (custom: max_lags=[6,4], more folds) ---");
    let caps5 = Caps {
        per_driver_max: vec![6, 4], // Longer lags for driver 1, moderate for driver 2
        default_cap: 5,
    };
    let mut lasso_settings5 = wc_fir::LassoSettings::default();
    lasso_settings5.folds = 5; // More cross-validation folds

    let strategy5 = LagSelect::Lasso {
        caps: caps5,
        lasso: lasso_settings5,
    };

    let fit5 = fit_ols_auto_lags(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &strategy5,
        &OlsOptions {
            intercept: true,
            ridge_lambda: 0.0,
            nonnegative: true,
        },
        &Guardrails::default(),
        &Truncation { pct_epsilon: 2.0 }, // More aggressive truncation
    );

    match fit5 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse_fit);
            println!("  R²:         {:.4}", result.r2_fit);
            println!(
                "  CV RMSE:    {}",
                result
                    .cv_rmse
                    .map(|v| format!("${:.2}", v))
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!("  Intercept:  ${:.2}", result.intercept);
            println!("  Selected lags: {:?}", result.per_driver_l);
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 6: Information Criterion (BIC)
    // ========================================================================
    println!("--- Strategy 6: IC Selection (BIC, grid search) ---");
    let caps6 = Caps {
        per_driver_max: vec![5, 3],
        default_cap: 5,
    };

    let strategy6 = LagSelect::Ic {
        caps: caps6,
        criterion: IcKind::Bic,
        search: IcSearchKind::Grid,
    };

    let fit6 = fit_ols_auto_lags(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &strategy6,
        &OlsOptions {
            intercept: true,
            ridge_lambda: 0.0,
            nonnegative: true,
        },
        &Guardrails::default(),
        &Truncation { pct_epsilon: 1.0 },
    );

    match fit6 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse_fit);
            println!("  R²:         {:.4}", result.r2_fit);
            println!(
                "  CV RMSE:    {}",
                result
                    .cv_rmse
                    .map(|v| format!("${:.2}", v))
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!("  Intercept:  ${:.2}", result.intercept);
            println!("  Selected lags: {:?}", result.per_driver_l);
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 7: Correlation Screening
    // ========================================================================
    println!("--- Strategy 7: Correlation Screening (threshold=0.3, top_k=10) ---");
    let caps7 = Caps {
        per_driver_max: vec![4, 3],
        default_cap: 4,
    };

    let strategy7 = LagSelect::Screen {
        caps: caps7,
        top_k: 10,          // Keep top 10 most correlated features
        min_abs_corr: 0.3,  // Minimum absolute correlation threshold
        prune_bic: true,    // Prune further using BIC
    };

    let fit7 = fit_ols_auto_lags(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &strategy7,
        &OlsOptions {
            intercept: true,
            ridge_lambda: 0.0,
            nonnegative: true,
        },
        &Guardrails::default(),
        &Truncation { pct_epsilon: 1.0 },
    );

    match fit7 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse_fit);
            println!("  R²:         {:.4}", result.r2_fit);
            println!(
                "  CV RMSE:    {}",
                result
                    .cv_rmse
                    .map(|v| format!("${:.2}", v))
                    .unwrap_or_else(|| "N/A".to_string())
            );
            println!("  Intercept:  ${:.2}", result.intercept);
            println!("  Selected lags: {:?}", result.per_driver_l);
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Strategy 8: No Regularization, Allow Negative Coefficients
    // ========================================================================
    println!("--- Strategy 8: Manual OLS (lags=[3,2], no constraints) ---");
    let fit8 = fit_ols(
        &[driver1.to_vec(), driver2.to_vec()],
        &ar,
        &[3, 2],
        &OlsOptions {
            intercept: true,
            ridge_lambda: 0.0,
            nonnegative: false, // Allow negative coefficients
        },
    );

    match fit8 {
        Ok(result) => {
            println!("  RMSE:       ${:.2}", result.rmse);
            println!("  R²:         {:.4}", result.r2);
            println!("  Intercept:  ${:.2}", result.coeffs.last().unwrap_or(&0.0));
            println!("  Raw coefficients: {:?}", result.coeffs);
            for (i, (scale, taps)) in result.per_driver.iter().enumerate() {
                println!("  Driver {}: scale={:.4}, taps={:?}", i + 1, scale, taps);
            }
        }
        Err(e) => println!("  Error: {}", e),
    }
    println!();

    // ========================================================================
    // Summary and Recommendations
    // ========================================================================
    println!("=== Summary & Recommendations ===\n");
    println!("Key Observations:");
    println!("  - Driver 2 is sparse (mostly zeros), likely low predictive power");
    println!("  - With limited data (26 periods), simpler models may generalize better");
    println!("  - Ridge regularization helps when using longer lag lengths");
    println!("  - Auto selection methods can help avoid overfitting\n");
    println!("Recommended Approach:");
    println!("  1. Start with automatic lag selection (Strategy 4 or 5)");
    println!("  2. Validate with hold-out period if you plan to forecast");
    println!("  3. Consider BIC if interpretability is crucial (Strategy 6)");
    println!("  4. Use ridge regularization for longer lags (Strategy 3)");
}
