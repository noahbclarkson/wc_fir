use wc_fir::{fit_ols_auto_lags, Caps, Guardrails, LagSelect, OlsOptions, Truncation};

fn main() {
    let len = 50;
    let bookings: Vec<f64> = (0..len)
        .map(|i| 1.5 + 0.08 * i as f64 + 0.02 * ((i as f64) / 6.0).sin())
        .collect();
    let cash: Vec<f64> = (0..len)
        .map(|i| 1.0 + 0.05 * i as f64 + 0.01 * ((i as f64) / 4.0).cos())
        .collect();
    let shipments: Vec<f64> = (0..len)
        .map(|i| 1.2 + 0.07 * i as f64 + 0.03 * ((i as f64) / 5.0).sin())
        .collect();

    let mut ar = vec![0.0; len];
    for t in 3..len {
        ar[t] =
            0.6 + 0.48 * bookings[t] + 0.22 * bookings[t - 1] + 0.3 * shipments[t] + 0.12 * cash[t];
    }
    let seeds = [ar[3], ar[4], ar[5]];
    ar[..3].copy_from_slice(&seeds);

    let caps = Caps {
        per_driver_max: vec![4, 2, 3],
        default_cap: 4,
    };
    let strategy = LagSelect::Screen {
        caps,
        top_k: 5,
        min_abs_corr: 0.2,
        prune_bic: true,
    };
    let result = fit_ols_auto_lags(
        &[bookings, cash, shipments],
        &ar,
        &strategy,
        &OlsOptions::default(),
        &Guardrails::default(),
        &Truncation::default(),
    )
    .unwrap();

    println!("Screening kept {} taps", result.coeffs.len());
    for (idx, (scale, percentages)) in result.per_driver.iter().enumerate() {
        println!("Driver {idx}: scale={scale:.3}, taps={percentages:?}");
    }
    println!("RMSE={:.6}  R2={:.6}", result.rmse_fit, result.r2_fit);
    if let Some(cv) = result.cv_rmse {
        println!("CV RMSE={cv:.6}");
    } else {
        println!("CV RMSE=N/A");
    }
    println!("Intercept={:.6}", result.intercept);
}
