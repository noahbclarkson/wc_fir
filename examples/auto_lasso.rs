use wc_fir::{
    fit_ols_auto_lags, Caps, Guardrails, LagSelect, LassoSettings, OlsOptions, Truncation,
};

fn main() {
    let len = 40;
    let mut d1 = Vec::with_capacity(len);
    let mut d2 = Vec::with_capacity(len);
    for i in 0..len {
        d1.push(120.0 + i as f64 * 1.5);
        d2.push(80.0 + i as f64);
    }

    let mut target = vec![0.0; len];
    for t in 2..len {
        target[t] = 12.0 + 0.6 * d1[t] + 0.25 * d1[t - 1] + 0.15 * d2[t];
    }
    target[0] = target[2];
    target[1] = target[2];

    let caps = Caps {
        per_driver_max: vec![3, 2],
        default_cap: 3,
    };
    let strategy = LagSelect::Lasso {
        caps,
        lasso: LassoSettings::default(),
    };

    let result = fit_ols_auto_lags(
        &[d1, d2],
        &target,
        &strategy,
        &OlsOptions::default(),
        &Guardrails::default(),
        &Truncation::default(),
    )
    .unwrap();

    println!("Lasso-selected lags: {:?}", result.per_driver_l);
    for (idx, (scale, percentages)) in result.per_driver.iter().enumerate() {
        println!("Driver {idx}: scale={scale:.3}, taps={percentages:?}");
    }
    println!("RMSE={:.4}  R2={:.4}", result.rmse_fit, result.r2_fit);
    if let Some(cv) = result.cv_rmse {
        println!("CV RMSE={cv:.4}");
    }
    println!("Intercept={:.4}", result.intercept);
}
