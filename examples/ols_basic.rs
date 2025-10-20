use wc_fir::{fit_ols, OlsOptions};

fn main() {
    let revenue: Vec<f64> = (0..80)
        .map(|i| 150.0 + 0.8 * i as f64 + 15.0 * ((i as f64) / 10.0).sin())
        .collect();
    let marketing: Vec<f64> = (0..80)
        .map(|i| 90.0 + 0.6 * i as f64 + 10.0 * ((i as f64) / 7.0).cos())
        .collect();

    let mut target = vec![0.0; revenue.len()];
    for t in 2..revenue.len() {
        target[t] = 5.0 + 0.55 * revenue[t] + 0.25 * revenue[t - 1] + 0.35 * marketing[t];
    }
    target[0] = target[2];
    target[1] = target[2];

    let fit = fit_ols(
        &[revenue, marketing],
        &target,
        &[2, 1],
        &OlsOptions::default(),
    )
    .unwrap();

    println!("Plain OLS coefficients: {:?}", fit.coeffs);
    for (idx, (scale, percentages)) in fit.per_driver.iter().enumerate() {
        println!("Driver {idx}: scale={scale:.3}, percentages={percentages:?}");
    }
    println!("RMSE={:.4}  R2={:.4}", fit.rmse, fit.r2);
    println!("Intercept={:.4}", fit.intercept);
}
