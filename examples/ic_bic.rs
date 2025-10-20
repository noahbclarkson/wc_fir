use wc_fir::{
    fit_ols_auto_lags, Caps, Guardrails, IcKind, IcSearchKind, LagSelect, OlsOptions, Truncation,
};

fn main() {
    let len = 60;
    let demand: Vec<f64> = (0..len).map(|i| 1.0 + 0.07 * i as f64).collect();
    let lead: Vec<f64> = (0..len).map(|i| 0.6 + 0.04 * i as f64).collect();

    let mut inventory = vec![0.0; len];
    for t in 3..len {
        inventory[t] = 0.5
            + 0.35 * demand[t]
            + 0.2 * demand[t - 1]
            + 0.1 * demand[t - 2]
            + 0.22 * lead[t]
            + 0.07 * lead[t - 1];
    }
    let seeds = [inventory[3], inventory[4], inventory[5]];
    inventory[..3].copy_from_slice(&seeds);

    let caps = Caps {
        per_driver_max: vec![3, 2],
        default_cap: 3,
    };
    let strategy = LagSelect::Ic {
        caps,
        criterion: IcKind::Bic,
        search: IcSearchKind::Grid,
    };
    let result = fit_ols_auto_lags(
        &[demand, lead],
        &inventory,
        &strategy,
        &OlsOptions::default(),
        &Guardrails::default(),
        &Truncation::default(),
    )
    .unwrap();

    println!("BIC-selected lags: {:?}", result.per_driver_l);
    for (idx, (scale, percentages)) in result.per_driver.iter().enumerate() {
        println!("Driver {idx}: scale={scale:.3}, taps={percentages:?}");
    }
    println!("RMSE={:.6}  R2={:.6}", result.rmse_fit, result.r2_fit);
    println!("Intercept={:.6}", result.intercept);
}
