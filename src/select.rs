use std::cmp::Ordering;
use std::collections::HashSet;

use crate::defaults::{DEFAULT_MAX_PARAMS_RATIO, DEFAULT_MAX_TOTAL_LAG};
use crate::fir::build_design;
use crate::lasso::{lasso_path, ts_cv_rmse_for_lambdas};
use crate::ols::{compute_metrics, reconstruct_fit};
use crate::types::{
    AutoLagResult, Caps, FirError, Guardrails, IcKind, IcSearchKind, LagSelect, OlsFit, OlsOptions,
    Truncation,
};
use linfa::dataset::Dataset;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};

const EPS: f64 = 1e-9;

#[derive(Clone, Copy)]
struct ScreenParams {
    top_k: usize,
    min_abs_corr: f64,
    prune_bic: bool,
}

pub fn fit_auto(drivers: &[Vec<f64>], target: &[f64]) -> Result<OlsFit, FirError> {
    let strategy = LagSelect::default();
    let opts = OlsOptions::default();
    let guards = Guardrails::default();
    let trunc = Truncation::default();
    let auto = fit_ols_auto_lags(drivers, target, &strategy, &opts, &guards, &trunc)?;

    Ok(OlsFit {
        coeffs: auto.coeffs,
        per_driver: auto.per_driver,
        rmse: auto.rmse_fit,
        r2: auto.r2_fit,
        n_rows: auto.rows_used,
        intercept: auto.intercept,
    })
}

pub fn fit_ols_auto_lags(
    drivers: &[Vec<f64>],
    target: &[f64],
    strategy: &LagSelect,
    opts: &OlsOptions,
    guards: &Guardrails,
    trunc: &Truncation,
) -> Result<AutoLagResult, FirError> {
    if drivers.is_empty() {
        return Err(FirError::EmptyInput);
    }
    if drivers[0].len() != target.len() {
        return Err(FirError::LengthMismatch);
    }

    // Validate truncation epsilon to prevent self-destructive values
    let trunc = trunc.clone().validated();

    let caps_cfg = match strategy {
        LagSelect::Lasso { caps, .. } => caps,
        LagSelect::Ic { caps, .. } => caps,
        LagSelect::Screen { caps, .. } => caps,
    };
    let mut per_driver_caps = derive_caps(drivers.len(), caps_cfg)?;
    enforce_total_cap(&mut per_driver_caps, guards.max_total_lag);

    // Build maximal design matrix
    let (x_full, burn_in) = build_design(drivers, &per_driver_caps)?;
    let rows = x_full.nrows();
    let y = Array1::from(target[burn_in..].to_vec());
    if y.len() != rows {
        return Err(FirError::LengthMismatch);
    }

    let column_map = build_column_map(&per_driver_caps);

    // Run strategy
    let (mask, cv_rmse) = match strategy {
        LagSelect::Lasso { lasso, .. } => {
            let (lambda_grid, path) = lasso_path(&x_full, &y, lasso, guards, None)?;
            let (cv_mean, cv_se) =
                ts_cv_rmse_for_lambdas(&x_full, &y, &lambda_grid, lasso, guards, lasso.folds)?;
            let best_idx = pick_lambda(&cv_mean, &cv_se, lasso.one_se_rule);
            let mut mask = path[best_idx].active_mask.clone();
            prune_mask_with_guardrails(&mut mask, &path[best_idx].beta, rows, guards)?;
            (mask, Some(cv_mean[best_idx]))
        }
        LagSelect::Ic {
            criterion, search, ..
        } => {
            let mask = ic_select(
                drivers,
                target,
                &per_driver_caps,
                *criterion,
                *search,
                opts,
                guards,
            )?;
            (mask, None)
        }
        LagSelect::Screen {
            top_k,
            min_abs_corr,
            prune_bic,
            ..
        } => {
            let params = ScreenParams {
                top_k: *top_k,
                min_abs_corr: *min_abs_corr,
                prune_bic: *prune_bic,
            };
            let mask = screen_select(&x_full, &y, &column_map, params, opts, guards)?;
            (mask, None)
        }
    };

    if !mask.iter().any(|&m| m) {
        return Err(FirError::Guardrail(
            "no features selected after guardrail pruning".to_string(),
        ));
    }

    // Slice matrix to active features
    let (x_sel, _selected_cols) = slice_columns(&x_full, &mask);
    let dataset = Dataset::new(x_sel.clone(), y.clone());
    let lin = LinearRegression::new().with_intercept(opts.intercept);
    let fitted = lin
        .fit(&dataset)
        .map_err(|e| FirError::Linalg(format!("{:?}", e)))?;
    let beta_sel = fitted.params().to_vec();
    let intercept = if opts.intercept {
        fitted.intercept()
    } else {
        0.0
    };

    // Expand beta to full vector aligned with mask
    let full_beta = expand_beta(&mask, &beta_sel)?;
    let per_driver_l = infer_selected_lags(&mask, &per_driver_caps);
    let (per_driver, coeffs_trimmed) = map_to_scales_and_percentages(
        &full_beta,
        &per_driver_caps,
        &per_driver_l,
        opts.nonnegative,
        trunc.pct_epsilon,
    );

    let y_hat = reconstruct_fit(
        drivers,
        &per_driver,
        &per_driver_l,
        burn_in,
        rows,
        intercept,
    );
    let (rmse, r2) = compute_metrics(&y, &y_hat);

    Ok(AutoLagResult {
        per_driver_l,
        coeffs: coeffs_trimmed,
        per_driver,
        rmse_fit: rmse,
        r2_fit: r2,
        cv_rmse,
        burn_in,
        rows_used: rows,
        intercept,
    })
}

fn derive_caps(num_drivers: usize, caps: &Caps) -> Result<Vec<usize>, FirError> {
    if caps.per_driver_max.is_empty() {
        Ok(vec![caps.default_cap; num_drivers])
    } else if caps.per_driver_max.len() == num_drivers {
        Ok(caps.per_driver_max.clone())
    } else {
        Err(FirError::InvalidConfig(format!(
            "caps length {} does not match drivers {}",
            caps.per_driver_max.len(),
            num_drivers
        )))
    }
}

fn enforce_total_cap(per_driver_caps: &mut [usize], max_total: usize) {
    let mut total: usize = per_driver_caps.iter().sum();
    if max_total == 0 {
        return;
    }
    while total > max_total {
        if let Some((idx, _)) = per_driver_caps
            .iter()
            .enumerate()
            .max_by_key(|&(_, cap)| cap)
        {
            if per_driver_caps[idx] > 1 {
                per_driver_caps[idx] -= 1;
            } else {
                break;
            }
        } else {
            break;
        }
        total = per_driver_caps.iter().sum();
    }
}

fn build_column_map(per_driver_caps: &[usize]) -> Vec<(usize, usize)> {
    let mut map = Vec::new();
    for (driver_idx, &cap) in per_driver_caps.iter().enumerate() {
        for lag in 0..cap {
            map.push((driver_idx, lag));
        }
    }
    map
}

fn slice_columns(x: &Array2<f64>, mask: &[bool]) -> (Array2<f64>, Vec<usize>) {
    let selected: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &m)| if m { Some(idx) } else { None })
        .collect();
    let mut out = Array2::<f64>::zeros((x.nrows(), selected.len()));

    for (dest_idx, &src_idx) in selected.iter().enumerate() {
        let src_col = x.column(src_idx);
        let mut dest_col = out.column_mut(dest_idx);
        dest_col.assign(&src_col);
    }

    (out, selected)
}

fn expand_beta(mask: &[bool], beta_sel: &[f64]) -> Result<Vec<f64>, FirError> {
    let active = mask.iter().filter(|&&m| m).count();
    if active != beta_sel.len() {
        return Err(FirError::InvalidConfig(format!(
            "selected beta length {} does not match mask active count {}",
            beta_sel.len(),
            active
        )));
    }
    let mut expanded = vec![0.0; mask.len()];
    let mut iter = beta_sel.iter();
    for (idx, &m) in mask.iter().enumerate() {
        if m {
            if let Some(&val) = iter.next() {
                expanded[idx] = val;
            }
        }
    }
    Ok(expanded)
}

fn infer_selected_lags(mask: &[bool], per_driver_caps: &[usize]) -> Vec<usize> {
    let mut per_driver_l = vec![0; per_driver_caps.len()];
    let mut offset = 0;
    for (driver_idx, &cap) in per_driver_caps.iter().enumerate() {
        let mut max_lag = None;
        for lag in 0..cap {
            if mask[offset + lag] {
                max_lag = Some(lag);
            }
        }
        per_driver_l[driver_idx] = max_lag.map(|l| l + 1).unwrap_or(0);
        offset += cap;
    }
    per_driver_l
}

fn map_to_scales_and_percentages(
    full_beta: &[f64],
    per_driver_caps: &[usize],
    per_driver_l: &[usize],
    nonnegative: bool,
    trunc_eps: f64,
) -> (Vec<(f64, Vec<f64>)>, Vec<f64>) {
    let mut coeffs_trimmed = Vec::new();
    let mut per_driver = Vec::with_capacity(per_driver_caps.len());
    let mut offset = 0;
    for (driver_idx, &cap) in per_driver_caps.iter().enumerate() {
        let len = per_driver_l[driver_idx];
        let block = if len == 0 {
            Vec::new()
        } else {
            full_beta[offset..offset + len].to_vec()
        };
        coeffs_trimmed.extend(block.iter().copied());

        let mut scale = block.iter().sum::<f64>();
        if block.is_empty() || scale.abs() < EPS {
            per_driver.push((0.0, Vec::new()));
        } else {
            let mut percentages: Vec<f64> = block.iter().map(|b| b / scale).collect();
            if nonnegative {
                for w in percentages.iter_mut() {
                    if *w < 0.0 {
                        *w = 0.0;
                    }
                }
            }
            let total_pre = percentages.iter().sum::<f64>();
            if total_pre.abs() < EPS {
                scale = 0.0;
                percentages.iter_mut().for_each(|w| *w = 0.0);
            } else {
                for w in percentages.iter_mut() {
                    *w /= total_pre;
                }
            }
            if trunc_eps > 0.0 {
                for w in percentages.iter_mut() {
                    if w.abs() < trunc_eps {
                        *w = 0.0;
                    }
                }
            }
            let total = percentages.iter().sum::<f64>();
            if total.abs() < EPS {
                // All taps truncated to zero - warn user
                eprintln!(
                    "Warning: All taps for driver {} were truncated to zero \
                     (pct_epsilon={:.4}). This driver will contribute only through \
                     the intercept. Consider using a smaller truncation threshold \
                     (e.g., 0.01-0.05).",
                    driver_idx, trunc_eps
                );
                scale = 0.0;
                percentages.iter_mut().for_each(|w| *w = 0.0);
            } else {
                for w in percentages.iter_mut() {
                    *w /= total;
                }
            }
            per_driver.push((scale, percentages));
        }
        offset += cap;
    }

    (per_driver, coeffs_trimmed)
}

fn prune_mask_with_guardrails(
    mask: &mut [bool],
    beta: &[f64],
    rows: usize,
    guards: &Guardrails,
) -> Result<(), FirError> {
    let mut active_indices: Vec<usize> = mask
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None })
        .collect();

    if active_indices.is_empty() {
        if let Some((idx, _)) = beta
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(Ordering::Equal))
        {
            mask[idx] = true;
            active_indices.push(idx);
        } else {
            return Err(FirError::Guardrail(
                "lasso produced no active coefficients".to_string(),
            ));
        }
    }

    let ratio = if guards.max_params_ratio > 0.0 {
        guards.max_params_ratio
    } else {
        DEFAULT_MAX_PARAMS_RATIO
    };
    let max_total = if guards.max_total_lag > 0 {
        guards.max_total_lag
    } else {
        DEFAULT_MAX_TOTAL_LAG
    };

    let allowed_by_ratio = ((rows as f64) / ratio).floor().max(1.0) as usize;
    let hard_cap = allowed_by_ratio.min(max_total).max(1);

    if active_indices.len() > hard_cap {
        active_indices.sort_by(|&a, &b| {
            beta[b]
                .abs()
                .partial_cmp(&beta[a].abs())
                .unwrap_or(Ordering::Equal)
        });
        let keep: HashSet<usize> = active_indices.into_iter().take(hard_cap).collect();
        for (idx, flag) in mask.iter_mut().enumerate() {
            *flag = keep.contains(&idx);
        }
    }

    Ok(())
}

fn pick_lambda(cv_mean: &[f64], cv_se: &[f64], one_se_rule: bool) -> usize {
    let mut best_idx = 0;
    for (idx, &rmse) in cv_mean.iter().enumerate() {
        if rmse < cv_mean[best_idx] {
            best_idx = idx;
        }
    }
    if !one_se_rule || cv_mean.is_empty() {
        return best_idx;
    }
    let threshold = cv_mean[best_idx] + cv_se[best_idx];
    for (idx, &rmse_value) in cv_mean.iter().enumerate() {
        if rmse_value <= threshold {
            return idx;
        }
    }
    best_idx
}

fn ic_select(
    drivers: &[Vec<f64>],
    target: &[f64],
    caps: &[usize],
    criterion: IcKind,
    search: IcSearchKind,
    opts: &OlsOptions,
    guards: &Guardrails,
) -> Result<Vec<bool>, FirError> {
    match search {
        IcSearchKind::Grid => ic_select_grid(drivers, target, caps, criterion, opts, guards),
        IcSearchKind::GreedyForward => {
            ic_select_greedy(drivers, target, caps, criterion, opts, guards)
        }
    }
}

/// Compute information criterion (BIC or AIC) from residual sum of squares.
///
/// Proper formula:
/// - BIC = n * ln(RSS/n) + k * ln(n)
/// - AIC = n * ln(RSS/n) + 2*k
///
/// where RSS is residual sum of squares, n is sample size, k is number of parameters.
fn ic_value_from_rss(rss: f64, k: usize, n: usize, criterion: IcKind) -> f64 {
    let n_f = n as f64;
    // Log-likelihood term: n * ln(RSS/n)
    let ll_term = n_f * (rss / n_f).max(1e-12).ln();

    // Penalty term
    let penalty = match criterion {
        IcKind::Bic => n_f.ln() * (k as f64),
        IcKind::Aic => 2.0 * (k as f64),
    };

    ll_term + penalty
}

/// Convenience wrapper that takes RMSE instead of RSS.
fn ic_value(rmse: f64, k: usize, n: usize, criterion: IcKind) -> f64 {
    let rss = (rmse * rmse) * (n as f64);
    ic_value_from_rss(rss, k, n, criterion)
}

fn ic_select_grid(
    drivers: &[Vec<f64>],
    target: &[f64],
    caps: &[usize],
    criterion: IcKind,
    opts: &OlsOptions,
    guards: &Guardrails,
) -> Result<Vec<bool>, FirError> {
    let mut best_mask = Vec::new();
    let mut best_ic = f64::INFINITY;

    let mut current = vec![0usize; caps.len()];
    {
        let mut visitor = |lags: &[usize]| -> Result<(), FirError> {
            let total: usize = lags.iter().sum();
            if total == 0 {
                return Ok(());
            }
            let ratio = if guards.max_params_ratio > 0.0 {
                guards.max_params_ratio
            } else {
                DEFAULT_MAX_PARAMS_RATIO
            };
            let allowed = ((target.len() as f64) / ratio).floor().max(1.0) as usize;
            if total > allowed || total > guards.max_total_lag {
                return Ok(());
            }
            let fit = crate::ols::fit_ols(drivers, target, lags, opts)?;
            let k = fit.coeffs.len();
            let ic = ic_value(fit.rmse, k, fit.n_rows, criterion);
            if ic < best_ic {
                best_ic = ic;
                best_mask = lags_to_mask(lags, caps);
            }
            Ok(())
        };
        grid_search(0, caps, &mut current, &mut visitor)?;
    }

    if best_mask.is_empty() {
        Err(FirError::Guardrail(
            "ic search did not find a valid configuration".to_string(),
        ))
    } else {
        Ok(best_mask)
    }
}

fn ic_select_greedy(
    drivers: &[Vec<f64>],
    target: &[f64],
    caps: &[usize],
    criterion: IcKind,
    opts: &OlsOptions,
    guards: &Guardrails,
) -> Result<Vec<bool>, FirError> {
    let mut lags = vec![0usize; caps.len()];
    let mut best_ic = f64::INFINITY;
    let mut improved = true;

    while improved {
        improved = false;
        for driver_idx in 0..caps.len() {
            if lags[driver_idx] >= caps[driver_idx] {
                continue;
            }
            lags[driver_idx] += 1;
            let total: usize = lags.iter().sum();
            if total == 0 {
                lags[driver_idx] -= 1;
                continue;
            }
            let ratio = if guards.max_params_ratio > 0.0 {
                guards.max_params_ratio
            } else {
                DEFAULT_MAX_PARAMS_RATIO
            };
            let allowed = ((target.len() as f64) / ratio).floor().max(1.0) as usize;
            if total > allowed || total > guards.max_total_lag {
                lags[driver_idx] -= 1;
                continue;
            }

            let fit = crate::ols::fit_ols(drivers, target, &lags, opts)?;
            let k = fit.coeffs.len();
            let ic = ic_value(fit.rmse, k, fit.n_rows, criterion);
            if ic + EPS < best_ic {
                best_ic = ic;
                improved = true;
            } else {
                lags[driver_idx] -= 1;
            }
        }
    }

    if lags.iter().all(|&l| l == 0) {
        Err(FirError::Guardrail(
            "greedy IC search failed to add any lags".to_string(),
        ))
    } else {
        Ok(lags_to_mask(&lags, caps))
    }
}

fn grid_search<F>(
    idx: usize,
    caps: &[usize],
    current: &mut [usize],
    f: &mut F,
) -> Result<(), FirError>
where
    F: FnMut(&[usize]) -> Result<(), FirError>,
{
    if idx == caps.len() {
        return f(current);
    }
    for lag in 0..=caps[idx] {
        current[idx] = lag;
        grid_search(idx + 1, caps, current, f)?;
    }
    Ok(())
}

fn lags_to_mask(lags: &[usize], caps: &[usize]) -> Vec<bool> {
    let mut mask = Vec::new();
    for (lag, cap) in lags.iter().zip(caps.iter()) {
        for idx in 0..*cap {
            mask.push(idx < *lag);
        }
    }
    mask
}

fn screen_select(
    x_full: &Array2<f64>,
    y: &Array1<f64>,
    column_map: &[(usize, usize)],
    params: ScreenParams,
    opts: &OlsOptions,
    guards: &Guardrails,
) -> Result<Vec<bool>, FirError> {
    let mut scored: Vec<(usize, f64)> = column_map
        .iter()
        .enumerate()
        .map(|(idx, _)| {
            let col = x_full.column(idx);
            let mut num = 0.0;
            let mut denom_x = 0.0;
            let mut denom_y = 0.0;
            for i in 0..col.len() {
                num += col[i] * y[i];
                denom_x += col[i] * col[i];
                denom_y += y[i] * y[i];
            }
            let corr = if denom_x <= EPS || denom_y <= EPS {
                0.0
            } else {
                num / denom_x.sqrt() / denom_y.sqrt()
            };
            (idx, corr.abs())
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

    let limit = params.top_k.max(1);
    let mut mask = vec![false; column_map.len()];
    for (selected, (idx, score)) in scored.into_iter().enumerate() {
        if selected >= limit {
            break;
        }
        if score < params.min_abs_corr {
            break;
        }
        mask[idx] = true;
    }

    if params.prune_bic && mask.iter().any(|&m| m) {
        prune_by_bic(x_full, y, &mut mask, opts, guards)?;
    }

    if !mask.iter().any(|&m| m) {
        Err(FirError::Guardrail(
            "screening strategy did not retain any features".to_string(),
        ))
    } else {
        Ok(mask)
    }
}

fn prune_by_bic(
    x_full: &Array2<f64>,
    y: &Array1<f64>,
    mask: &mut [bool],
    opts: &OlsOptions,
    guards: &Guardrails,
) -> Result<(), FirError> {
    loop {
        let active_cols: Vec<usize> = mask
            .iter()
            .enumerate()
            .filter_map(|(idx, &m)| if m { Some(idx) } else { None })
            .collect();
        if active_cols.len() <= 1 {
            break;
        }
        let current_bic = bic_for_mask(x_full, y, mask, opts)?;
        let mut best_drop = None;
        let mut best_bic = current_bic;

        for &col in active_cols.iter() {
            mask[col] = false;
            let total_active = mask.iter().filter(|&&m| m).count();
            if total_active == 0 {
                mask[col] = true;
                continue;
            }
            if total_active > guards.max_total_lag {
                mask[col] = true;
                continue;
            }
            let bic = bic_for_mask(x_full, y, mask, opts)?;
            if bic + EPS < best_bic {
                best_bic = bic;
                best_drop = Some(col);
            }
            mask[col] = true;
        }

        if let Some(drop_col) = best_drop {
            mask[drop_col] = false;
        } else {
            break;
        }
    }
    Ok(())
}

fn bic_for_mask(
    x_full: &Array2<f64>,
    y: &Array1<f64>,
    mask: &[bool],
    opts: &OlsOptions,
) -> Result<f64, FirError> {
    let (x_sel, _) = slice_columns(x_full, mask);
    let dataset = Dataset::new(x_sel.clone(), y.clone());
    let lin = LinearRegression::new().with_intercept(opts.intercept);
    let fitted = lin
        .fit(&dataset)
        .map_err(|e| FirError::Linalg(format!("{:?}", e)))?;
    let beta = fitted.params();
    if beta.len() > x_sel.nrows() {
        return Ok(f64::INFINITY);
    }
    let mut preds = x_sel.dot(&Array1::from(beta.to_vec()));
    let intercept = fitted.intercept();
    if opts.intercept {
        for v in preds.iter_mut() {
            *v += intercept;
        }
    }
    let rss = preds
        .iter()
        .zip(y.iter())
        .map(|(p, a)| {
            let diff = a - p;
            diff * diff
        })
        .sum::<f64>();
    let n = y.len();
    let mut k = beta.len();
    if opts.intercept {
        k += 1;
    }
    // Use proper BIC formula: n * ln(RSS/n) + k * ln(n)
    Ok(ic_value_from_rss(rss, k, n, IcKind::Bic))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::LassoSettings;

    fn build_series() -> (Vec<Vec<f64>>, Vec<f64>) {
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
        (vec![d1, d2], target)
    }

    #[test]
    fn test_auto_lag_selection_recovers_signal() {
        let (drivers, target) = build_series();
        let strategy = LagSelect::Lasso {
            caps: Caps {
                per_driver_max: vec![3, 2],
                default_cap: 3,
            },
            lasso: LassoSettings::default(),
        };
        let result = fit_ols_auto_lags(
            &drivers,
            &target,
            &strategy,
            &OlsOptions::default(),
            &Guardrails::default(),
            &Truncation::default(),
        )
        .unwrap();

        assert_eq!(result.per_driver.len(), drivers.len());
        assert!(result.per_driver_l[0] >= 2);
        assert!(result.rmse_fit < 5.0);
        assert!(result.intercept.abs() > 1.0);
    }

    #[test]
    fn test_guardrails_limit_total_features() {
        let (drivers, target) = build_series();
        let strategy = LagSelect::Lasso {
            caps: Caps {
                per_driver_max: vec![4, 3],
                default_cap: 4,
            },
            lasso: LassoSettings::default(),
        };
        let guards = Guardrails {
            max_params_ratio: 10.0,
            max_total_lag: 3,
            seed: 42,
        };
        let result = fit_ols_auto_lags(
            &drivers,
            &target,
            &strategy,
            &OlsOptions::default(),
            &guards,
            &Truncation::default(),
        )
        .unwrap();

        let total_lags: usize = result.per_driver_l.iter().sum();
        assert!(total_lags <= guards.max_total_lag);
    }

    #[test]
    fn test_truncation_eliminates_small_percentages() {
        let (drivers, target) = build_series();
        // Use a reasonable epsilon that won't be clamped (< 0.5)
        let trunc = Truncation { pct_epsilon: 0.05 };
        let strategy = LagSelect::Lasso {
            caps: Caps {
                per_driver_max: vec![3, 2],
                default_cap: 3,
            },
            lasso: LassoSettings::default(),
        };
        let result = fit_ols_auto_lags(
            &drivers,
            &target,
            &strategy,
            &OlsOptions::default(),
            &Guardrails::default(),
            &trunc,
        )
        .unwrap();

        // After validation and truncation, all percentages should be 0 or >= epsilon
        let first_driver = &result.per_driver[0].1;
        for &p in first_driver {
            assert!(p == 0.0 || p >= 0.05 - 1e-6);
        }
    }

    #[test]
    fn test_truncation_validation_clamps_high_values() {
        // Test that validation clamps unreasonably high epsilon values
        let trunc_high = Truncation { pct_epsilon: 0.6 };
        let trunc_validated = trunc_high.validated();
        assert_eq!(trunc_validated.pct_epsilon, 0.05);

        let trunc_negative = Truncation { pct_epsilon: -0.1 };
        let trunc_validated2 = trunc_negative.validated();
        assert_eq!(trunc_validated2.pct_epsilon, 0.0);

        let trunc_ok = Truncation { pct_epsilon: 0.03 };
        let trunc_validated3 = trunc_ok.validated();
        assert_eq!(trunc_validated3.pct_epsilon, 0.03);
    }

    #[test]
    fn test_fit_auto_reports_length_mismatch() {
        let (drivers, mut target) = build_series();
        target.pop();
        let err = fit_auto(&drivers, &target).unwrap_err();
        assert!(matches!(err, FirError::LengthMismatch));
    }

    #[test]
    fn test_ic_strategy_executes() {
        let (drivers, target) = build_series();
        let caps = Caps {
            per_driver_max: vec![3, 3],
            default_cap: 3,
        };
        let strategy = LagSelect::Ic {
            caps,
            criterion: IcKind::Bic,
            search: IcSearchKind::Grid,
        };

        let result = fit_ols_auto_lags(
            &drivers,
            &target,
            &strategy,
            &OlsOptions::default(),
            &Guardrails::default(),
            &Truncation::default(),
        )
        .unwrap();
        assert!(result.per_driver_l.iter().any(|&l| l > 0));
    }

    #[test]
    fn test_screen_strategy_executes() {
        let (drivers, target) = build_series();
        let caps = Caps {
            per_driver_max: vec![4, 4],
            default_cap: 4,
        };
        let strategy = LagSelect::Screen {
            caps,
            top_k: 3,
            min_abs_corr: 0.1,
            prune_bic: true,
        };

        let result = fit_ols_auto_lags(
            &drivers,
            &target,
            &strategy,
            &OlsOptions::default(),
            &Guardrails::default(),
            &Truncation::default(),
        )
        .unwrap();
        assert!(result.per_driver_l.iter().any(|&l| l > 0));
    }
}
