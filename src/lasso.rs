use crate::defaults::{DEFAULT_CD_MAX_ITER, DEFAULT_CD_TOL, DEFAULT_LAMBDA_PATH};
use crate::types::{FirError, Guardrails, LassoSettings};
use ndarray::{s, Array1, Array2, Axis};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Clone, Debug)]
pub struct LassoOutcome {
    pub beta: Vec<f64>,
    pub active_mask: Vec<bool>,
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

fn standardize(
    x: &Array2<f64>,
    y: &Array1<f64>,
    enable: bool,
) -> (Array2<f64>, Array1<f64>, Vec<f64>) {
    let (n, p) = x.dim();
    let mut x_std = x.clone();
    let mut y_std = y.clone();
    let mut scales = vec![1.0; p];

    if enable {
        let mean = y.mean().unwrap_or(0.0);
        y_std -= mean;
    }

    if enable {
        for (j, mut col) in x_std.axis_iter_mut(Axis(1)).enumerate() {
            let mean = col.mean().unwrap_or(0.0);
            col -= mean;

            let mut var = 0.0;
            for &v in col.iter() {
                var += v * v;
            }
            let std = (var / n.max(1) as f64).sqrt();
            if std > 1e-12 {
                col /= std;
                scales[j] = std;
            } else {
                scales[j] = 1.0;
                for v in col.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }

    (x_std, y_std, scales)
}

fn build_lambda_grid(
    x: &Array2<f64>,
    y: &Array1<f64>,
    settings: &LassoSettings,
) -> Result<Vec<f64>, FirError> {
    if let Some(grid) = settings.lambda_grid.clone() {
        if grid.is_empty() {
            return Err(FirError::InvalidConfig(
                "lambda_grid must contain at least one value".to_string(),
            ));
        }
        return Ok(grid);
    }

    let (n, p) = x.dim();
    if n == 0 || p == 0 {
        return Err(FirError::InvalidConfig(
            "design matrix cannot be empty for lambda grid computation".to_string(),
        ));
    }

    // lambda_max = max_j |x_j^T y| / n
    let mut lambda_max: f64 = 0.0;
    for j in 0..p {
        let col = x.column(j);
        let mut dot = 0.0;
        for i in 0..n {
            dot += col[i] * y[i];
        }
        lambda_max = lambda_max.max(dot.abs() / n as f64);
    }
    if lambda_max <= 1e-12 {
        lambda_max = 1.0;
    }

    let lambda_min = lambda_max * 1e-3;
    let k = DEFAULT_LAMBDA_PATH.max(2);
    let log_max = lambda_max.ln();
    let log_min = lambda_min.ln();
    let step = (log_min - log_max) / (k as f64 - 1.0);

    let mut grid = Vec::with_capacity(k);
    for i in 0..k {
        let val = (log_max + step * i as f64).exp();
        grid.push(val);
    }
    Ok(grid)
}

fn coordinate_descent(
    x_std: &Array2<f64>,
    y_std: &Array1<f64>,
    alpha: f64,
    lambda: f64,
    beta_init: &[f64],
    rng: &mut StdRng,
) -> Result<Vec<f64>, FirError> {
    let (n, p) = x_std.dim();
    if n == 0 || p == 0 {
        return Ok(vec![0.0; p]);
    }

    let mut beta = beta_init.to_vec();
    if beta.len() != p {
        beta.resize(p, 0.0);
    }

    // r = y - X beta
    let mut residual = y_std.clone();
    if beta.iter().any(|b| b.abs() > 0.0) {
        for i in 0..n {
            let mut dot = 0.0;
            for j in 0..p {
                dot += x_std[[i, j]] * beta[j];
            }
            residual[i] -= dot;
        }
    }

    let n_f = n as f64;
    let lambda_alpha = lambda * alpha;
    let two_norm_cache: Vec<f64> = (0..p)
        .map(|j| {
            let col = x_std.column(j);
            col.iter().map(|&v| v * v).sum::<f64>() / n_f
        })
        .collect();

    let mut coords: Vec<usize> = (0..p).collect();

    for _iter in 0..DEFAULT_CD_MAX_ITER {
        coords.shuffle(rng);
        let mut max_delta: f64 = 0.0;

        for &j in coords.iter() {
            if two_norm_cache[j] <= 1e-12 {
                beta[j] = 0.0;
                continue;
            }
            let col = x_std.column(j);
            let old = beta[j];

            let mut rho = 0.0;
            for i in 0..n {
                rho += col[i] * (residual[i] + col[i] * old);
            }
            rho /= n_f;
            let denom = two_norm_cache[j] + lambda * (1.0 - alpha);
            if denom <= 1e-12 {
                continue;
            }

            let updated = soft_threshold(rho, lambda_alpha) / denom;
            let diff = updated - old;
            if diff.abs() > 0.0 {
                for i in 0..n {
                    residual[i] -= diff * col[i];
                }
                beta[j] = updated;
                max_delta = max_delta.max(diff.abs());
            }
        }

        if max_delta < DEFAULT_CD_TOL {
            return Ok(beta);
        }
    }

    Err(FirError::ConvergenceFailed(format!(
        "coordinate descent exceeded {} iterations without convergence",
        DEFAULT_CD_MAX_ITER
    )))
}

fn unstandardize_beta(beta_std: &[f64], scales: &[f64]) -> Vec<f64> {
    beta_std
        .iter()
        .zip(scales.iter())
        .map(|(&b, &s)| if s > 0.0 { b / s } else { 0.0 })
        .collect()
}

pub fn lasso_path(
    x: &Array2<f64>,
    y: &Array1<f64>,
    settings: &LassoSettings,
    guards: &Guardrails,
    lambda_override: Option<&[f64]>,
) -> Result<(Vec<f64>, Vec<LassoOutcome>), FirError> {
    if x.nrows() != y.len() {
        return Err(FirError::LengthMismatch);
    }
    if settings.alpha <= 0.0 || settings.alpha > 1.0 {
        return Err(FirError::InvalidConfig(
            "alpha must be in (0, 1] (1.0 = lasso)".to_string(),
        ));
    }

    let (x_std, y_std, scales) = standardize(x, y, settings.standardize);

    let lambda_grid = if let Some(override_grid) = lambda_override {
        override_grid.to_vec()
    } else {
        build_lambda_grid(&x_std, &y_std, settings)?
    };

    let mut outcomes = Vec::with_capacity(lambda_grid.len());
    let mut beta_prev = vec![0.0; x.ncols()];
    let mut rng = StdRng::seed_from_u64(guards.seed);

    for &lambda in lambda_grid.iter() {
        let beta_std =
            coordinate_descent(&x_std, &y_std, settings.alpha, lambda, &beta_prev, &mut rng)?;
        // Warm-start next lambda with current solution
        beta_prev = beta_std.clone();
        let beta_raw = if settings.standardize {
            unstandardize_beta(&beta_std, &scales)
        } else {
            beta_std.clone()
        };
        let active_mask = beta_raw.iter().map(|&b| b.abs() > 1e-8).collect::<Vec<_>>();
        outcomes.push(LassoOutcome {
            beta: beta_raw,
            active_mask,
        });
    }

    Ok((lambda_grid, outcomes))
}

pub fn ts_cv_rmse_for_lambdas(
    x: &Array2<f64>,
    y: &Array1<f64>,
    lambdas: &[f64],
    settings: &LassoSettings,
    guards: &Guardrails,
    folds: usize,
) -> Result<(Vec<f64>, Vec<f64>), FirError> {
    if folds < 2 {
        return Err(FirError::InvalidConfig(
            "time-series CV requires at least 2 folds".to_string(),
        ));
    }
    let n = x.nrows();
    if n != y.len() {
        return Err(FirError::LengthMismatch);
    }
    if n <= folds {
        return Err(FirError::InvalidConfig(
            "not enough rows for requested CV folds".to_string(),
        ));
    }

    let mut rmse_sum = vec![0.0; lambdas.len()];
    let mut rmse_sq_sum = vec![0.0; lambdas.len()];
    let mut counts = vec![0usize; lambdas.len()];
    let fold_span = (n as f64 / (folds as f64 + 1.0)).floor() as usize;
    let min_span = 2.max(fold_span);

    for fold in 0..folds {
        let train_end = (min_span * (fold + 1)).min(n - 1);
        let valid_end = ((fold + 2) * min_span).min(n);
        if train_end == 0 || valid_end <= train_end {
            continue;
        }

        let x_train = x.slice(s![0..train_end, ..]).to_owned();
        let y_train = y.slice(s![0..train_end]).to_owned();
        let x_valid = x.slice(s![train_end..valid_end, ..]).to_owned();
        let y_valid = y.slice(s![train_end..valid_end]).to_owned();

        let override_grid = lambdas;
        let (grid, path) = lasso_path(&x_train, &y_train, settings, guards, Some(override_grid))?;

        for (idx, outcome) in path.iter().enumerate() {
            let beta = &outcome.beta;
            let preds = x_valid
                .rows()
                .into_iter()
                .map(|row| {
                    row.iter()
                        .zip(beta.iter())
                        .map(|(&x_ij, &b)| x_ij * b)
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();

            let mut se = 0.0;
            for (pred, actual) in preds.iter().zip(y_valid.iter()) {
                let diff = actual - pred;
                se += diff * diff;
            }
            let rmse = (se / preds.len().max(1) as f64).sqrt();
            if idx < rmse_sum.len() {
                rmse_sum[idx] += rmse;
                rmse_sq_sum[idx] += rmse * rmse;
                counts[idx] += 1;
            }
        }

        // In case override grid lost precision
        debug_assert_eq!(grid.len(), lambdas.len());
    }

    let means: Vec<f64> = rmse_sum
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &c)| if c > 0 { sum / c as f64 } else { f64::INFINITY })
        .collect();
    let ses: Vec<f64> = rmse_sq_sum
        .iter()
        .zip(counts.iter())
        .zip(means.iter())
        .map(|((&sq_sum, &c), &mean)| {
            if c > 1 {
                let variance = (sq_sum / c as f64) - mean * mean;
                (variance.max(0.0).sqrt()) / (c as f64).sqrt()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    Ok((means, ses))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Guardrails, LassoSettings};
    use ndarray::{array, Array2};

    #[test]
    fn test_lasso_path_shapes_and_support() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 0.8, 0.2, 1.2, 0.4, 1.6, 0.6, 2.0, 0.8, 2.4, 1.0],
        )
        .unwrap();
        let y = array![1.0, 1.1, 1.4, 1.8, 2.1, 2.5];

        let settings = LassoSettings::default();
        let guards = Guardrails::default();
        let (grid, path) = lasso_path(&x, &y, &settings, &guards, None).unwrap();

        assert_eq!(grid.len(), path.len());
        assert_eq!(path[0].beta.len(), 2);

        let mut prev = 0;
        for outcome in path.iter() {
            let active = outcome.active_mask.iter().filter(|&&m| m).count();
            assert!(active >= prev);
            prev = active;
        }
    }

    #[test]
    fn test_ts_cv_rmse_deterministic() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 0.0, 0.5, 1.2, 0.2, 0.6, 1.4, 0.4, 0.7, 1.6, 0.6, 0.8, 1.8, 0.8, 0.9, 2.0,
                1.0, 1.0, 2.2, 1.2, 1.1, 2.4, 1.4, 1.2,
            ],
        )
        .unwrap();
        let y = array![1.0, 1.1, 1.3, 1.6, 1.8, 2.0, 2.2, 2.4];

        let settings = LassoSettings::default();
        let guards = Guardrails::default();
        let (grid, _) = lasso_path(&x, &y, &settings, &guards, None).unwrap();
        let (means_a, _) =
            ts_cv_rmse_for_lambdas(&x, &y, &grid, &settings, &guards, settings.folds).unwrap();
        let (means_b, _) =
            ts_cv_rmse_for_lambdas(&x, &y, &grid, &settings, &guards, settings.folds).unwrap();

        for (a, b) in means_a.iter().zip(means_b.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }
}
