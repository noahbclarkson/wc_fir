use crate::types::{FirError, Lag, OlsFit, OlsOptions};
use linfa::dataset::Dataset;
use linfa::traits::Fit;
use linfa_linear::LinearRegression;
use ndarray::{Array1, Array2};

/// Augment design matrix and target for ridge regression via Tikhonov method.
///
/// Implements ridge regression as: min ||[X; sqrt(λ)I]β - [y; 0]||²
/// This is equivalent to: min ||Xβ - y||² + λ||β||²
///
/// # Arguments
/// * `x_raw` - Original design matrix (n x p)
/// * `y` - Original response vector (length n)
/// * `lambda` - Ridge penalty parameter
///
/// # Returns
/// Tuple of (augmented_X, augmented_y) for standard OLS solving
fn augment_for_ridge(
    x_raw: &Array2<f64>,
    y: &Array1<f64>,
    lambda: f64,
) -> Result<(Array2<f64>, Array1<f64>), FirError> {
    if lambda <= 0.0 {
        return Ok((x_raw.clone(), y.clone()));
    }

    let (n, p) = x_raw.dim();
    let sqrt_l = lambda.sqrt();

    // Build sqrt(lambda) * I_p (identity matrix scaled by sqrt(lambda))
    let mut reg = Array2::<f64>::zeros((p, p));
    for j in 0..p {
        reg[[j, j]] = sqrt_l;
    }

    // X_aug = [X; sqrt(lambda)*I], y_aug = [y; 0]
    let mut x_aug = Array2::<f64>::zeros((n + p, p));
    x_aug.slice_mut(ndarray::s![0..n, ..]).assign(x_raw);
    x_aug.slice_mut(ndarray::s![n.., ..]).assign(&reg);

    let mut y_aug = Array1::<f64>::zeros(n + p);
    y_aug.slice_mut(ndarray::s![0..n]).assign(y);
    // Tail stays zeros (penalizes large coefficients)

    Ok((x_aug, y_aug))
}

/// Fit FIR taps from drivers & target using ordinary least squares.
///
/// Constructs design matrix from lagged drivers, solves least-squares using Linfa,
/// and maps coefficients back to per-driver {scale, percentages}.
///
/// # Arguments
/// * `drivers` - Slice of driver series (all must have same length)
/// * `target` - Target series to fit
/// * `lags` - Per-driver lag lengths
/// * `opts` - OLS fitting options (intercept, ridge, nonnegativity)
///
/// # Returns
/// `OlsFit` containing raw coefficients, per-driver decomposition, and fit metrics
///
/// # Errors
/// Returns errors if validation fails, data is insufficient, or linear algebra fails
pub fn fit_ols(
    drivers: &[Vec<f64>],
    target: &[f64],
    lags: &[Lag],
    opts: &OlsOptions,
) -> Result<OlsFit, FirError> {
    // Build design matrix and get burn-in
    let (x_raw, burn) = crate::fir::build_design(drivers, lags)?;
    let rows = x_raw.nrows();

    // Align target with design matrix (skip burn-in period)
    let y = Array1::from(target[burn..].to_vec());
    if y.len() != rows {
        return Err(FirError::LengthMismatch);
    }

    // Apply ridge regularization if requested via data augmentation
    let (x_used, y_used) = if opts.ridge_lambda > 0.0 {
        augment_for_ridge(&x_raw, &y, opts.ridge_lambda)?
    } else {
        (x_raw.clone(), y.clone())
    };

    // Create Linfa dataset and fit linear regression
    let dataset = Dataset::new(x_used, y_used);
    let linreg = LinearRegression::new().with_intercept(opts.intercept);
    let fitted = linreg
        .fit(&dataset)
        .map_err(|e| FirError::Linalg(format!("{:?}", e)))?;

    // Extract coefficients (params) - these are the FIR taps
    let beta_vec = fitted.params().to_vec();
    let intercept = if opts.intercept {
        fitted.intercept()
    } else {
        0.0
    };

    // Map β to per-driver blocks: each block represents one driver's taps
    let mut per_driver = Vec::with_capacity(lags.len());
    let mut off = 0;

    for &lag in lags {
        let block = &beta_vec[off..off + lag];
        let mut scale: f64 = block.iter().sum();

        // Calculate percentages (normalized tap weights)
        let mut percentages = if scale.abs() > 1e-12 {
            block.iter().map(|&v| v / scale).collect::<Vec<_>>()
        } else {
            vec![0.0; lag]
        };

        // Apply nonnegativity constraint if requested
        if opts.nonnegative {
            // Clip negative percentages to zero
            for w in percentages.iter_mut() {
                if *w < 0.0 {
                    *w = 0.0;
                }
            }
            // Renormalize after clipping
            let total: f64 = percentages.iter().sum();
            if total > 1e-12 {
                for w in percentages.iter_mut() {
                    *w /= total;
                }
            } else {
                for w in percentages.iter_mut() {
                    *w = 0.0;
                }
                scale = 0.0;
            }
        }

        per_driver.push((scale, percentages));
        off += lag;
    }

    // Compute fit metrics on the fit window
    let y_hat = reconstruct_fit(drivers, &per_driver, lags, burn, rows, intercept);
    let (rmse, r2) = compute_metrics(&y, &y_hat);

    Ok(OlsFit {
        coeffs: beta_vec,
        per_driver,
        rmse,
        r2,
        n_rows: rows,
        intercept,
    })
}

/// Reconstruct fitted values using per-driver contributions.
///
/// Applies the FIR model to the input drivers to produce predictions.
pub(crate) fn reconstruct_fit(
    drivers: &[Vec<f64>],
    per_driver: &[(f64, Vec<f64>)],
    lags: &[Lag],
    burn: usize,
    rows: usize,
    intercept: f64,
) -> Array1<f64> {
    let mut y_hat = Array1::<f64>::zeros(rows);

    for (k, &_lag) in lags.iter().enumerate() {
        let (scale_k, ref pct_k) = per_driver[k];
        for (j, &w) in pct_k.iter().enumerate() {
            for r in 0..rows {
                let ti = burn + r;
                y_hat[r] += scale_k * w * drivers[k][ti - j];
            }
        }
    }

    if intercept.abs() > 0.0 {
        for v in y_hat.iter_mut() {
            *v += intercept;
        }
    }

    y_hat
}

/// Compute RMSE and R² metrics.
///
/// # Arguments
/// * `y_actual` - Actual target values
/// * `y_pred` - Predicted values from the model
///
/// # Returns
/// Tuple of (RMSE, R²) fit quality metrics
pub(crate) fn compute_metrics(y_actual: &Array1<f64>, y_pred: &Array1<f64>) -> (f64, f64) {
    let n = y_actual.len() as f64;

    // RMSE
    let mse = y_actual
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        / n;
    let rmse = mse.sqrt();

    // R²
    let y_mean = y_actual.mean().unwrap_or(0.0);
    let ss_tot: f64 = y_actual.iter().map(|&v| (v - y_mean).powi(2)).sum();
    let ss_res: f64 = y_actual
        .iter()
        .zip(y_pred.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    let r2 = 1.0 - ss_res / ss_tot.max(1e-12);

    (rmse, r2)
}

/// Generate predictions from a fitted OLS model.
///
/// This is used for out-of-sample prediction in rolling CV.
/// Applies the FIR convolution using the fitted per-driver scales and percentages.
///
/// # Arguments
/// * `drivers` - Driver time series for prediction period
/// * `fit` - Fitted OLS result containing per_driver taps and intercept
/// * `lags` - Per-driver lag lengths (must match what was used in fitting)
///
/// # Returns
/// Vector of predictions for the given drivers
pub fn predict_ols(
    drivers: &[Vec<f64>],
    fit: &OlsFit,
    lags: &[Lag],
) -> Result<Vec<f64>, FirError> {
    if drivers.len() != fit.per_driver.len() {
        return Err(FirError::LengthMismatch);
    }
    if drivers.len() != lags.len() {
        return Err(FirError::LengthMismatch);
    }
    if drivers.is_empty() {
        return Err(FirError::EmptyInput);
    }

    let t = drivers[0].len();
    for d in drivers.iter() {
        if d.len() != t {
            return Err(FirError::LengthMismatch);
        }
    }

    // Apply FIR for each driver, then sum + intercept
    let mut predictions = vec![fit.intercept; t];

    for (driver_idx, driver) in drivers.iter().enumerate() {
        let (scale, percentages) = &fit.per_driver[driver_idx];
        if percentages.is_empty() {
            continue; // Driver not used
        }

        // Apply FIR convolution
        for (lag_idx, &pct) in percentages.iter().enumerate() {
            if pct.abs() < 1e-12 {
                continue;
            }
            for t_idx in lag_idx..t {
                predictions[t_idx] += scale * pct * driver[t_idx - lag_idx];
            }
        }
    }

    Ok(predictions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ols_simple() {
        // Simple test: y = 2*x
        let drivers = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
        let target = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let lags = vec![1];
        let opts = OlsOptions::default();

        let result = fit_ols(&drivers, target.as_slice(), &lags, &opts).unwrap();

        // Should recover scale ≈ 2.0, percentages ≈ [1.0]
        assert_eq!(result.per_driver.len(), 1);
        let (scale, pct) = &result.per_driver[0];
        assert!((scale - 2.0).abs() < 0.1);
        assert!((pct[0] - 1.0).abs() < 0.1);
        assert!(result.intercept.abs() < 1e-6);
    }

    #[test]
    fn test_ols_with_intercept() {
        let drivers = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let target = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
        let lags = vec![1];
        let opts = OlsOptions {
            intercept: true,
            ..Default::default()
        };

        let result = fit_ols(&drivers, target.as_slice(), &lags, &opts).unwrap();
        assert!(result.rmse < 5.0);
        assert!(result.r2 > 0.8);
        assert!((result.intercept - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_ols_multiple_drivers() {
        let drivers = vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![5.0, 12.0, 14.0, 22.0, 28.0, 35.0],
        ];
        let target = vec![15.0, 32.0, 44.0, 62.0, 78.0, 95.0];
        let lags = vec![1, 1];
        let opts = OlsOptions::default();

        let result = fit_ols(&drivers, target.as_slice(), &lags, &opts).unwrap();
        assert_eq!(result.per_driver.len(), 2);
    }

    #[test]
    fn test_ridge_augmentation() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let lambda = 0.5;

        let (x_aug, y_aug) = augment_for_ridge(&x, &y, lambda).unwrap();

        // Augmented X should have shape (3 + 2, 2) = (5, 2)
        assert_eq!(x_aug.shape(), &[5, 2]);
        // Augmented y should have length 5
        assert_eq!(y_aug.len(), 5);

        // First 3 rows of X_aug should match original X
        assert_eq!(x_aug[[0, 0]], 1.0);
        assert_eq!(x_aug[[0, 1]], 2.0);

        // Last 2 rows should be sqrt(lambda) * I
        let sqrt_lambda = lambda.sqrt();
        assert!((x_aug[[3, 0]] - sqrt_lambda).abs() < 1e-10);
        assert!((x_aug[[4, 1]] - sqrt_lambda).abs() < 1e-10);
        assert!((x_aug[[3, 1]]).abs() < 1e-10);
        assert!((x_aug[[4, 0]]).abs() < 1e-10);
    }

    #[test]
    fn test_ols_with_ridge() {
        let drivers = vec![
            vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            vec![5.0, 10.0, 15.0, 20.0, 25.0, 30.0],
        ];
        let target = vec![15.0, 30.0, 45.0, 60.0, 75.0, 90.0];
        let lags = vec![1, 1];
        let opts = OlsOptions {
            ridge_lambda: 0.1,
            ..Default::default()
        };

        let result = fit_ols(&drivers, target.as_slice(), &lags, &opts);
        assert!(result.is_ok());
        let fit = result.unwrap();
        assert_eq!(fit.per_driver.len(), 2);
    }

    #[test]
    fn test_nonnegativity_constraint() {
        let drivers = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]];
        let target = vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
        let lags = vec![1];
        let opts = OlsOptions {
            nonnegative: true,
            ..Default::default()
        };

        let result = fit_ols(&drivers, target.as_slice(), &lags, &opts).unwrap();

        // Check all percentages are non-negative
        for (_scale, pct) in &result.per_driver {
            for &p in pct {
                assert!(p >= 0.0, "Percentage should be non-negative");
            }
        }
    }
}
