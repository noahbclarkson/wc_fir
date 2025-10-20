use crate::types::{FirError, Lag, ManualProfile};
use ndarray::Array2;

/// Apply a single driver's causal FIR to produce a per-driver contribution.
///
/// Percentages p[0] applies to t, p[1] to t-1, etc. (causal FIR).
///
/// # Arguments
/// * `driver` - Input driver series
/// * `percentages` - Tap weights (p[0] for current period, p[1] for lag-1, etc.)
/// * `scale` - Overall scaling factor for this driver
///
/// # Returns
/// Vector of contributions from this driver at each time point
pub fn apply_driver_fir(driver: &[f64], percentages: &[f64], scale: f64) -> Vec<f64> {
    let t = driver.len();
    let mut y = vec![0.0; t];

    for (i, &w) in percentages.iter().enumerate() {
        if i >= t {
            break;
        }
        let len = t - i;
        for j in 0..len {
            y[i + j] += w * driver[j];
        }
    }

    y.into_iter().map(|v| v * scale).collect()
}

/// Apply manual FIR profiles to multiple drivers to produce synthetic balance.
///
/// Returns the full-length synthetic balance, with causal lags applied.
///
/// # Arguments
/// * `drivers` - Slice of driver series (all must have same length)
/// * `profiles` - Slice of manual profiles (must match drivers.len())
///
/// # Errors
/// Returns `FirError::LengthMismatch` if drivers and profiles lengths don't match,
/// or if drivers have different lengths.
pub fn manual_apply(drivers: &[Vec<f64>], profiles: &[ManualProfile]) -> Result<Vec<f64>, FirError> {
    if drivers.is_empty() {
        return Err(FirError::EmptyInput);
    }

    if drivers.len() != profiles.len() {
        return Err(FirError::LengthMismatch);
    }

    let t = drivers[0].len();

    // Validate all drivers have same length
    for driver in drivers.iter().skip(1) {
        if driver.len() != t {
            return Err(FirError::LengthMismatch);
        }
    }

    let mut out = vec![0.0; t];

    for (driver, profile) in drivers.iter().zip(profiles.iter()) {
        let contrib = apply_driver_fir(driver, &profile.percentages, profile.scale);
        for (i, c) in contrib.into_iter().enumerate() {
            out[i] += c;
        }
    }

    Ok(out)
}

/// Build design matrix X for M drivers with per-driver lags.
///
/// Columns are concatenated blocks: [D1_t, D1_{t-1},..., D2_t, ...].
/// Rows correspond to time indices where all lags are available (after burn-in).
///
/// # Arguments
/// * `drivers` - Slice of driver series
/// * `lags` - Per-driver lag lengths
///
/// # Returns
/// Tuple of (design_matrix, burn_in) where:
/// - design_matrix is an (n_rows x total_lags) matrix
/// - burn_in is the number of initial periods skipped
///
/// # Errors
/// Returns errors if validation fails or data is insufficient for the requested lags.
pub fn build_design(
    drivers: &[Vec<f64>],
    lags: &[Lag],
) -> Result<(Array2<f64>, usize), FirError> {
    // Validate first
    let (burn_in, rows) = crate::data::validate_and_align(drivers, None, lags)?;

    let cols: usize = lags.iter().sum();
    let mut x = Array2::<f64>::zeros((rows, cols));

    let mut col = 0;
    for (driver, &lag) in drivers.iter().zip(lags.iter()) {
        for j in 0..lag {
            // Column = D_{t-j}
            for r in 0..rows {
                let ti = burn_in + r;
                x[[r, col]] = driver[ti - j];
            }
            col += 1;
        }
    }

    Ok((x, burn_in))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_driver_fir_simple() {
        let driver = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let percentages = vec![0.5, 0.3, 0.2];
        let scale = 1.0;

        let result = apply_driver_fir(&driver, &percentages, scale);

        // At t=0: 0.5*1.0 = 0.5
        // At t=1: 0.5*2.0 + 0.3*1.0 = 1.3
        // At t=2: 0.5*3.0 + 0.3*2.0 + 0.2*1.0 = 2.3
        // At t=3: 0.5*4.0 + 0.3*3.0 + 0.2*2.0 = 3.3
        // At t=4: 0.5*5.0 + 0.3*4.0 + 0.2*3.0 = 4.3

        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.5).abs() < 1e-10);
        assert!((result[1] - 1.3).abs() < 1e-10);
        assert!((result[2] - 2.3).abs() < 1e-10);
    }

    #[test]
    fn test_manual_apply_single_driver() {
        let drivers = vec![vec![10.0, 20.0, 30.0, 40.0]];
        let profiles = vec![ManualProfile {
            percentages: vec![0.6, 0.4],
            scale: 0.5,
        }];

        let result = manual_apply(&drivers, &profiles).unwrap();
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_manual_apply_multiple_drivers() {
        let drivers = vec![
            vec![10.0, 20.0, 30.0, 40.0],
            vec![5.0, 10.0, 15.0, 20.0],
        ];
        let profiles = vec![
            ManualProfile {
                percentages: vec![1.0],
                scale: 0.5,
            },
            ManualProfile {
                percentages: vec![1.0],
                scale: 0.3,
            },
        ];

        let result = manual_apply(&drivers, &profiles).unwrap();
        assert_eq!(result.len(), 4);

        // At t=0: 0.5*10.0 + 0.3*5.0 = 6.5
        assert!((result[0] - 6.5).abs() < 1e-10);
    }

    #[test]
    fn test_build_design_simple() {
        let d1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d2 = vec![10.0, 20.0, 30.0, 40.0, 50.0];

        let (x, burn_in) = build_design(&[d1.clone(), d2.clone()], &[2, 2]).unwrap();

        // burn_in = max(2, 2) - 1 = 1
        assert_eq!(burn_in, 1);
        // rows = 5 - 1 = 4
        assert_eq!(x.shape(), &[4, 4]);

        // First row (t=1, index 1):
        // [d1[1], d1[0], d2[1], d2[0]] = [2, 1, 20, 10]
        assert_eq!(x[[0, 0]], 2.0);
        assert_eq!(x[[0, 1]], 1.0);
        assert_eq!(x[[0, 2]], 20.0);
        assert_eq!(x[[0, 3]], 10.0);
    }

    #[test]
    fn test_manual_apply_length_mismatch() {
        let drivers = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let profiles = vec![ManualProfile {
            percentages: vec![1.0],
            scale: 1.0,
        }];

        let result = manual_apply(&drivers, &profiles);
        assert!(matches!(result, Err(FirError::LengthMismatch)));
    }
}
