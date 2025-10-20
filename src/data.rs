use crate::types::{FirError, Lag};

/// Compute burn-in period (maximum lag - 1) and validate data lengths.
///
/// Returns (burn_in, n_rows) where n_rows is the number of valid rows after alignment.
///
/// # Arguments
/// * `drivers` - Slice of driver series
/// * `target` - Optional target series (for OLS mode)
/// * `lags` - Per-driver lag lengths
///
/// # Errors
/// Returns `FirError::LengthMismatch` if driver lengths differ or if target length doesn't match.
/// Returns `FirError::InsufficientData` if any series is too short for the required burn-in.
pub fn validate_and_align(
    drivers: &[Vec<f64>],
    target: Option<&Vec<f64>>,
    lags: &[Lag],
) -> Result<(usize, usize), FirError> {
    if drivers.is_empty() {
        return Err(FirError::EmptyInput);
    }

    if drivers.len() != lags.len() {
        return Err(FirError::LengthMismatch);
    }

    // Check all drivers have the same length
    let len = drivers[0].len();
    for driver in drivers.iter().skip(1) {
        if driver.len() != len {
            return Err(FirError::LengthMismatch);
        }
    }

    // Check target length if provided
    if let Some(t) = target {
        if t.len() != len {
            return Err(FirError::LengthMismatch);
        }
    }

    // Compute burn-in: max(lags) - 1
    let burn_in = lags.iter().copied().max().unwrap_or(1).saturating_sub(1);

    if len <= burn_in {
        return Err(FirError::InsufficientData { burn_in });
    }

    let n_rows = len - burn_in;
    Ok((burn_in, n_rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_empty() {
        let result = validate_and_align(&[], None, &[]);
        assert!(matches!(result, Err(FirError::EmptyInput)));
    }

    #[test]
    fn test_validate_length_mismatch_drivers() {
        let d1 = vec![1.0, 2.0, 3.0];
        let d2 = vec![4.0, 5.0];
        let result = validate_and_align(&[d1, d2], None, &[2, 2]);
        assert!(matches!(result, Err(FirError::LengthMismatch)));
    }

    #[test]
    fn test_validate_length_mismatch_lags() {
        let d1 = vec![1.0, 2.0, 3.0];
        let d2 = vec![4.0, 5.0, 6.0];
        let result = validate_and_align(&[d1, d2], None, &[2]);
        assert!(matches!(result, Err(FirError::LengthMismatch)));
    }

    #[test]
    fn test_validate_insufficient_data() {
        let d1 = vec![1.0, 2.0];
        let d2 = vec![4.0, 5.0];
        let result = validate_and_align(&[d1, d2], None, &[3, 3]);
        assert!(matches!(result, Err(FirError::InsufficientData { .. })));
    }

    #[test]
    fn test_validate_success() {
        let d1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let result = validate_and_align(&[d1, d2], None, &[2, 3]);
        assert!(result.is_ok());
        let (burn_in, n_rows) = result.unwrap();
        assert_eq!(burn_in, 2); // max(2, 3) - 1 = 2
        assert_eq!(n_rows, 3); // 5 - 2 = 3
    }

    #[test]
    fn test_validate_with_target() {
        let d1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let target = vec![11.0, 12.0, 13.0, 14.0, 15.0];
        let result = validate_and_align(&[d1, d2], Some(&target), &[2, 3]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_with_target_mismatch() {
        let d1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d2 = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let target = vec![11.0, 12.0, 13.0];
        let result = validate_and_align(&[d1, d2], Some(&target), &[2, 3]);
        assert!(matches!(result, Err(FirError::LengthMismatch)));
    }
}
