//! # wc_fir
//!
//! A Rust library for Manual & Auto (OLS) FIR modelling of working capital drivers.
//!
//! This crate provides two main modes:
//!
//! * **Manual**: Apply user-supplied FIR profiles (percentages + scale) to drivers
//! * **Auto (OLS)**: Estimate FIR taps via ordinary least squares, then map to {scale, percentages}
//!
//! ## Example
//!
//! ```
//! use wc_fir::{fit_auto, ManualProfile, OlsOptions, fit_ols, manual_apply};
//!
//! // Historical series (same length)
//! let d1: Vec<f64> = (0..24).map(|i| 120.0 + 2.0 * i as f64).collect();
//! let d2: Vec<f64> = (0..24).map(|i| 80.0 + 1.5 * i as f64).collect();
//! let y: Vec<f64> = d1
//!     .iter()
//!     .zip(d2.iter())
//!     .map(|(&a, &b)| 0.6 * a + 0.3 * b + 12.0)
//!     .collect();
//!
//! // Manual mode
//! let manual = manual_apply(
//!     &[d1.clone(), d2.clone()],
//!     &[
//!         ManualProfile { percentages: vec![0.5, 0.35, 0.15], scale: 0.9 },
//!         ManualProfile { percentages: vec![0.3, 0.7], scale: 0.6 },
//!     ],
//! ).unwrap();
//!
//! // Auto (OLS) mode
//! let fit = fit_ols(
//!     &[d1.clone(), d2.clone()],
//!     y.as_slice(),
//!     &[3, 2], // lags per driver
//!     &OlsOptions::default(),
//! ).unwrap();
//!
//! // Auto defaults: build max design, lasso-select, refit OLS
//! let auto = fit_auto(&[d1, d2], y.as_slice()).unwrap();
//! println!("Auto RMSE: {:.4}  R²: {:.4}", auto.rmse, auto.r2);
//!
//! println!("OLS per driver: {:?}", fit.per_driver);
//! println!("RMSE: {:.4}  R²: {:.4}", fit.rmse, fit.r2);
//! ```

// Module declarations
pub mod data;
mod defaults;
pub mod fir;
mod lasso;
pub mod ols;
mod select;
mod types;

// Re-export public types
pub use types::{
    AutoLagResult, Caps, FirError, Guardrails, IcKind, IcSearchKind, Lag, LagSelect, LassoSettings,
    ManualProfile, OlsFit, OlsOptions, Truncation,
};

// Re-export main public functions
pub use fir::manual_apply;
pub use ols::{fit_ols, predict_ols};
pub use select::{fit_auto, fit_auto_prefix, fit_ols_auto_lags};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_end_to_end_manual() {
        let drivers = vec![
            vec![100.0, 110.0, 120.0, 130.0, 140.0],
            vec![50.0, 55.0, 60.0, 65.0, 70.0],
        ];

        let profiles = vec![
            ManualProfile {
                percentages: vec![0.6, 0.4],
                scale: 0.8,
            },
            ManualProfile {
                percentages: vec![0.5, 0.5],
                scale: 0.5,
            },
        ];

        let result = manual_apply(&drivers, &profiles);
        assert!(result.is_ok());
        let balance = result.unwrap();
        assert_eq!(balance.len(), 5);
    }

    #[test]
    fn test_end_to_end_ols() {
        let drivers = vec![
            vec![120.0, 125.0, 130.0, 128.0, 140.0, 150.0, 160.0, 170.0],
            vec![80.0, 82.0, 85.0, 90.0, 95.0, 98.0, 100.0, 105.0],
        ];
        let target = vec![95.0, 100.0, 108.0, 115.0, 120.0, 130.0, 140.0, 155.0];

        let result = fit_ols(&drivers, target.as_slice(), &[3, 2], &OlsOptions::default());
        assert!(result.is_ok());

        let fit = result.unwrap();
        assert_eq!(fit.per_driver.len(), 2);
        assert!(fit.rmse >= 0.0);
        assert!(fit.r2 <= 1.0);
    }

    #[test]
    fn test_fit_auto_basic() {
        let len = 24;
        let mut d1 = Vec::with_capacity(len);
        let mut d2 = Vec::with_capacity(len);
        let mut target = Vec::with_capacity(len);
        for i in 0..len {
            let v1 = 100.0 + i as f64 * 5.0;
            let v2 = 60.0 + i as f64 * 2.0;
            d1.push(v1);
            d2.push(v2);
            target.push(0.65 * v1 + 0.25 * v2 + 20.0);
        }
        let drivers = vec![d1, d2];

        let fit = fit_auto(&drivers, target.as_slice()).unwrap();
        assert!(!fit.coeffs.is_empty());
        assert_eq!(fit.per_driver.len(), drivers.len());
        assert!(fit.rmse < 10.0);
    }

    #[test]
    fn test_ols_with_options() {
        let drivers = vec![
            vec![120.0, 125.0, 130.0, 128.0, 140.0, 150.0, 160.0, 170.0],
            vec![80.0, 82.0, 85.0, 90.0, 95.0, 98.0, 100.0, 105.0],
        ];
        let target = vec![95.0, 100.0, 108.0, 115.0, 120.0, 130.0, 140.0, 155.0];

        let opts = OlsOptions {
            intercept: true,
            ridge_lambda: 0.1,
            nonnegative: true,
        };

        let result = fit_ols(&drivers, target.as_slice(), &[2, 2], &opts);
        assert!(result.is_ok());

        let fit = result.unwrap();
        // Check nonnegativity constraint
        for (_scale, pct) in &fit.per_driver {
            for &p in pct {
                assert!(p >= 0.0, "Percentage should be non-negative");
            }
        }
    }

    #[test]
    fn test_round_trip() {
        // Fit OLS, then apply manual with recovered profiles
        let drivers = vec![
            vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            vec![50.0, 55.0, 60.0, 65.0, 70.0, 75.0],
        ];
        let target = vec![80.0, 90.0, 100.0, 110.0, 120.0, 130.0];

        // Fit OLS
        let fit = fit_ols(&drivers, target.as_slice(), &[2, 2], &OlsOptions::default()).unwrap();

        // Convert to manual profiles
        let profiles: Vec<ManualProfile> = fit
            .per_driver
            .into_iter()
            .map(|(scale, percentages)| ManualProfile { percentages, scale })
            .collect();

        // Apply manual
        let manual_result = manual_apply(&drivers, &profiles).unwrap();

        assert_eq!(manual_result.len(), 6);
    }
}
