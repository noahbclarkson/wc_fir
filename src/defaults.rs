//! Default constants for automatic lag selection and fitting.

pub const DEFAULT_GLOBAL_CAP_AR: usize = 9;
pub const DEFAULT_MAX_TOTAL_LAG: usize = 24;
pub const DEFAULT_MAX_PARAMS_RATIO: f64 = 5.0;
pub const DEFAULT_TRUNC_EPS: f64 = 0.01;
pub const DEFAULT_FOLDS: usize = 4;
pub const DEFAULT_ALPHA: f64 = 1.0;
pub const DEFAULT_ONE_SE: bool = true;
pub const DEFAULT_STANDARDIZE: bool = true;
pub const DEFAULT_SEED: u64 = 7;
pub const DEFAULT_LAMBDA_PATH: usize = 50;
pub const DEFAULT_CD_TOL: f64 = 1e-6;
pub const DEFAULT_CD_MAX_ITER: usize = 10_000;
