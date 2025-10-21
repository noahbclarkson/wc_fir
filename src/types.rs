use crate::defaults::{
    DEFAULT_ALPHA, DEFAULT_FOLDS, DEFAULT_GLOBAL_CAP_AR, DEFAULT_MAX_PARAMS_RATIO,
    DEFAULT_MAX_TOTAL_LAG, DEFAULT_ONE_SE, DEFAULT_SEED, DEFAULT_STANDARDIZE, DEFAULT_TRUNC_EPS,
};

/// Per-driver lag length.
pub type Lag = usize;

/// Manual profile per driver: percentages (tap weights) + scale.
///
/// Invariant: percentages.len() == lag_k; usually sum == 1.0 and entries >= 0.
///
/// # Example
/// ```
/// use wc_fir::ManualProfile;
/// let profile = ManualProfile {
///     percentages: vec![0.5, 0.35, 0.15],
///     scale: 0.9,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct ManualProfile {
    pub percentages: Vec<f64>,
    pub scale: f64,
}

/// Plain OLS fit result used across the crate.
#[derive(Clone, Debug)]
pub struct OlsFit {
    /// Raw FIR coefficients concatenated by driver: [b1_0..b1_L1-1, b2_0.., ...]
    pub coeffs: Vec<f64>,
    /// For each driver k: (scale_k, percentages_k)
    pub per_driver: Vec<(f64, Vec<f64>)>,
    /// Root mean squared error on the fit window
    pub rmse: f64,
    /// R² (coefficient of determination) on the fit window
    pub r2: f64,
    /// Number of rows used after burn-in alignment
    pub n_rows: usize,
    /// Intercept recovered by linear regression
    pub intercept: f64,
}

/// Output of automatic lag selection followed by an OLS refit.
#[derive(Clone, Debug)]
pub struct AutoLagResult {
    pub per_driver_l: Vec<usize>,
    pub coeffs: Vec<f64>,
    pub per_driver: Vec<(f64, Vec<f64>)>,
    pub rmse_fit: f64,
    pub r2_fit: f64,
    pub cv_rmse: Option<f64>,
    pub burn_in: usize,
    pub rows_used: usize,
    pub intercept: f64,
}

/// Output of Auto (OLS) fit, mapped for auditability.
///
/// Contains the raw FIR coefficients, per-driver decomposition into scale and percentages,
/// and fit quality metrics (RMSE, R²).
/// Options for OLS fit.
///
/// # Example
/// ```
/// use wc_fir::OlsOptions;
/// let opts = OlsOptions {
///     intercept: false,
///     ridge_lambda: 0.0,
///     nonnegative: false,
///     constrain_scale_0_1: false,
/// };
/// ```
#[derive(Clone, Debug)]
pub struct OlsOptions {
    /// Add a column of ones to X (intercept). Default: true.
    pub intercept: bool,
    /// Optional ridge (L2) strength; if 0.0, plain least squares.
    pub ridge_lambda: f64,
    /// Enforce nonnegativity post-fit (clip small negatives then renormalize).
    pub nonnegative: bool,
    /// Enforce scale to be between 0 and 1, adjusting intercept for overflow.
    pub constrain_scale_0_1: bool,
}

impl Default for OlsOptions {
    fn default() -> Self {
        Self {
            intercept: true,
            ridge_lambda: 0.0,
            nonnegative: true,
            constrain_scale_0_1: false,
        }
    }
}

/// Truncation of tiny tap weights after mapping to percentages.
#[derive(Clone, Debug)]
pub struct Truncation {
    pub pct_epsilon: f64,
}

impl Default for Truncation {
    fn default() -> Self {
        Self {
            pct_epsilon: DEFAULT_TRUNC_EPS,
        }
    }
}

impl Truncation {
    /// Validates and clamps pct_epsilon to a sensible range.
    ///
    /// Percentages sum to 1.0, so epsilon values >= 0.5 would eliminate
    /// most/all taps. This method clamps to [0.0, 0.5] and warns if needed.
    pub fn validated(self) -> Self {
        let mut eps = self.pct_epsilon;
        if eps < 0.0 {
            eps = 0.0;
        }
        if eps > 0.5 {
            eprintln!(
                "Warning: pct_epsilon={:.2} is very high (>= 0.5); clamping to 0.05 \
                 to avoid zeroing all taps. Use values like 0.01-0.05 for typical use.",
                self.pct_epsilon
            );
            eps = 0.05;
        }
        Self { pct_epsilon: eps }
    }
}

/// Guardrails to keep fits statistically sane.
#[derive(Clone, Debug)]
pub struct Guardrails {
    pub max_params_ratio: f64,
    pub max_total_lag: usize,
    pub seed: u64,
}

impl Default for Guardrails {
    fn default() -> Self {
        Self {
            max_params_ratio: DEFAULT_MAX_PARAMS_RATIO,
            max_total_lag: DEFAULT_MAX_TOTAL_LAG,
            seed: DEFAULT_SEED,
        }
    }
}

/// Per-driver lag caps used when building the maximal design matrix.
#[derive(Clone, Debug)]
pub struct Caps {
    pub per_driver_max: Vec<usize>,
    pub default_cap: usize,
}

impl Default for Caps {
    fn default() -> Self {
        Self {
            per_driver_max: Vec::new(),
            default_cap: DEFAULT_GLOBAL_CAP_AR,
        }
    }
}

/// Lasso / Elastic-net settings used for automatic lag selection.
#[derive(Clone, Debug)]
pub struct LassoSettings {
    pub alpha: f64,
    pub lambda_grid: Option<Vec<f64>>,
    pub folds: usize,
    pub standardize: bool,
    pub one_se_rule: bool,
}

impl Default for LassoSettings {
    fn default() -> Self {
        Self {
            alpha: DEFAULT_ALPHA,
            lambda_grid: None,
            folds: DEFAULT_FOLDS,
            standardize: DEFAULT_STANDARDIZE,
            one_se_rule: DEFAULT_ONE_SE,
        }
    }
}

/// Information-criterion variants.
#[derive(Clone, Debug, Copy)]
pub enum IcKind {
    Bic,
    Aic,
}

/// Search strategies for IC-based selection.
#[derive(Clone, Debug, Copy)]
pub enum IcSearchKind {
    Grid,
    GreedyForward,
}

/// Lag selection strategies supported by the crate.
#[derive(Clone, Debug)]
pub enum LagSelect {
    Lasso {
        caps: Caps,
        lasso: LassoSettings,
    },
    Ic {
        caps: Caps,
        criterion: IcKind,
        search: IcSearchKind,
    },
    Screen {
        caps: Caps,
        top_k: usize,
        min_abs_corr: f64,
        prune_bic: bool,
    },
    /// Prefix CV: choose per-driver lag lengths via rolling CV,
    /// using contiguous prefix blocks [0..L_k-1] per driver.
    ///
    /// Unlike sparse selection (Lasso/Screen), this maintains the same
    /// hypothesis class as manual OLS, just automating the choice of lag length.
    PrefixCv {
        caps: Caps,
        folds: usize,
        shared: bool, // if true, enforce L_1 = L_2 = ... (simpler)
    },
}

impl Default for LagSelect {
    fn default() -> Self {
        Self::Lasso {
            caps: Caps::default(),
            lasso: LassoSettings::default(),
        }
    }
}

/// Library error type.
#[derive(thiserror::Error, Debug)]
pub enum FirError {
    #[error("input lengths mismatch")]
    LengthMismatch,
    #[error("insufficient data for lags (burn-in {burn_in} exceeds series length)")]
    InsufficientData { burn_in: usize },
    #[error("linear algebra failure: {0}")]
    Linalg(String),
    #[error("empty input")]
    EmptyInput,
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("optimization failed to converge: {0}")]
    ConvergenceFailed(String),
    #[error("guardrail triggered: {0}")]
    Guardrail(String),
}
