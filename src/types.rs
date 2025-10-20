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

/// Output of Auto (OLS) fit, mapped for auditability.
///
/// Contains the raw FIR coefficients, per-driver decomposition into scale and percentages,
/// and fit quality metrics (RMSE, R²).
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
}

/// Options for OLS fit.
///
/// # Example
/// ```
/// use wc_fir::OlsOptions;
/// let opts = OlsOptions {
///     intercept: false,
///     ridge_lambda: 0.0,
///     nonnegative: false,
/// };
/// ```
#[derive(Clone, Debug, Default)]
pub struct OlsOptions {
    /// Add a column of ones to X (intercept). Default: false.
    pub intercept: bool,
    /// Optional ridge (L2) strength; if 0.0, plain least squares.
    pub ridge_lambda: f64,
    /// Enforce nonnegativity post-fit (clip small negatives then renormalize).
    pub nonnegative: bool,
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
}
