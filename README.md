# wc_fir

> A pure-Rust library for modeling working capital drivers using Finite Impulse Response (FIR) filters

[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](LICENSE)

## Overview

`wc_fir` is a specialized library for financial modeling that applies **Finite Impulse Response (FIR)** filters to working capital drivers. It supports two complementary approaches:

- **Manual Mode**: Apply user-defined FIR profiles with explicit tap weights and scaling factors
- **Auto Mode (OLS)**: Automatically estimate FIR taps from historical data using Ordinary Least Squares regression

The library is designed for analysts and financial engineers who need to model time-lagged relationships between drivers (e.g., revenue, production volume) and working capital components (e.g., accounts receivable, inventory).

### Key Features

- 🦀 **Pure Rust**: No external dependencies (BLAS, LAPACK, MKL) required
- 🌍 **Cross-platform**: Works seamlessly on Windows, Linux, and macOS
- 📊 **Dual modes**: Manual profiles for known relationships, OLS for data-driven discovery
- 🎯 **Flexible**: Supports multiple drivers, varying lag lengths, and intercept terms
- 🔧 **Regularization**: Built-in ridge regression for handling collinearity
- ✅ **Constraints**: Optional non-negativity enforcement for interpretable coefficients
- 📈 **Metrics**: Automatic RMSE and R² calculation for model quality assessment

## Installation

Add `wc_fir` to your `Cargo.toml`:

```toml
[dependencies]
wc_fir = "0.1.0"
```

Or use `cargo add`:

```bash
cargo add wc_fir
```

### Requirements

- Rust 1.70 or newer
- No system dependencies required

## Quick Start

### Example 1: Manual Mode

When you know the relationship between drivers and working capital (e.g., from industry standards or prior analysis):

```rust
use wc_fir::{manual_apply, ManualProfile};

fn main() {
    // Historical driver series
    let revenue = vec![100.0, 110.0, 120.0, 130.0, 140.0, 150.0];
    let production = vec![80.0, 85.0, 90.0, 95.0, 100.0, 105.0];

    // Define FIR profiles:
    // - Revenue impact: 60% immediate, 30% lag-1, 10% lag-2, scaled by 0.9
    // - Production impact: 50% immediate, 50% lag-1, scaled by 0.4
    let profiles = vec![
        ManualProfile {
            percentages: vec![0.6, 0.3, 0.1],
            scale: 0.9,
        },
        ManualProfile {
            percentages: vec![0.5, 0.5],
            scale: 0.4,
        },
    ];

    // Generate synthetic working capital balance
    let balance = manual_apply(&[revenue, production], &profiles).unwrap();

    println!("Working capital balance: {:?}", balance);
}
```

### Example 2: Auto Mode (OLS)

When you want to discover the relationships from historical data:

```rust
use wc_fir::{fit_ols, OlsOptions};

fn main() {
    // Historical drivers
    let revenue = vec![120.0, 125.0, 130.0, 128.0, 140.0, 150.0, 160.0, 170.0];
    let production = vec![80.0, 82.0, 85.0, 90.0, 95.0, 98.0, 100.0, 105.0];

    // Historical working capital (target to fit)
    let actual_wc = vec![95.0, 100.0, 108.0, 115.0, 120.0, 130.0, 140.0, 155.0];

    // Fit OLS model with 3 lags for revenue, 2 lags for production
    let fit = fit_ols(
        &[revenue, production],
        &actual_wc,
        &[3, 2],  // Lag lengths per driver
        &OlsOptions::default(),
    ).unwrap();

    // Inspect results
    println!("Model quality:");
    println!("  RMSE: {:.2}", fit.rmse);
    println!("  R²:   {:.4}", fit.r2);

    println!("\nPer-driver impact:");
    for (i, (scale, percentages)) in fit.per_driver.iter().enumerate() {
        println!("  Driver {}: scale = {:.3}, taps = {:?}", i, scale, percentages);
    }
}
```

## Detailed Usage

### Understanding FIR Profiles

A FIR profile defines how a driver influences working capital over time:

$$\text{Balance}(t) = \sum_{k} \text{scale}_k \times \sum_{j} \text{percentage}_{k,j} \times \text{Driver}_k(t - j)$$

- **Scale**: Overall magnitude of the driver's impact
- **Percentages**: Distribution of impact across time lags (typically sum to 1.0)
  - `percentages[0]`: Immediate impact (lag 0)
  - `percentages[1]`: One-period lag
  - `percentages[2]`: Two-period lag, etc.

#### Example Interpretation

```rust
ManualProfile {
    percentages: vec![0.5, 0.35, 0.15],
    scale: 0.8,
}
```

This means:

- Total impact = 80% of driver value (scale = 0.8)
- 50% appears immediately (lag 0)
- 35% appears one period later (lag 1)
- 15% appears two periods later (lag 2)

### OLS Fitting Options

The `OlsOptions` struct provides fine-grained control over the fitting process:

```rust
use wc_fir::OlsOptions;

let opts = OlsOptions {
    // Add intercept term (baseline offset independent of drivers)
    intercept: true,

    // Ridge regularization strength (0.0 = none, higher = more shrinkage)
    ridge_lambda: 0.1,

    // Force all tap percentages to be non-negative
    nonnegative: true,
};
```

#### When to Use Each Option

| Option | Use When | Example Scenario |
|--------|----------|------------------|
| `intercept: true` | There's a baseline level independent of drivers | Fixed overhead costs |
| `ridge_lambda > 0` | Drivers are correlated or data is limited | Revenue and units sold move together |
| `nonnegative: true` | Only positive relationships make sense | Inventory can't decrease with sales |

### Working with Multiple Drivers

```rust
use wc_fir::{fit_ols, OlsOptions, manual_apply, ManualProfile};

// Scenario: Model accounts receivable based on:
// - Sales revenue (3-month impact window)
// - Credit sales ratio (2-month impact window)
// - Customer count (2-month impact window)

let sales_revenue = vec![100.0, 105.0, 110.0, 108.0, 115.0, 120.0, 125.0, 130.0];
let credit_ratio = vec![0.65, 0.68, 0.70, 0.72, 0.70, 0.68, 0.71, 0.73];
let customers = vec![450.0, 460.0, 470.0, 465.0, 480.0, 490.0, 500.0, 510.0];

let actual_ar = vec![65.0, 71.0, 77.0, 78.0, 81.0, 82.0, 89.0, 95.0];

// Fit model
let fit = fit_ols(
    &[sales_revenue, credit_ratio, customers],
    &actual_ar,
    &[3, 2, 2],  // Lag structure
    &OlsOptions {
        intercept: true,
        ridge_lambda: 0.05,  // Light regularization
        nonnegative: true,   // AR increases with these drivers
    },
).unwrap();

// Convert to manual profiles for forecasting
let profiles: Vec<ManualProfile> = fit.per_driver
    .into_iter()
    .map(|(scale, percentages)| ManualProfile { percentages, scale })
    .collect();

// Now use profiles for forecasting new periods
```

### Handling Insufficient Data

The library automatically handles burn-in periods based on the maximum lag:

```rust
let drivers = vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]];
let target = vec![10.0, 20.0, 30.0, 40.0, 50.0];

// With lag=3, first 2 periods (lag-1) are burn-in
// Effective fit window: periods 3-5 (indices 2-4)
let fit = fit_ols(&drivers, &target, &[3], &OlsOptions::default());

match fit {
    Ok(result) => {
        println!("Fitted {} rows after burn-in", result.n_rows);
    }
    Err(e) => {
        eprintln!("Error: {}", e);
    }
}
```

## How It Works

### Manual Mode: FIR Application

Manual mode applies a **causal FIR filter** to each driver:

1. For each time period `t`:
   - Sum contributions from current period: `scale * percentage[0] * driver[t]`
   - Add lagged contributions: `scale * percentage[j] * driver[t-j]` for j > 0
2. Aggregate across all drivers to produce final balance

This is a **forward-looking convolution** where lags reference historical driver values.

### Auto Mode: OLS Estimation

Auto mode constructs a regression problem and solves it using Linfa (pure Rust ML):

#### Step 1: Design Matrix Construction

Build matrix `X` where each row represents a time period (after burn-in), and columns are:

```text
[D1(t), D1(t-1), ..., D1(t-L1+1), D2(t), D2(t-1), ..., D2(t-L2+1), ...]
```

Example with 2 drivers, lags [2, 2]:

```text
       D1(t)  D1(t-1)  D2(t)  D2(t-1)
t=1    100    95       50     48
t=2    105    100      52     50
t=3    110    105      55     52
...
```

#### Step 2: OLS Regression

Solve: $\min \|X\beta - y\|^2$ where $y$ is the target working capital series.

- If `intercept=true`: Linfa adds column of ones automatically
- If `ridge_lambda > 0`: Apply **data augmentation** (Tikhonov trick):

  $$
  X_{\text{aug}} = \begin{bmatrix} X \\\\ \sqrt{\lambda}I \end{bmatrix}, \quad y_{\text{aug}} = \begin{bmatrix} y \\\\ 0 \end{bmatrix}
  $$

  This transforms ridge regression into OLS: $\min \|X_{\text{aug}}\beta - y_{\text{aug}}\|^2$

#### Step 3: Coefficient Mapping

Map raw coefficients `β` back to per-driver profiles:

1. Partition `β` into driver blocks: `[β1_0, ..., β1_L1-1, β2_0, ..., β2_L2-1, ...]`
2. For each driver block:
   - **Scale** = sum of coefficients in block
   - **Percentages** = each coefficient / scale (normalized tap weights)
3. If `nonnegative=true`: Clip negatives to zero and renormalize

#### Step 4: Quality Metrics

- **RMSE**: $\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$
- **R²**: $R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}}$ where:
  - $SS_{\text{res}} = \sum_{i}(y_i - \hat{y}_i)^2$ (sum of squared residuals)
  - $SS_{\text{tot}} = \sum_{i}(y_i - \bar{y})^2$ (total sum of squares)

### Ridge Regression Implementation

Instead of using matrix inversion (which requires LAPACK), we use **data augmentation**:

$$
\begin{align*}
\text{Original problem:} \quad & \min \|X\beta - y\|^2 + \lambda\|\beta\|^2 \\
\text{Augmented problem:} \quad & \min \left\|\begin{bmatrix} X \\ \sqrt{\lambda}I \end{bmatrix}\beta - \begin{bmatrix} y \\ 0 \end{bmatrix}\right\|^2
\end{align*}
$$

This allows us to use standard OLS solvers (Linfa's pure Rust QR decomposition) without external dependencies.

## API Reference

### Core Functions

#### `manual_apply`

```rust
pub fn manual_apply(
    drivers: &[Vec<f64>],
    profiles: &[ManualProfile],
) -> Result<Vec<f64>, FirError>
```

Apply manual FIR profiles to drivers.

**Arguments:**

- `drivers`: Slice of driver time series (all must have same length)
- `profiles`: Slice of FIR profiles (must match number of drivers)

**Returns:** Synthetic balance time series (same length as inputs)

**Errors:**

- `FirError::LengthMismatch` if drivers/profiles counts differ or drivers have different lengths
- `FirError::EmptyInput` if no drivers provided

---

#### `fit_ols`

```rust
pub fn fit_ols(
    drivers: &[Vec<f64>],
    target: &[f64],
    lags: &[Lag],
    opts: &OlsOptions,
) -> Result<OlsFit, FirError>
```

Estimate FIR taps from historical data using OLS.

**Arguments:**

- `drivers`: Slice of driver time series
- `target`: Target working capital series to fit
- `lags`: Per-driver lag lengths (e.g., `&[3, 2]` for 3 lags on driver 1, 2 on driver 2)
- `opts`: Fitting options (intercept, ridge, non-negativity)

**Returns:** `OlsFit` struct containing:

- `coeffs`: Raw FIR coefficients (concatenated)
- `per_driver`: Vec of `(scale, percentages)` tuples
- `rmse`: Root mean squared error on fit window
- `r2`: R-squared coefficient
- `n_rows`: Number of periods used after burn-in

**Errors:**

- `FirError::LengthMismatch` if inputs have mismatched lengths
- `FirError::InsufficientData` if series too short for requested lags
- `FirError::Linalg` if regression fails (e.g., singular matrix)

### Data Types

#### `ManualProfile`

```rust
pub struct ManualProfile {
    pub percentages: Vec<f64>,  // Tap weights (typically sum to 1.0)
    pub scale: f64,              // Overall scaling factor
}
```

#### `OlsOptions`

```rust
pub struct OlsOptions {
    pub intercept: bool,      // Add intercept term (default: false)
    pub ridge_lambda: f64,    // Ridge penalty (default: 0.0)
    pub nonnegative: bool,    // Enforce non-negative taps (default: false)
}
```

#### `OlsFit`

```rust
pub struct OlsFit {
    pub coeffs: Vec<f64>,                    // Raw coefficients
    pub per_driver: Vec<(f64, Vec<f64>)>,   // (scale, percentages) per driver
    pub rmse: f64,                           // Root mean squared error
    pub r2: f64,                             // R-squared
    pub n_rows: usize,                       // Rows used in fit
}
```

#### `FirError`

```rust
pub enum FirError {
    LengthMismatch,
    InsufficientData { burn_in: usize },
    Linalg(String),
    EmptyInput,
}
```

## Performance Considerations

### Computational Complexity

- **Manual mode**: O(T × M × L) where T = time periods, M = drivers, L = avg lag length
- **OLS fit**: O(T × P² + P³) where P = total parameters (sum of lags)
  - Dominated by QR decomposition: O(T × P²) when T > P (typical case)
  - Matrix formation: O(T × P)

### Memory Usage

- Design matrix: `T × P × 8 bytes` (f64)
- Augmented matrix (ridge): `(T + P) × P × 8 bytes`

For typical working capital models:

- T = 36-120 periods (3-10 years monthly data)
- P = 5-20 parameters (2-4 drivers, 2-5 lags each)
- Memory: < 100 KB

### Optimization Tips

1. **Use ridge regularization** for better numerical stability when:
   - Drivers are highly correlated
   - Limited historical data (T < 2P)

2. **Choose appropriate lag lengths**:
   - Too long: Overfitting, higher variance
   - Too short: Underfitting, biased estimates
   - Rule of thumb: L ≤ T/4

3. **Preprocessing**:
   - Normalize/standardize drivers if they have very different scales
   - Consider log-transforming drivers with exponential growth

## Troubleshooting

### Common Errors

#### "Linalg(NonInvertible)" or singular matrix

**Cause**: Perfect collinearity in design matrix (e.g., two drivers are identical or linearly dependent)

**Solutions**:

```rust
// Option 1: Use ridge regression
let opts = OlsOptions {
    ridge_lambda: 0.1,
    ..Default::default()
};

// Option 2: Remove or combine collinear drivers
// Option 3: Reduce lag lengths
```

#### "InsufficientData"

**Cause**: Time series too short for requested lags

**Solution**: Reduce lag lengths or provide more historical data

```rust
// Need at least max(lags) periods for burn-in
// For lags=[3, 2], need at least 3 data points
```

#### Poor R² despite low RMSE

**Cause**: Target has low variance or model captures level but not variation

**Solution**:

- Check if intercept should be enabled/disabled
- Verify drivers are actually predictive of target
- Consider differencing/detrending the series

## Examples

Additional complete examples are available in the repository:

- [Basic forecasting workflow](examples/forecast_ar.rs) - End-to-end AR forecasting
- [Cross-validation](examples/cross_validate.rs) - Model selection with train/test splits
- [Multi-driver analysis](examples/multi_driver.rs) - Complex working capital modeling

## Technical Details

### Pure Rust Implementation

This library uses **Linfa** (Rust ML toolkit) for linear regression, which provides:

- Pure Rust QR decomposition (no BLAS/LAPACK required)
- Cross-platform compatibility (Windows, Linux, macOS, even WebAssembly)
- Predictable performance without external library dependencies

### Mathematical Foundation

The FIR model is a discrete-time linear time-invariant (LTI) system:

$$y[t] = \sum_{k=1}^{M} \left(\text{scale}_k \times \sum_{j=0}^{L_k-1} h_k[j] \times \text{driver}_k[t-j]\right)$$

Where:

- $h_k[j]$ are the FIR tap weights (percentages)
- $\text{scale}_k$ is the driver-specific gain
- $M$ is the number of drivers
- $L_k$ is the lag length for driver $k$

This is equivalent to a **distributed lag model** in econometrics or a **convolutional layer** in deep learning (without bias or activation).

## Contributing

Contributions are welcome! Areas of interest:

- Additional regularization methods (elastic net, LASSO)
- Time-varying coefficients
- Seasonal adjustment utilities
- More comprehensive examples

Please open an issue to discuss significant changes before submitting a PR.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Acknowledgments

Built with:

- [ndarray](https://github.com/rust-ndarray/ndarray) - N-dimensional arrays for Rust
- [Linfa](https://github.com/rust-ml/linfa) - Pure Rust machine learning toolkit
- [thiserror](https://github.com/dtolnay/thiserror) - Error handling

Inspired by classical signal processing FIR filters and econometric distributed lag models.

---

**Made with ❤️ in Rust** | [Report Issues](https://github.com/yourusername/wc_fir/issues) | [Documentation](https://docs.rs/wc_fir)
