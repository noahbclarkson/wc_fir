# Repository Guidelines

## Project Structure & Module Organization

`src/lib.rs` exposes the public API and re-exports helpers from the focused modules: `fir.rs` for manual FIR application, `ols.rs` for regression-based fitting, `data.rs` for alignment and validation utilities, `lasso.rs`/`select.rs` for automatic lag selection, and `types.rs` for shared structs and errors. Module-level integration tests live at the bottom of `src/lib.rs`; add similar coverage next to the code you touch. Build artifacts land in `target/`—never commit that directory. Root-level `Cargo.toml` and `Cargo.lock` pin dependencies; update them together when bumping versions.

## Build, Test, and Development Commands

- `cargo check` — Quick type-check to validate incremental changes.
- `cargo fmt` — Format the codebase with the canonical Rust style.
- `cargo clippy -- -D warnings` — Lint with Clippy and fail on new warnings.
- `cargo test` — Run unit and integration tests across all modules.
- `cargo run --example <name>` — Smoke-check example binaries whenever you tweak documentation or the default settings.
- `cargo doc --open` — Optional: render API docs locally when reviewing public exports.

## Coding Style & Naming Conventions

Rely on `rustfmt` defaults (4-space indentation, trailing commas where idiomatic). Keep modules small and cohesive; prefer top-level functions over deeply nested closures for readability. Use descriptive, snake_case names for functions and variables, and PascalCase for public structs/enums such as `ManualProfile`. Surface error paths through `thiserror`-backed types in `types.rs` rather than ad-hoc strings. Document public functions with Rustdoc examples wherever possible.

## Testing Guidelines

Extend existing tests with targeted scenarios that cover new lag logic, error branches, or regression behavior. Prefer colocated `#[cfg(test)]` modules for unit tests and add `tests/` integration suites when exercising end-to-end flows. Run `cargo test -- --nocapture` during debugging to inspect numeric output, but keep assertions deterministic. Aim to maintain or improve coverage by verifying both manual and OLS pathways.

## Commit & Pull Request Guidelines

Follow the repo’s history of concise, sentence-case commit subjects written in the imperative mood (e.g., “Add OLS ridge guard”). Group related changes into a single commit to keep diffs reviewable. Pull requests should describe the problem, summarize the solution, and call out impacts on numerical results or API signatures. Link GitHub issues when applicable, attach before/after metrics for financial accuracy changes, and include screenshots or tables only when visual evidence clarifies the outcome.
