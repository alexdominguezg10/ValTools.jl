# Changelog

All notable changes to ValTools.jl are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [0.2.0] — 2026-07-31

First documented release since the type-hierarchy redesign. Covers the
4-stage improvements roadmap (rotary spectra consolidation → type system →
polarization/coherence metrics → typed plotting), started 2026-07-02.

### Added

- **Type hierarchy**: `TimeSeriesVector`, `TimeSeriesMatrix`,
  `TimeSeriesCollection`, `SpectralEstimate`, `RotarySpectralEstimate`,
  `RotaryCoherenceEstimate`, `CrossSpectralEstimate`,
  `EllipsePolarizationEstimate`, `ColocatedObservation` — concrete,
  Unitful-aware structs with multiple dispatch, replacing raw `NamedTuple`/
  tuple returns across loaders, metrics, and spectral estimators. Hard
  break from the old tuple-return convention (no back-compat shim), except
  `RotarySpectralEstimate` which supports tuple destructuring
  (`f, ccw, cw = spec`) for the old `(freqs, S_ccw, S_cw)` call pattern.
- **Rotary spectral analysis**: unified `rotary_spectrum()` (replacing two
  independent prior implementations), with jackknife confidence intervals
  and Thomson (1982) harmonic F-test significance per CW/CCW branch.
- **Rotary and non-rotary coherence**: `rotary_coherence()` (Gonella 1972 /
  Mooers 1973 / Kundu 1976 CW/CCW decomposition) and `cross_coherence()`
  (real-signal counterpart), both with significance thresholds.
- **Ellipse polarization**: `ellipse_polarization()` — spectral-matrix
  eigendecomposition (port of jLab's `polparams`/`specdiag`) giving
  major/minor-axis power, orientation, and Cartesian/rotary polarization
  ratios, with jackknife CIs including a **circular** jackknife for the
  orientation angle `theta` (handles the branch-cut wraparound a plain
  linear jackknife gets wrong).
- **Significance testing**: `ftest_ccw`/`ftest_cw` on rotary spectral
  peaks; `wavelet_significance` + `ridge_significant` (Monte Carlo
  white/red-noise surrogates, Torrence & Compo 1998 style) for wavelet
  ridge detections, since no analytic reconstruction-factor formula exists
  for the generalized Morse wavelets this package uses.
- **Makie plotting recipes per type**: `plot_timeseries`, `plot_spectrum`,
  `plot_rotary_spectrum`, `plot_rotary_coherence`, `plot_cross_spectrum`,
  `plot_ellipse_polarization`, `plot_colocation` — dispatch across the
  typed structs above, with axis labels generated from Unitful units.
- **New observation loaders**: CloudDrift, Glider.
- Examples gallery (Oceananigans-style).
- `scripts/validate_gulfdrifters.jl` — GulfDrifters (Lilly & Perez-Brunius
  2021) baseline validation: eddy census counts, mesoscale kinetic-energy
  fraction, and mean rotary coefficient, against the public
  GulfDriftersOpen dataset.

### Changed

- **Multitaper.jl is now a weak dependency** (package extension,
  `ValToolsMultitaperExt`), not a hard dependency — avoids forcing
  Multitaper's GPL license on downstream users who don't need multitaper
  spectral estimation. `mspec`/`sleptap`/`spectral_multitaper` all require
  `using Multitaper` before use.
- `spectral_multitaper` returns `Types.SpectralEstimate` instead of
  Multitaper.jl's raw `MTSpectrum`.

### Fixed

- `correlation`/`skill_score`/`validate` silently mishandled mixed
  Unitful units instead of erroring.
- `TimeSeriesVector`/`TimeSeriesMatrix`/`SpectralEstimate` type-parameter
  arity bug: `Unitful.Quantity` has three type parameters (value, dimension,
  units), not two — an earlier 2-parameter struct definition silently
  matched no real `Quantity`, causing confusing `MethodError`s on every
  construction attempt.
- `eddy_census` docstring accuracy.
- `EllipsePolarizationEstimate.P` docstring had the circular-vs-linear
  interpretation backwards (`P≈1` for *both* rectilinear and circular
  signals, not one or the other; `P≈0` only for isotropic noise).

### Known gaps (carried forward, not blocking this release)

- `load_gulfdrifters()`'s docstring promises a `time` field that the
  function doesn't actually return (the underlying NetCDF has one; the
  reader just doesn't extract it). `scripts/validate_gulfdrifters.jl`
  works around this by assuming the documented hourly sampling rather than
  reading `time` a second time.
- No CPU/GPU performance regression baseline exists yet to compare against
  — `test_cpu_perf.jl` records today's CPU throughput as a first data
  point; GPU numbers require an Ixachi run (see release notes).
- `spectral_multitaper_batch_gpu` was reported as having an unresolved
  issue as of the Phase 2 work; not independently re-verified here.
