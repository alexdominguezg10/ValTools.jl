# Changelog

All notable changes to ValTools.jl are documented here. Format loosely
follows [Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Added

- **N-dimensional wavelet support** (jLab trailing-dims semantics —
  Flare Table 1's "N-dimensional support" wavelet void): `wavetrans` now
  accepts any array with time along dimension 1 and independent signals
  along trailing dimensions, returning `(N, n_freqs, trailing...)` from a
  single batched FFT pass; `rotary_wavetrans` likewise. New jLab-style
  multi-input form `wavetrans(x, y, z)` (one shared frequency grid, tuple
  of per-input transforms — `spheretrans.m`'s `[wx,wy,wz]=wavetrans(x,y,z)`),
  now used internally by `rotary_wavetrans_sphere`. `tiredecode` applies
  over trailing dims; ridge functions (`ridgemap`, `ridgechains`,
  `ridgechains_jlab`, `transmax`) get explicit N-D guards with per-signal
  guidance instead of `MethodError`s. `wavelet_significance` batches all
  noise surrogates through one N-D transform (was `n_surrogates`
  sequential calls; identical output for a fixed RNG seed) and gains a
  `gpu` kwarg. Complex-valued `wavetrans` input now genuinely works (was
  broken by an internal `Float64` collect despite the code implying
  support).
- **Typed wavelet API**: new `Types.WaveletTransform` struct; `wavetrans`
  methods on `TimeSeriesVector`/`TimeSeriesMatrix` (dt derived from the
  time axis, units recorded in `params.unit`) and on
  `TimeSeriesCollection` (ragged records → `Vector{WaveletTransform}`);
  `tiredecode(::WaveletTransform)` re-applies the source unit to
  amplitudes. `Types._dt_hours_from_time` hoisted from
  ValToolsMultitaperExt to Types for shared use.
- **MATLAB cross-check harness** `scripts/jlab_crosscheck_wavetrans_nd.jl`
  (+ `.m`): verifies against real jLab that (a) jLab's matrix path equals
  its per-column path (diff 0.0), (b) our N-D path equals our scalar path
  (≤5e-16), (c) our output matches jLab within the scalar port's
  established tolerance (exact at low/mid frequencies; a known
  pre-existing near-Nyquist filter-tail difference bounds the highest
  frequencies at ~2e-4 relative near record edges).

### Fixed

- `wavetrans(x; boundary=:mirror, gpu=true)` returned the wrong `N`
  samples (the first mirrored block instead of the middle — the boundary
  offset was never passed to the GPU path).
- `wavetrans_batch` ignored `boundary` entirely (no kwarg existed); it is
  now a thin wrapper over the N-D `wavetrans` and honors it.
- The `wavetrans(::CuVector)` auto-detect override silently dropped the
  `fs` and `boundary` kwargs; it now forwards all kwargs (and covers any
  `CuArray` rank).

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
- `scripts/validate_gulfdrifters.jl` — GulfDrifters (Lilly & Pérez-Brunius
  2021) baseline validation using `eddy_census()`'s existing fixed-threshold
  heuristic: eddy census counts, mesoscale kinetic-energy fraction, and mean
  rotary coefficient, against the public GulfDriftersOpen dataset (2684/2731
  usable drifters). **2804 events, 51 longest-subset (≥180d) events, 54.4%
  mesoscale KE fraction, −0.228 mean rotary coefficient.**
- `scripts/validate_gulfdrifters_significant.jl` — a second GulfDrifters
  validation combining `eddy_census`'s ridge-chaining with
  `wavelet_significance()`/`ridge_significant()` (genuine Monte Carlo
  noise-surrogate significance testing, as `eddy_census`'s own docstring
  recommends, rather than a fixed amplitude threshold). Runs on a
  fixed-seed 150-drifter subsample (full census not practical — calibrated
  at ~10s/drifter). **466 significant events (3.11/drifter) with
  `background=:red`**, the noise model adopted as default after a
  same-subsample comparison against `background=:white` (787 events,
  5.25/drifter) — red noise is the more defensible null model for
  reddish oceanographic background turbulence, matching a direct
  observation in Lilly & Pérez-Brunius (2021) Sect. 4.3 that white noise
  generates more spurious ridges than red.
  - **None of these numbers should be read as "the" GulfDrifters baseline.**
    They aren't comparable to each other (different method, and the
    significance-test number is a 150-drifter subsample, not a full
    census) or to the paper's own published result: Lilly & Pérez-Brunius
    (2021), Sect. 4.7, report **1033 statistically significant ridges**
    (41% of 2520 non-inertial) on their full, partly-proprietary
    `GulfDriftersAll` dataset (3770 drifters — larger than our public
    2731-drifter `GulfDriftersOpen` subset) using a proper density-ratio
    significance criterion (ρ_X = L·ξ̄⁴ against noise-surrogate survival
    functions) that neither script here implements. The paper's own
    published eddy census (GOMED, Zenodo DOI `10.5281/zenodo.3978803`) is
    access-restricted; access was requested but not granted as of this
    release. Revisit once available.

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
- **`spectral_multitaper_batch_gpu` was completely unreachable** — flagged
  as "has an issue" since Phase 2 but never diagnosed. Root cause: it was
  defined as a bare, unqualified function inside `ValToolsCUDAExt`'s own
  module instead of extending `ValTools.JLab` via `JL.` qualification (the
  pattern every other GPU function in that extension correctly uses), so
  `ValTools.JLab.spectral_multitaper_batch_gpu` genuinely didn't exist.
  Fixed by adding a stub in `src/JLab/spectral.jl` and requalifying both
  GPU spectral functions as `JL.spectral_multitaper_gpu`/
  `JL.spectral_multitaper_batch_gpu`. Note for anyone touching this
  pattern again: the stub must be **loosely typed**
  (`AbstractMatrix`/`AbstractVector`), not matching the extension's
  concrete `Matrix{Float64}` signature exactly — an identical signature
  makes Julia treat the override as an illegal same-method redefinition
  across modules, rejected at precompilation even though it works at
  runtime. **Verified on Ixachi (H200): 46.2× GPU speedup** (warm timing,
  2048×10 batch), clearing the roadmap's "15× GPU maintained" target.
- `test_gpu_final.jl` referenced a stale `spec.S` field (renamed `.power`
  in the Phase 2 type refactor) and had no GPU warm-up call before timing
  (same cold-start-inflates-the-number issue as its own CPU "Test 1").
- `load_gulfdrifters()`'s docstring promised a `time` field the function
  never actually returned (the NetCDF has one; the reader just didn't
  extract it) — now extracted and segmented the same way as
  `lon`/`lat`/`u`/`v`, purely additive to the returned `NamedTuple`.

### Known gaps (carried forward, not blocking this release)

- **GulfDrifters baseline is not settled** — see the three non-comparable
  numbers under Added, above. Revisit once GOMED access is granted, or a
  full (non-subsampled) significance-test census is run.
- `envs/cpu/Project.toml` and `envs/gpu/Project.toml` don't list
  `Multitaper` as a direct dependency (only the published package's
  weakdep mechanism provides it), which meant every ad hoc validation
  script this release needed a throwaway scratch environment instead of
  using the tracked dev envs directly. Undecided whether to add it there
  (this is a different tradeoff than the published-package GPL
  consideration that justified the weakdep in the first place, since these
  are internal dev/test environments, not the public package).
- `RAFOSLoader`'s `bbox`/`date_range`/`pressure_range` filters and
  `rafos_velocity_estimates()` are silent no-ops due to a `:col in
  names(df)` vs. `hasproperty(df, :col)` bug (DataFrames.jl returns
  `Vector{String}` from `names()`, so the `Symbol in` check is always
  `false`). Found 2026-07-03/04, reported, not yet fixed pending
  confirmation this is production code safe to change.
- No CPU/GPU performance regression baseline exists yet to compare against
  in future releases — `test_cpu_perf.jl` records this release's CPU
  throughput as a first data point; the 46.2× GPU number above is this
  release's first GPU data point too.
