# API Reference

For full API documentation and docstrings, use the Julia REPL:
```julia
?TimeSeriesVector
?rotary_spectrum
?wavetrans
# ... etc
```

Or see the [GitHub README](https://github.com/alexdominguezg10/ValTools.jl#readme).

## Types

- **TimeSeriesVector**, **TimeSeriesMatrix**, **TimeSeriesCollection** — typed time series with Unitful support
- **SpectralEstimate**, **RotarySpectralEstimate**, **CrossSpectralEstimate** — spectral analysis results
- **EllipsePolarizationEstimate** — polarization parameters
- **WaveletTransform** — wavelet transform results

## Spectral Analysis

### Multitaper methods
- `spectral_multitaper()` — Thomson's multitaper spectrum with F-tests
- `rotary_spectrum()` — rotary spectral decomposition (CCW/CW split)
- `rotary_coherence()` — rotary spectral coherence
- `cross_coherence()` — cross-spectrum coherence

### Wavelet methods
- `wavetrans()` — continuous wavelet transform (generalized Morse wavelets)
- `rotary_wavetrans()` — rotary wavelet transform
- `ridgemap()`, `ridgechains()` — wavelet ridge chain tracking
- `multivariate_ridges()` — joint ridge analysis (N-channel)
- `wavelet_significance()` — Monte Carlo significance testing
- `tiredecode()` — instantaneous frequency, amplitude, phase
- `instmom()` — univariate instantaneous moments
- `jointmom()` — multivariate joint moments

### Polarization
- `ellipse_polarization()` — frequency-domain polarization (frequency-domain)
- `ellsig()`, `ellpol()` — time-domain ellipse synthesis & characterization
- `ellipsefit()` — least-squares ellipse fit
- `msvd()` — SVD-based coherence detection (Park et al. 1987)
- `rotary()` — rotary decomposition

## Model I/O

- `CROCOReader`, `ROMSReader`, `NEMOReader` — read ROMS-family model output
- `ssh()`, `sst()`, `temperature()`, `salinity()`, `velocities()` — extract fields
- `sigma_to_z()`, `interp_z()` — vertical coordinate transforms
- `sigma_to_z_gpu()`, `interp_z_gpu()` — GPU-accelerated transforms

## Observation Loaders

- `ArgoLoader` — Argo float profiles
- `DUACSLoader` — DUACS SSH/SLA satellite data
- `SWOTLoader` — SWOT SSH observations
- `NDBCLoader` — NDBC buoy data (waves, winds)
- `RAFOSLoader` — RAFOS float trajectories
- `IESLoader` — IES acoustic tomography
- `MooringCurrentLoader` — moored current meter arrays
- `ThermistorLoader` — thermistor chains
- `CloudDriftLoader` — CloudDrift trajectories
- `GliderLoader` — glider profiles & sections
- `GEMBuilder` — gravitational equivalent model from Argo

## Colocation & Validation

- `colocate_model_obs()` — match model to observation points
- `colocate_model_grid()` — sample model on observation grid
- `colocate_model_trajectory()` — follow a drifter path
- `colocate_model_mooring()` — extract virtual mooring time series
- `colocate_model_section()` — cross-section colocation

## Validation Metrics

- `compute_metrics()` — RMSE, correlation, bias, skill score
- `taylor_stats()` — Taylor diagram statistics
- `bootstrap_metrics()` — bootstrap confidence intervals
- `ellipse_polarization()` — current ellipse metrics

## Visualization

- `plot_timeseries()`, `plot_spectrum()` — typed dispatch plotting
- `plot_rotary_spectrum()`, `plot_rotary_coherence()`
- `plot_ellipse_polarization()`, `plot_colocation()`
- `taylor_diagram()` — Taylor diagram
- `plot_wind_rose()` — wind rose visualization
- `plot_lic()` — line-integral convolution flow viz
- `animate_field_realtime()` — animated field playback

## Utilities

- `pressure_to_depth()`, `depth_to_pressure()`
- `mooring_knockdown_model()`, `fit_knockdown()` — mooring dynamics
- `virtual_mooring()` — synthesize mooring time series
- `fillgaps()`, `detrend()`, `hilbert()` — signal preprocessing
- `bandpass()`, `highpass()`, `lowpass()` — filtering
