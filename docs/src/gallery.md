# Examples Gallery

The ValTools.jl gallery showcases every major capability of the package — from basic signal analysis to advanced oceanographic workflows. Each example is **self-contained and runnable** (synthetic data, no external downloads). Figures are generated inline using your plotting standards.

## Getting Started

```@raw html
<div class="demo-grid">
<div class="demo-card-future">
  <h3>💧 5-Minute Tour</h3>
  <p>A gentle introduction to Unitful types, time series objects, and dispatch-based plotting.</p>
</div>
</div>
```

## Signal Analysis

### Wavelets & Ridges (10/10 complete — the foundation)

These examples demonstrate the crown jewel of ValTools.jl: wavelet analysis for oceanographic signals.

```@raw html
<div class="demo-grid">
<a class="demo-card" href="generated/inertial_oscillation.html">
  <img src="assets/inertial_oscillation_thumb.png" alt="Inertial oscillation">
  <h3>🌊 Detecting Inertial Oscillations</h3>
  <p>Simulate 40 days of noisy current-meter data at 25°N and use rotary spectral decomposition to recover the 26.7-hour inertial period. Shows hodograph + rotary spectrum.</p>
</a>

<a class="demo-card" href="generated/eddy_spindown.html">
  <img src="assets/eddy_spindown_thumb.png" alt="Eddy spindown">
  <h3>🌀 Tracking Eddy Decay with Wavelets</h3>
  <p>A rotating eddy slows down over 30 days. The wavelet ridge chain tracks the time-varying frequency and amplitude, capturing the spin-down dynamics.</p>
</a>

<a class="demo-card" href="generated/ridge_ellipse_polarization.html">
  <img src="assets/ridge_ellipse_polarization_thumb.png" alt="Ridge polarization">
  <h3>🔄 Ridge Ellipse Polarization</h3>
  <p>Extract polarization parameters (ellipticity, orientation, phase) from a wavelet ridge. Useful for characterizing rotational modes (clockwise vs. counter-clockwise).</p>
</a>

<a class="demo-card" href="generated/multivariate_common_oscillation.html">
  <img src="assets/multivariate_common_oscillation_thumb.png" alt="Multivariate ridges">
  <h3>📊 Multivariate Wavelet Ridge Analysis</h3>
  <p>Track multiple ocean variables (u, v, SST, SSH) simultaneously using a joint wavelet transform. Identify common oscillations across the system.</p>
</a>

<a class="demo-card" href="generated/svd_polarization_detection.html">
  <img src="assets/svd_polarization_detection_thumb.png" alt="SVD polarization">
  <h3>🔍 SVD-Based Polarization Detection</h3>
  <p>Use singular-value decomposition (Msvd) to extract dominant polarization modes from multi-channel velocity data. No assumptions about rotation sense.</p>
</a>
</div>
```

### Multitaper Spectral Analysis

```@raw html
<div class="demo-grid">
<a class="demo-card" href="generated/multitaper_line_detection.html">
  <img src="assets/multitaper_line_detection_thumb.png" alt="Line detection">
  <h3>📈 Line Detection with F-Tests</h3>
  <p>Identify statistically significant harmonic lines (e.g., M₂ tidal constituent) in noisy spectral estimates using Thomson's harmonic F-test.</p>
</a>

<div class="demo-card-future">
  <h3>🎯 Rotary Spectrum with Confidence Intervals</h3>
  <p>Split velocity time series into counter-clockwise and clockwise components, with jackknife confidence intervals for each.</p>
</div>

<div class="demo-card-future">
  <h3>🌐 2-D Wavenumber Spectra</h3>
  <p>Along-track wavenumber spectrum from SWOT or satellite observations, showing mesoscale/submesoscale structure.</p>
</div>
</div>
```

## Model Readers & Loaders

```@raw html
<div class="demo-grid">
<div class="demo-card-future">
  <h3>🗂️ Observation Zoo (Loaders Overview)</h3>
  <p>A unified tour of all 11 observation loaders (Argo, DUACS, SWOT, NDBC, RAFOS, IES, thermistor, mooring, glider, CloudDrift, CANEK).</p>
</div>

<div class="demo-card-future">
  <h3>💾 Reading CROCO/ROMS Output</h3>
  <p>Load SSH, temperature, salinity, velocities from a NetCDF model output; demonstrate sigma-to-z transform (CPU and GPU).</p>
</div>

<div class="demo-card-future">
  <h3>🏗️ Gravitational Equivalent Model from Argo</h3>
  <p>Build a 3-D density-gradient model from Argo profiles; extract SSH anomaly estimates for comparison with satellite.</p>
</div>
</div>
```

## Colocation & Validation

```@raw html
<div class="demo-grid">
<a class="demo-card" href="generated/mooring_array_batch.html">
  <img src="assets/mooring_array_batch_thumb.png" alt="Mooring colocation">
  <h3>🎯 Colocation: Model vs. Mooring Array</h3>
  <p>Interpolate model fields (CROCO) to a mooring location; compute Taylor diagram statistics and bias/RMSE breakdowns by season.</p>
</a>

<div class="demo-card-future">
  <h3>🚫 Mooring Knockdown Correction</h3>
  <p>Account for vertical motion of moored instruments under strong currents; correct depth-averaged fields using knockdown dynamics.</p>
</div>

<div class="demo-card-future">
  <h3>🌊 Virtual Mooring Synthesis</h3>
  <p>Extract a synthetic mooring time series from model output at arbitrary depths; compare to real mooring records.</p>
</div>

<div class="demo-card-future">
  <h3>🛰️ Drifter Trajectory Colocation</h3>
  <p>Follow a drifter's path with model velocity fields; separate Lagrangian drift from Eulerian shear.</p>
</div>
</div>
```

## GPU Acceleration

```@raw html
<div class="demo-grid">
<div class="demo-card-future">
  <h3>⚡ CPU vs. GPU Wavelet Transform Benchmark</h3>
  <p>Time a batched wavelet transform on CPU (Threads.@threads) vs. GPU (CUDA); show 10–15× speedup on H200.</p>
</div>

<div class="demo-card-future">
  <h3>🎨 LIC Flow Visualization (GPU)</h3>
  <p>Line-Integral Convolution texture rendering of a 2-D velocity field, computed entirely on GPU via CUDA kernels.</p>
</div>
</div>
```

## Case Studies (Flagship Demonstrations)

```@raw html
<div class="demo-grid">
<div class="demo-card-future">
  <h3>🌀 GOMED Eddy Census</h3>
  <p>Extract cyclonic/anticyclonic eddies from CROCO output using wavelet ridge chains; compute census statistics (radius, decay rate, translation speed) and compare to satellite observations.</p>
</div>

<div class="demo-card-future">
  <h3>🌊 GulfDrifters Velocity Wavenumber Spectrum</h3>
  <p>Analyze historical drifter velocities from the GulfDrifters program; compute significant-wavenumber spectrum using rotary-wavelet decomposition.</p>
</div>

<div class="demo-card-future">
  <h3>✅ End-to-End Mooring Validation</h3>
  <p>Start with CROCO model output, extract a virtual mooring at a real mooring location, colocate observations, compute spectral metrics, and build a Taylor diagram — demonstrating the full ValTools.jl pipeline.</p>
</div>
</div>
```

## Roadmap (Future Capabilities)

These methods are in active development and will be added to the gallery as they land:

```@raw html
<div class="demo-grid">
<div class="demo-card-future">
  <h3>📐 Parametric Spectral Analysis</h3>
  <p><b>Planned:</b> Matérn covariance, debiased Whittle likelihood, composite oMp models, transfer function estimation, spectrograms.</p>
  <p><i>Status:</i> 0/5 features implemented; ~30–40h estimated effort.</p>
</div>

<div class="demo-card-future">
  <h3>🧮 N-Dimensional Spectral Methods</h3>
  <p><b>Planned:</b> Extend multitaper and parametric spectral to N-D arrays; support arbitrary domains (spherical, flat, circular).</p>
  <p><i>Status:</i> 1/4 complete (wavelets done 2026-08-03); multitaper N-D, parametric N-D still void.</p>
</div>

<div class="demo-card-future">
  <h3>🌐 Advanced Colocation</h3>
  <p><b>Planned:</b> Section-based colocation (CANEK), trajectory-following on unstructured grids, spherical geometry handling.</p>
  <p><i>Status:</i> Core methods done; 2–3 examples planned.</p>
</div>
</div>
```

---

## About the Gallery

Each example in this gallery is:
- **Self-contained**: generates its own synthetic data or loads a small bundled sample
- **Runnable**: full scripts available for download
- **Verified**: includes assertions against known values (recovered frequency, recovered amplitude, etc.)
- **Cross-referenced**: where applicable, links to jLab (Matlab) or Python implementations

**Want to contribute an example?** See [Contributing Guide](contributing.md) or open an issue on GitHub.
