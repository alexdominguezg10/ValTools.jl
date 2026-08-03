"""
    JLab

Julia port of oceanographic signal analysis methods from Jonathan M. Lilly's jLab.

## Overview

This module brings powerful spectral analysis, wavelet transforms, and eddy
characterization tools to Julia, ported from the excellent MATLAB toolbox jLab.

**Original Author & License:**
- Author: Jonathan M. Lilly
- Repository: https://github.com/jonathanlilly/jlab
- License: BSD 2-Clause

**See `ATTRIBUTION.md` in this directory for full references and citations.**

## Submodules

- **Wavelets**: Continuous wavelet transforms, ridge detection, instantaneous properties
- **Spectral**: Multitaper spectral analysis, Slepian tapers, filter design
- **TimeSeries**: Detrending, gap filling, bandpass filtering
- **Ellipse**: Eddy characterization, ellipse fitting, rotary spectra
- **Validation**: Model-observation validation workflows

## GPU Acceleration

When `CUDA.jl` is loaded, GPU-accelerated paths activate automatically:

```julia
using CUDA                       # activates ValToolsCUDAExt
using ValTools.JLab

# Single signal — GPU wavelet transform
wt, scales = wavetrans(x; dt=dt, nv=8, gpu=true)

# Batch of signals — the GPU sweet spot (10–50× speedup on H200).
# wavetrans is N-D (jLab column-signal convention): dim 1 is time, any
# trailing dims are independent signals — e.g., velocity at 100 mooring
# depths as an (N, 100) matrix, or (N, depths, moorings) rank-3.
wt3d, scales = wavetrans(X; dt=dt, nv=8, gpu=true)

# Or pass a CuArray directly — GPU path auto-detected
wt, scales = wavetrans(CuArray(x); dt=dt, nv=8)
```

## Quick Example

```julia
using ValTools.JLab

time = 0:0.01:100
freq_inertial = 0.04  # Hz, typical for GoM
x = cos.(2π .* freq_inertial .* time) + 0.1 .* randn(length(time))

wt, scales = wavetrans(x; dt=0.01, nv=8)
amp   = tiredecode(wt, scales; kind="amp")
phase = tiredecode(wt, scales; kind="phase")
ridge_freq, ridge_amp = ridgemap(wt, scales)
```

## Attribution & Citation

All ported functions are documented with:
1. **Matlab origin** (e.g., jWavelet/wavetrans.m)
2. **Theoretical references** (Lilly & Olhede 2009–2011, etc.)
3. **Usage examples** with oceanographic context

When publishing results using JLab methods, cite the original papers
and acknowledge the jLab toolbox. See ATTRIBUTION.md for exact citations.

## Functions by Category

### Wavelet Analysis
- `wavetrans()` — Continuous wavelet transform
- `tiredecode()` — Extract amplitude, phase, instantaneous frequency
- `transmax()` — Find time-frequency maxima

### Ridge Detection & Tracking
- `ridgemap()` — Identify and map wavelet ridges
- `ridgechains()` — Connect ridge points into continuous chains
- `ridgewalk()` — Follow ridge paths through time-frequency plane
- `wavelet_significance()` — Monte Carlo noise threshold for ridge amplitudes
- `ridge_significant()` — Flag which ridge points clear that threshold

### Spectral Analysis
- `spectral_multitaper()` — Multitaper power spectral density (new, GPU-capable!)
- `mspec()` — Multitaper power spectral density (deprecated)
- `sleptap()` — Generate Slepian (DPSS) tapers (deprecated)
- `polparams()` — Extract polarization parameters

### Ellipse & Eddy Analysis
- `ellipsefit()` — Fit ellipse to velocity hodographs
- `ellpol()` — Polarization analysis
- `rotary()` — Rotary spectral decomposition

### Time Series Utilities
- `bandpass()` — Butterworth bandpass filtering
- `detrend()` — Polynomial detrending
- `fillgaps()` — Interpolate missing data
- `hilbert()` — Analytic signal via Hilbert transform

### Data & Validation
- `load_jdata()` — Load reference datasets (NDBC, drifters, etc.)
- `validate_model_spectra()` — Compare model vs. observation spectra

## Modules Included

"""
module JLab

using LinearAlgebra, Statistics, FFTW, SpecialFunctions, Random
import Unitful
import Dates
using ..Metrics
using ..Types

# Include submodules
include("wavelets.jl")
include("spectral.jl")
include("timeseries.jl")
include("ellipse.jl")
include("validation.jl")
include("jdata.jl")

# Re-export key functions for convenience
export wavetrans, wavetrans_batch, tiredecode, transmax,
       morsewave, morsewave_freq, morsefreq, morseprops, morsespace,
       ridgemap, ridgechains, RidgeEvent, wavelet_significance, ridge_significant,
       rotary_ridge_properties, rotary_ridge_properties_sphere,
       rotary_wavetrans_sphere, density_ratio_significance,
       quadinterp, ridgechains_jlab,
       spectral_multitaper, mspec, sleptap,
       ellipsefit, ellsig, ellpol, rotary, rotary_wavetrans, rotary_ridge,
       bandpass, highpass, lowpass, detrend, fillgaps, hilbert,
       validate_spectra, validate_rotary, validate_model_spectra,
       kinetic_energy_budget, eddy_census,
       inertial_frequency, local_tangent_plane, get_freq_band,
       FREQ_BANDS_HOURS,
       JDATA_CATALOG, jdata_list, jdata_download,
       load_gulfdrifters, load_gulfflow, load_gomed

end # module
