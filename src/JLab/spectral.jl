"""
Multitaper Spectral Analysis

Uses Multitaper.jl for DPSS taper generation and spectral estimation.
GPU acceleration available via CUDA extension when loaded.

**References:**
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  Proceedings of the IEEE, 70(9), 1055–1096.
- Slepian, D. (1978). Prolate spheroidal wave functions, Fourier analysis and uncertainty.
  Bell System Technical Journal, 57(5), 1371–1430.
"""

import FFTW: fft, fftfreq
using Multitaper: multispec, dpss_tapers, MTSpectrum
using ..Types
using Unitful

# ============================================================================
# NEW: Wrapper around Multitaper.jl
# ============================================================================

"""
    spectral_multitaper(x::Union{Vector, Matrix}, dt::Real=1.0;
                       nw::Float64=4.0, ntapers::Int=0,
                       detrend::String="linear")

Multitaper spectral estimation using Multitaper.jl.

Computes power spectral density with adaptive weighting, F-test p-values,
and jackknifed confidence intervals.

# Arguments
- `x`: (N,) vector or (N, M) matrix of time series
- `dt`: sampling interval (default 1.0)
- `nw`: time-bandwidth product (default 4.0)
- `ntapers`: number of tapers (default: 2*nw-1)
- `detrend`: "none", "constant", or "linear" (default "linear")

# Returns
- `Types.SpectralEstimate` (or a `Vector` of them for a batch of `M` signals) with:
  - `freq`: frequency vector (positive frequencies)
  - `power`: power spectral density, carrying `unit^2` when `unit` is given
  - `ftest_pval`: F-test p-values for line components
  - `jkvar`: jackknifed variance (confidence intervals)
  - `params`: NamedTuple of estimation parameters (nw, ntapers, dt, detrend, N)

# Example
```julia
x = randn(1000)
spec = spectral_multitaper(x, 0.1; nw=4.0, unit=u"m/s")
plot(spec.freq, spec.power)
```
"""
function spectral_multitaper(x::Union{Vector{Float64}, Matrix{Float64}},
                             dt::Real=1.0;
                             nw::Float64=4.0, ntapers::Int=0,
                             detrend::String="linear",
                             unit=Unitful.NoUnits)

    # Ensure Float64
    x = Float64.(x)
    if x isa Vector
        x = reshape(x, :, 1)
    end

    N, M = size(x)

    # Default ntapers
    if ntapers <= 0
        ntapers = max(1, 2*floor(Int, nw) - 1)
    end

    # Detrend
    x_detrended = copy(x)
    if detrend == "linear"
        for m in 1:M
            x_detrended[:, m] = _detrend_linear(x_detrended[:, m])
        end
    elseif detrend == "constant"
        for m in 1:M
            x_detrended[:, m] .-= mean(x_detrended[:, m])
        end
    elseif detrend != "none"
        error("detrend must be 'none', 'constant', or 'linear'")
    end

    # Call Multitaper.multispec for each signal
    # Features enabled: adaptive=true, Ftest=true, jk=true
    if M == 1
        spec = multispec(x_detrended[:, 1]; NW=nw, K=ntapers, dt=dt,
                        ctr=false, a_weight=true, Ftest=true, jk=true)
        return _to_spectral_estimate(spec, nw, ntapers, dt, detrend, unit)
    else
        # Batch: return vector of SpectralEstimate
        specs = [multispec(x_detrended[:, m]; NW=nw, K=ntapers, dt=dt,
                          ctr=false, a_weight=true, Ftest=true, jk=true)
                 for m in 1:M]
        return [_to_spectral_estimate(s, nw, ntapers, dt, detrend, unit) for s in specs]
    end
end

# Convert a Multitaper.jl MTSpectrum into a ValTools Types.SpectralEstimate
function _to_spectral_estimate(spec::MTSpectrum, nw, ntapers, dt, detrend, unit)
    freq = collect(Float64, spec.f)
    power = spec.S .* unit^2
    ftest_pval = spec.Fpval === nothing ? nothing : collect(Float64, spec.Fpval)
    jkvar = spec.jkvar === nothing ? nothing : collect(Float64, spec.jkvar)
    params = (nw=nw, ntapers=ntapers, dt=dt, detrend=detrend, N=spec.params.N)
    return Types.SpectralEstimate(freq, power, ftest_pval, jkvar, params)
end


# ============================================================================
# DEPRECATED ALIASES (jLab-compatible names)
# ============================================================================

"""
    sleptap(n::Integer, k::Integer; bandwidth::Real=4.0)

**Deprecated** — thin wrapper over `Multitaper.dpss_tapers` kept for
backward compatibility with jLab's `sleptap.m` and existing call sites.
Prefer calling `spectral_multitaper` directly for new code.

Returns `(tapers, lambdas)`: an `(n, k)` matrix of Slepian (DPSS) tapers
and their concentration eigenvalues, sorted descending.
"""
function sleptap(n::Integer, k::Integer; bandwidth::Real=4.0)
    n > 0 || error("sleptap: n must be positive, got $n")
    k > 0 || error("sleptap: k must be positive, got $k")
    tapers, lambdas = dpss_tapers(n, Float64(bandwidth), k, :both)
    return tapers, lambdas
end

"""
    mspec(x, dt=1.0; ntapers=5, detrend="linear", gpu=false)

**Deprecated** — thin wrapper over [`spectral_multitaper`](@ref) kept for
backward compatibility with jLab's `mspec.m` and existing call sites
(e.g. `JLab.validate_spectra`). Unwraps the `Types.SpectralEstimate`
into a plain `(freqs, psd)` tuple.
"""
function mspec(x::Union{Vector{Float64}, Matrix{Float64}}, dt::Real=1.0;
               ntapers::Int=5, detrend::String="linear", gpu::Bool=false)
    if gpu
        spectral_multitaper_gpu(x, dt; ntapers=ntapers, detrend=detrend, gpu=gpu)
    end
    spec = spectral_multitaper(x, dt; ntapers=ntapers, detrend=detrend)
    return spec.freq, spec.power
end


# ============================================================================
# GPU PATH (loaded by CUDA extension)
# ============================================================================

# Stub: overridden by ValToolsCUDAExt if CUDA is available
function spectral_multitaper_gpu(x::Union{Vector, Matrix}, dt::Real=1.0;
                                  nw::Float64=4.0, ntapers::Int=0,
                                  detrend::String="linear", gpu::Bool=true)
    error("GPU spectral_multitaper requires CUDA.jl. Load it first: `using CUDA`")
end


# ============================================================================
# HELPERS
# ============================================================================

function _detrend_linear(x::AbstractVector)
    n = length(x)
    t = collect(1.0:n)
    A = [ones(n) t]
    coeffs = A \ x
    return x .- A * coeffs
end
