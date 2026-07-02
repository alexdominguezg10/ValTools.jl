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
- `MTSpectrum` struct with:
  - `f`: frequency vector (positive frequencies)
  - `S`: power spectral density
  - `params`: MTParameters (nw, K, dt, etc.)
  - `Fpval`: F-test p-values for line components
  - `jkvar`: jackknifed variance (confidence intervals)

# Example
```julia
x = randn(1000)
spec = spectral_multitaper(x, 0.1; nw=4.0)
plot(spec.f, spec.S)
```
"""
function spectral_multitaper(x::Union{Vector{Float64}, Matrix{Float64}},
                             dt::Real=1.0;
                             nw::Float64=4.0, ntapers::Int=0,
                             detrend::String="linear")

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
        return spec
    else
        # Batch: return vector of MTSpectrum
        specs = [multispec(x_detrended[:, m]; NW=nw, K=ntapers, dt=dt,
                          ctr=false, a_weight=true, Ftest=true, jk=true)
                 for m in 1:M]
        return specs
    end
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
