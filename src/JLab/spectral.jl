"""
Multitaper Spectral Analysis

Real implementations live in the ValToolsMultitaperExt package extension,
activated automatically when `using Multitaper` is loaded alongside
`using ValTools`. GPU acceleration available via CUDA extension when loaded.

**References:**
- Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
  Proceedings of the IEEE, 70(9), 1055–1096.
- Slepian, D. (1978). Prolate spheroidal wave functions, Fourier analysis and uncertainty.
  Bell System Technical Journal, 57(5), 1371–1430.
"""

using ..Types
using Unitful

# ============================================================================
# Wrapper around Multitaper.jl (implemented in ValToolsMultitaperExt)
# ============================================================================

"""
    spectral_multitaper(x::Union{Vector, Matrix}, dt::Real=1.0;
                       nw::Float64=4.0, ntapers::Int=0,
                       detrend::String="linear", unit=Unitful.NoUnits)

Multitaper spectral estimation using Multitaper.jl.

Computes power spectral density with adaptive weighting, F-test p-values,
and jackknifed confidence intervals.

# Arguments
- `x`: (N,) vector or (N, M) matrix of time series
- `dt`: sampling interval (default 1.0)
- `nw`: time-bandwidth product (default 4.0)
- `ntapers`: number of tapers (default: 2*nw-1)
- `detrend`: "none", "constant", or "linear" (default "linear")
- `unit`: Unitful unit of `x`'s values (default dimensionless); `power` carries `unit^2`

# Returns
- `Types.SpectralEstimate` (or a `Vector` of them for a batch of `M` signals) with:
  - `freq`: frequency vector (positive frequencies)
  - `power`: power spectral density, carrying `unit^2` when `unit` is given
  - `ftest_pval`: F-test p-values for line components
  - `jkvar`: jackknifed variance (confidence intervals)
  - `params`: NamedTuple of estimation parameters (nw, ntapers, dt, detrend, N)

# Example
```julia
using ValTools, Multitaper, Unitful
x = randn(1000)
spec = spectral_multitaper(x, 0.1; nw=4.0, unit=u"m/s")
plot(spec.freq, spec.power)
```

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function spectral_multitaper(x::Union{Vector, Matrix},
                             dt::Real=1.0;
                             nw::Float64=4.0, ntapers::Int=0,
                             detrend::String="linear",
                             unit=Unitful.NoUnits)
    error("spectral_multitaper requires Multitaper.jl. Load it first: `using Multitaper`")
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

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function sleptap(n, k; bandwidth::Real=4.0)
    error("sleptap requires Multitaper.jl. Load it first: `using Multitaper`")
end

"""
    mspec(x, dt=1.0; ntapers=5, detrend="linear", gpu=false)

**Deprecated** — thin wrapper over [`spectral_multitaper`](@ref) kept for
backward compatibility with jLab's `mspec.m` and existing call sites
(e.g. `JLab.validate_spectra`). Unwraps the `Types.SpectralEstimate`
into a plain `(freqs, psd)` tuple.

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function mspec(x::Union{Vector, Matrix}, dt::Real=1.0;
               ntapers::Int=5, detrend::String="linear", gpu::Bool=false)
    error("mspec requires Multitaper.jl. Load it first: `using Multitaper`")
end


# ============================================================================
# GPU PATH (loaded by CUDA extension)
# ============================================================================

# Stub: overridden by ValToolsCUDAExt if CUDA (and Multitaper) is available
function spectral_multitaper_gpu(x::Union{Vector, Matrix}, dt::Real=1.0;
                                  nw::Float64=4.0, ntapers::Int=0,
                                  detrend::String="linear", gpu::Bool=true)
    error("GPU spectral_multitaper requires CUDA.jl and Multitaper.jl. Load them first: `using CUDA, Multitaper`")
end

# Stub: overridden by ValToolsCUDAExt if CUDA is available. Deliberately
# looser-typed (AbstractMatrix/AbstractVector) than the extension's
# concrete Matrix{Float64}/Vector{Float64} override -- matching that exact
# signature here would make Julia treat the two as the SAME method
# (defined in two different modules), which precompilation rejects as
# "Method overwriting is not permitted" even though the override is
# working as intended. A looser stub signature keeps them as genuinely
# distinct, non-conflicting methods.
function spectral_multitaper_batch_gpu(X::AbstractMatrix, tapers::AbstractMatrix,
                                        lambdas::AbstractVector, dt::Real, N_fft::Int)
    error("GPU spectral_multitaper_batch_gpu requires CUDA.jl. Load it first: `using CUDA`")
end
