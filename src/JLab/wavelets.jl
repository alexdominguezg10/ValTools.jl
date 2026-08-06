"""
Continuous Wavelet Transform, Ridge Detection, and Time-Frequency Analysis

Ported from Jonathan M. Lilly's jWavelet & jRidges (https://github.com/jonathanlilly/jlab)

Calibrated against jLab's exact algorithms:
- Morse wavelet: ψ̂(ω) = aₚ ω^β exp(-ω^γ), bandpass-normalized (peak = 1)
- Scale generation: morsespace log-spaced with proper high/low frequency cutoffs
- Factor of 2 for real-valued signals (one-sided analytic wavelet)
- Ridge chaining: greedy frequency-matching with α=0.25 threshold

GPU acceleration: pass `gpu=true` — all FFTs run on GPU via CUFFT.

**References:**
- Lilly, J. M., & S. C. Olhede (2009). Bivariate instantaneous frequency and bandwidth.
  IEEE Trans. Sig. Proc., 57(2), 555–569. https://doi.org/10.1109/TSP.2009.2015841
- Lilly, J. M., & S. C. Olhede (2010, 2011). Generalized Morse wavelets.
- Lilly, J. M. (2011). Element analysis. J. Comput. Graph. Stat., 20(4), 907–929.
"""

import FFTW: fft, ifft, fftfreq, fftshift

# ============================================================================
# MORSE WAVELET CORE — calibrated to match jLab exactly
# ============================================================================

"""
    morsefreq(gamma, beta)

Peak (modal) radian frequency of the Morse wavelet.

    ωₘ = (β/γ)^(1/γ)

# References
jWavelet/morsefreq.m; Lilly & Olhede (2009)
"""
function morsefreq(gamma::Real, beta::Real)
    beta == 0 && return log(2)^(1/gamma)
    return (beta / gamma)^(1 / gamma)
end

"""
    morseprops(gamma, beta)

Time-frequency properties of the Morse wavelet.

# Returns
- `P`: Time-frequency product √(βγ)
- `skew`: Skewness (γ-3)/P
- `kurt`: Kurtosis 3 - skew² - 2/P²
"""
function morseprops(gamma::Real, beta::Real)
    P = sqrt(beta * gamma)
    skew = (gamma - 3) / P
    kurt = 3 - skew^2 - 2 / P^2
    return P, skew, kurt
end

"""
    morseafun(gamma, beta; norm="bandpass")

Normalization constant for the Morse wavelet.

- Bandpass: `aₚ = (γe/β)^(β/γ)` — sets peak of ψ̂(ω) to 1
- Energy: unit L² energy

# References
jWavelet/morseafun.m
"""
function morseafun(gamma::Real, beta::Real; norm::String="bandpass")
    if norm == "bandpass"
        beta == 0 && return 1.0
        return (gamma * exp(1) / beta)^(beta / gamma)
    elseif norm == "energy"
        r = (2beta + 1) / gamma
        return sqrt(2π * gamma * 2^r * gamma_func(1) / gamma_func(r))
    else
        error("Unknown normalization: $norm")
    end
end

gamma_func(x) = exp(logabsgamma(x)[1])

"""
    morsewave_freq(N::Int, gamma::Real, beta::Real, fs::AbstractVector;
                   norm="bandpass")

Build Morse wavelet filter bank in frequency domain.

Returns matrix `(N, length(fs))` where each column is the wavelet at one
analysis frequency `fs[j]`, ready for element-wise multiply with the FFT
of the signal.

The wavelet is:
    ψ̂(ω) = aₚ · ω^β · exp(-ω^γ)

evaluated at ω = 2π·k/(N·fact) where fact = fs[j] / ωₘ.

# References
jWavelet/morsewave.m (morsewave1 subroutine)
"""
function morsewave_freq(N::Int, gamma::Real, beta::Real,
                        fs::AbstractVector;
                        norm::String="bandpass")

    omega_peak = morsefreq(gamma, beta)
    a_norm = morseafun(gamma, beta; norm=norm)

    n_fs = length(fs)
    bank = zeros(Float64, N, n_fs)

    for j in 1:n_fs
        fact = fs[j] / omega_peak

        for k in 1:N
            omega = 2π * (k - 1) / N / fact
            if omega > 0
                # ψ̂(ω) = aₚ · ω^β · exp(-ω^γ)
                log_psi = log(a_norm) + beta * log(omega) - omega^gamma
                bank[k, j] = exp(log_psi)
            end
        end

        # Half the DC component (Heaviside convention)
        bank[1, j] *= 0.5

        # Energy normalization: √(1/fact) for each scale
        if norm == "energy"
            bank[:, j] .*= sqrt(1.0 / fact)
        end
    end

    return bank
end

"""
    morsewave(N::Int, gamma::Real, beta::Real; dtype=ComplexF64)

Generate a single generalized Morse wavelet (time + frequency domain).

# References
jWavelet/morsewave.m; Lilly & Olhede (2009, 2010, 2011)
"""
function morsewave(N::Int, gamma::Real, beta::Real; dtype=ComplexF64)
    (N < 1) && error("N must be positive")
    (gamma <= 0 || beta <= 0) && error("gamma and beta must be positive")

    omega_peak = morsefreq(gamma, beta)
    fs_single = [omega_peak]  # at peak frequency
    bank = morsewave_freq(N, gamma, beta, fs_single)
    psi_f = bank[:, 1]
    psi = ifft(complex.(psi_f)) .* N
    return psi, psi_f, 1.0
end

# ============================================================================
# MORSESPACE — calibrated frequency vector generation
# ============================================================================

"""
    morsespace(gamma, beta, N; density=4, eta=0.1, R=5,
               f_high=nothing, f_low=nothing)

Generate log-spaced analysis frequencies for the Morse wavelet, matching
jLab's morsespace.m algorithm.

# Arguments
- `gamma`, `beta`: Morse wavelet parameters
- `N`: Length of time series
- `density`: Frequency overlap (default 4; higher = finer resolution)
- `eta`: Nyquist decay fraction (default 0.1)
- `R`: Packing number — minimum wavelet widths that fit in signal (default 5)
- `f_high`, `f_low`: Override frequency limits (radian)

# Returns
- `fs::Vector{Float64}`: Analysis frequencies in radians per sample (descending)

# References
jWavelet/morsespace.m
"""
function morsespace(gamma::Real, beta::Real, N::Int;
                    density::Int=4, eta::Real=0.1, R::Real=5.0,
                    f_high::Union{Real, Nothing}=nothing,
                    f_low::Union{Real, Nothing}=nothing)

    P, _, _ = morseprops(gamma, beta)
    omega_peak = morsefreq(gamma, beta)

    # High-frequency cutoff: largest f where ψ̂(π) ≤ η·ψ̂(ωₘ)
    if f_high === nothing
        f_high = _morsehigh(gamma, beta, eta)
    end
    f_high = min(Float64(f_high), π)

    # Low-frequency cutoff: R wavelet widths must fit in N samples
    if f_low === nothing
        f_low = 2 * sqrt(2) * P * R / N
    end
    f_low = max(Float64(f_low), 2π / N)

    if f_low >= f_high
        return [f_high]
    end

    # Logarithmic spacing: r = 1 + 1/(D·P)
    r = 1.0 + 1.0 / (density * P)
    n_f = floor(Int, log(f_high / f_low) / log(r))
    n_f = max(n_f, 1)

    fs = f_high ./ r .^ (0:n_f)
    return collect(fs)
end

function _morsehigh(gamma::Real, beta::Real, eta::Real)
    omega_peak = morsefreq(gamma, beta)
    a_norm = morseafun(gamma, beta)

    # Evaluate log(ψ̂) on dense grid, find where it drops below log(η)
    log_peak = log(a_norm) + beta * log(omega_peak) - omega_peak^gamma
    log_threshold = log_peak + log(eta)

    # Search: what f_high makes ψ̂(π·ωₘ/f_high) = η·ψ̂(ωₘ)?
    # Equivalently: at frequency f, the wavelet sees ω = π/fact = π·ωₘ/f
    best_f = π  # start at Nyquist
    for trial_f in range(0.01, π; length=10000)
        omega = π * omega_peak / trial_f
        if omega > 0
            log_psi = log(a_norm) + beta * log(omega) - omega^gamma
            if log_psi >= log_threshold
                best_f = trial_f
                break
            end
        end
    end
    return best_f
end

# ============================================================================
# FILTER BANK — build all scaled wavelets at once (GPU-friendly)
# ============================================================================

"""
    _build_filter_bank(N_fft, fs, gamma, beta; norm="bandpass")

Build the wavelet filter bank for analysis frequencies `fs`.
Returns `(N_fft, length(fs))` real matrix.

Uses the calibrated morsewave_freq which matches jLab exactly.
"""
function _build_filter_bank(N_fft::Int, fs::AbstractVector,
                            gamma::Real, beta::Real;
                            norm::String="bandpass")
    return morsewave_freq(N_fft, gamma, beta, fs; norm=norm)
end

# ============================================================================
# WAVELET TRANSFORM — calibrated, dispatches CPU or GPU
# ============================================================================

"""
    wavetrans(X::AbstractArray{<:Number}; dt=1.0, fs=nothing, nv=8,
              mother="morse", gamma=3.0, beta=8.0, gpu=false, boundary=:zeros)
    wavetrans(x, y, zs...; kwargs...)

Continuous wavelet transform using generalized Morse wavelets,
calibrated to match jLab's wavetrans.m.

Key differences from a naive CWT:
- Morse wavelet is bandpass-normalized (peak of ψ̂ = 1)
- Real-valued input is multiplied by 2 (one-sided analytic wavelet)
- Frequencies auto-generated via morsespace (log-spaced, proper cutoffs)
- Linear detrend applied by default

Dimension 1 of `X` is time; any trailing dimensions index independent
signals, following jLab's column-signal convention (`wavetrans.m` on a
matrix transforms each column). A vector input returns a `(N, n_freqs)`
matrix as before; an `(N, d2, d3, ...)` array returns
`(N, n_freqs, d2, d3, ...)`, every signal sharing one frequency grid and
one batched FFT. Complex-valued input is transformed as-is (no factor of
2), isolating its positive-rotary content — this is what
[`rotary_wavetrans`](@ref) builds on.

The multi-argument form `wavetrans(x, y, z)` mirrors jLab's
`[wx,wy,wz]=wavetrans(x,y,z,...)` (see `spheretrans.m`): all inputs must
have identical size and all-real or all-complex element type; they are
transformed in a single batched call sharing one frequency grid, and the
per-input transforms are returned as a tuple, with `fs` last.

# Arguments
- `X`: Input time series (time along dimension 1)
- `dt`: Sampling interval (default 1.0)
- `fs`: Analysis frequencies in rad/sample (auto via morsespace if nothing)
- `nv`: Voices per octave (used only if fs=nothing; mapped to density)
- `mother`: "morse" (default)
- `gamma`, `beta`: Morse wavelet parameters (default 3.0, 8.0)
- `gpu`: Use GPU (default false)
- `boundary`: `:zeros` (default) or `:mirror` (see `_boundary_pad`)

# Returns
- `wt::Array{ComplexF64}`: Wavelet coefficients `(N, n_freqs, trailing dims...)`
- `fs::Vector{Float64}`: Analysis frequencies (rad/sample)

# References
Lilly & Olhede (2009); jWavelet/wavetrans.m
"""
function wavetrans(X::AbstractArray{<:Number};
                   dt::Real=1.0,
                   fs::Union{AbstractVector, Nothing}=nothing,
                   scales::Union{AbstractVector, Nothing}=nothing,
                   nv::Int=8,
                   mother::String="morse",
                   gamma::Real=3.0,
                   beta::Real=8.0,
                   gpu::Bool=false,
                   boundary::Symbol=:zeros)

    N = size(X, 1)
    (N < 1) && error("Input must have at least 1 sample")
    trailing = size(X)[2:end]
    n_sig = prod(trailing)

    g = Float64(gamma)
    b = Float64(beta)

    # Flatten trailing signal dims to columns; keep real input real so the
    # forward FFT and the one-sided factor-of-2 convention below apply.
    T = eltype(X) <: Real ? Float64 : ComplexF64
    Xm = Matrix{T}(reshape(X, N, n_sig))

    # Linear detrend (jLab default), all columns at once
    Xm = _detrend_signal(Xm)

    fs_v = _resolve_freq_grid(N, g, b, nv, fs, scales, dt)

    X_pad, N_fft, offset = _boundary_pad(Xm, N, boundary)

    # Build filter bank (calibrated Morse wavelet)
    bank = _build_filter_bank(N_fft, fs_v, g, b)

    # Dispatch
    if gpu
        wt3 = _wavetrans_nd_gpu(X_pad, bank, N, offset)
    else
        wt3 = _wavetrans_nd_cpu(X_pad, bank, N, offset)
    end

    # Factor of 2 for real-valued input (jLab convention)
    if eltype(X) <: Real
        wt3 .*= 2
    end

    wt = reshape(wt3, N, length(fs_v), trailing...)
    return wt, fs_v
end

# jLab multi-input form: [wx,wy,wz] = wavetrans(x,y,z,...) — one batched
# transform, one shared frequency grid, per-input outputs returned as a
# tuple with fs last (spheretrans.m:76 is the canonical jLab call site).
function wavetrans(x::AbstractArray{<:Number}, y::AbstractArray{<:Number},
                   rest::AbstractArray{<:Number}...; kwargs...)
    inputs = (x, y, rest...)
    sz = size(x)
    all(size(a) == sz for a in inputs) ||
        error("wavetrans: all inputs must have identical size (got $(map(size, inputs)))")
    # Mixing real and complex inputs would silently change the real inputs'
    # scaling (the one-sided ×2 convention applies only to all-real batches)
    all(a -> eltype(a) <: Real, inputs) || all(a -> eltype(a) <: Complex, inputs) ||
        error("wavetrans: inputs must be all-real or all-complex, not mixed")
    stacked = cat(inputs...; dims=ndims(x) + 1)
    wt, fs_v = wavetrans(stacked; kwargs...)
    nd = ndims(wt)
    outs = ntuple(k -> copy(selectdim(wt, nd, k)), length(inputs))
    return (outs..., fs_v)
end

# Shared frequency-grid resolution (same precedence as always: an explicit
# `scales=` wins for backward compatibility, then `fs=`, then morsespace).
function _resolve_freq_grid(N::Int, g::Float64, b::Float64, nv::Int,
                            fs::Union{AbstractVector, Nothing},
                            scales::Union{AbstractVector, Nothing}, dt::Real)
    if fs === nothing && scales === nothing
        P, _, _ = morseprops(g, b)
        density = max(1, round(Int, nv * P / 4))
        return morsespace(g, b, N; density=density)
    elseif scales !== nothing
        # Backward compatibility: convert scales to radian frequencies
        return 2π .* collect(Float64, scales) .* dt
    end
    return collect(Float64, fs)
end

# Boundary-condition padding before the wavelet FFT (timeseries_boundary.m).
# `:zeros` (this file's long-standing default): zero-pad to the next power
# of 2 beyond 2N-1, extract the FIRST N samples of the ifft -- functionally
# jLab's 'zeros' option, though sized for FFT efficiency rather than
# exactly N. `:mirror`: reflect the series about both endpoints
# (`[flipud(x);x;flipud(x)]`, exactly 3N, no further padding -- matches
# jLab's `timeseries_boundary.m:98-99` exactly), extract the MIDDLE N
# samples (`index=N+1:2N`, `wavetrans_continue`, jLab's own extraction
# after the mirror pad). `eddyridges_loop` (jOceans/eddyridges.m:562-564)
# explicitly requests 'mirror', not jLab's own wavetrans default of
# 'periodic' -- so :mirror is the one to use for eddy-ridge-faithful
# analysis; :zeros stays the default here for backward compatibility with
# every other caller in this file.
# Columns are independent signals; padding acts along dimension 1 only.
function _boundary_pad(X::AbstractMatrix{T}, N::Int, boundary::Symbol) where {T<:Union{Float64, ComplexF64}}
    if boundary === :zeros
        N_fft = 2^ceil(Int, log2(2*N - 1))
        return vcat(X, zeros(T, N_fft - N, size(X, 2))), N_fft, 0
    elseif boundary === :mirror
        return vcat(reverse(X; dims=1), X, reverse(X; dims=1)), 3N, N
    else
        error("boundary must be :zeros or :mirror, got $boundary")
    end
end

# Least squares against a real [1 t] design matrix; for complex input this
# decouples into independent detrends of the real and imaginary parts, so
# the rotary u+iv path detrends identically to detrending u and v separately.
function _detrend_signal(X::AbstractVecOrMat{T}) where {T<:Union{Float64, ComplexF64}}
    n = size(X, 1)
    n < 2 && return X
    t = collect(1.0:n)
    A = [ones(n) t]
    return X .- A * (A \ X)
end

# ── CPU path ────────────────────────────────────────────────────────────────

# One forward FFT for every signal at once, then a scale loop broadcasting
# each filter across all signal columns — memory stays O(N_fft × n_sig) per
# step regardless of how many scales there are.
function _wavetrans_nd_cpu(X_pad::AbstractMatrix{<:Number}, bank::Matrix{Float64},
                           N::Int, offset::Int)
    Xf = fft(X_pad, 1)
    n_scales = size(bank, 2)
    n_sig = size(X_pad, 2)
    wt = Array{ComplexF64, 3}(undef, N, n_scales, n_sig)

    for j in 1:n_scales
        Y = Xf .* conj.(@view(bank[:, j]))     # conj(ψ̂) · X̂ (jLab convention)
        Yt = ifft(Y, 1)
        @views wt[:, j, :] .= Yt[offset+1:offset+N, :]
    end
    return wt
end

# ── GPU path (stub — implemented by ValToolsCUDAExt) ────────────────────────

function _wavetrans_nd_gpu(X_pad, bank, N, offset)
    error("""GPU wavelet transform requires CUDA.jl.
    Load it first:  `using CUDA`
    Then call:       `wavetrans(x; gpu=true)`""")
end

# ============================================================================
# ROTARY WAVELET TRANSFORM & RIDGE ANALYSIS
# ============================================================================

"""
    rotary_wavetrans(u, v; dt=1.0, fs=nothing, nv=8, gamma=3.0, beta=8.0)

Rotary (CW/CCW) wavelet decomposition of a velocity time series `w = u + iv`,
extending the CW/CCW split of [`Metrics.rotary_spectrum`](@ref) to the
time-frequency domain (Lilly & Olhede 2010).

Because a Morse wavelet is analytic (one-sided: its spectrum has support only
at positive frequencies), transforming the complex signal `w` directly
isolates its counter-clockwise (positive-frequency) content, while
transforming `conj(w) = u - iv` with the *same* wavelet bank isolates the
clockwise (negative-frequency) content — no separate CW filter bank needed.

# Arguments
- `u`, `v`: east-west / north-south velocity components (identical size;
  time along dimension 1, trailing dimensions index independent signals,
  exactly as in [`wavetrans`](@ref))
- `dt`, `fs`, `nv`, `gamma`, `beta`: as in [`wavetrans`](@ref)

# Returns
- `wt_ccw::Array{ComplexF64}`: CCW-rotating wavelet coefficients `(N, n_freqs, trailing dims...)`
- `wt_cw::Array{ComplexF64}`: CW-rotating wavelet coefficients, same shape
- `fs::Vector{Float64}`: analysis frequencies (rad/sample), shared by both

Both `wt_ccw` and `wt_cw` are ordinary `wavetrans`-shaped outputs, so
[`ridgemap`](@ref), [`ridgechains`](@ref), and [`tiredecode`](@ref) all work
on them directly — or use [`rotary_ridge`](@ref) for a one-call ridge
summary of both components.

# References
Lilly, J. M. & Olhede, S. C. (2010). Bivariate instantaneous frequency and
bandwidth. IEEE Trans. Signal Process., 58(2), 591–603.
"""
function rotary_wavetrans(u::AbstractArray{<:Real}, v::AbstractArray{<:Real};
                          dt::Real=1.0,
                          fs::Union{AbstractVector, Nothing}=nothing,
                          nv::Int=8,
                          gamma::Real=3.0,
                          beta::Real=8.0,
                          boundary::Symbol=:zeros)
    size(u) == size(v) || error("u and v must have the same size")
    N = size(u, 1)
    (N < 1) && error("Input must have at least 1 sample")

    g, b = Float64(gamma), Float64(beta)

    # Resolve the grid once from N so both branches share it exactly
    fs_v = _resolve_freq_grid(N, g, b, nv, fs, nothing, dt)

    # An analytic (one-sided) wavelet on w = u+iv isolates CCW content; on
    # conj(w) = u-iv, CW content. wavetrans detrends internally (complex
    # detrend ≡ separate detrends of u and v; see _detrend_signal).
    #
    # jLab's real wavetrans.m documents an explicit 1/sqrt(2) prefactor for
    # complex (bivariate) input: WP = (1/sqrt(2))*(WX+iWY). This was missing
    # here, making every rotary wavelet/ridge amplitude sqrt(2) too large
    # (frequency/timing unaffected, since those come from ratios/phase, not
    # amplitude). Found 2026-08-05 crosschecking against real jLab
    # wavetrans.m/ridgewalk.m -- see scripts/jlab_crosscheck_flare_fig23.{jl,m}.
    W = (ComplexF64.(u) .+ im .* ComplexF64.(v)) ./ sqrt(2)
    wt_ccw, _ = wavetrans(W; fs=fs_v, gamma=g, beta=b, boundary=boundary)
    wt_cw, _ = wavetrans(conj.(W); fs=fs_v, gamma=g, beta=b, boundary=boundary)

    return wt_ccw, wt_cw, fs_v
end

"""
    rotary_ridge(u, v; dt=1.0, fs=nothing, nv=8, gamma=3.0, beta=8.0, thresh=0.0)

Wavelet ridge tracking of the CW and CCW rotary components of a velocity
time series, i.e. the time-varying dominant frequency and amplitude of each
sense of rotation (e.g. tracking an inertial oscillation's frequency drift).
Combines [`rotary_wavetrans`](@ref) with [`ridgemap`](@ref) on each branch.

# Returns
A `NamedTuple` with:
- `freq_ccw`, `amp_ccw`: ridge frequency/amplitude of the CCW component at each time
- `freq_cw`, `amp_cw`: ridge frequency/amplitude of the CW component at each time
- `rotary_coefficient`: instantaneous `(amp_ccw² - amp_cw²) / (amp_ccw² + amp_cw²)`,
  `NaN` wherever either ridge is undetected
"""
function rotary_ridge(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                      dt::Real=1.0,
                      fs::Union{AbstractVector, Nothing}=nothing,
                      nv::Int=8,
                      gamma::Real=3.0,
                      beta::Real=8.0,
                      thresh::Real=0.0)
    wt_ccw, wt_cw, fs_out = rotary_wavetrans(u, v; dt=dt, fs=fs, nv=nv, gamma=gamma, beta=beta)

    freq_ccw, amp_ccw = ridgemap(wt_ccw, fs_out; thresh=thresh)
    freq_cw, amp_cw = ridgemap(wt_cw, fs_out; thresh=thresh)

    total = amp_ccw .^ 2 .+ amp_cw .^ 2
    rotary_coefficient = fill(NaN, length(total))
    valid = .!isnan.(amp_ccw) .& .!isnan.(amp_cw) .& (total .> 0)
    rotary_coefficient[valid] = (amp_ccw[valid] .^ 2 .- amp_cw[valid] .^ 2) ./ total[valid]

    return (freq_ccw=freq_ccw, amp_ccw=amp_ccw, freq_cw=freq_cw, amp_cw=amp_cw,
            rotary_coefficient=rotary_coefficient)
end

# ============================================================================
# BATCH WAVELET TRANSFORM
# ============================================================================

"""
    wavetrans_batch(X::AbstractMatrix; dt=1.0, fs=nothing, nv=8,
                    gamma=3.0, beta=8.0, gpu=false, boundary=:zeros)

Wavelet transform of multiple signals simultaneously.
Each column of `X` is an independent time series.

Superseded by calling [`wavetrans`](@ref) directly on the matrix (or any
higher-rank array) — kept as a compatibility wrapper; it now also honors
`boundary`, which the original standalone implementation did not.

# Returns
- `wt::Array{ComplexF64, 3}`: `(N, n_freqs, n_signals)`
- `fs::Vector{Float64}`
"""
wavetrans_batch(X::AbstractMatrix; kwargs...) = wavetrans(X; kwargs...)

# ============================================================================
# DECODE: amplitude, phase, instantaneous frequency & bandwidth
# ============================================================================

"""
    tiredecode(wt::AbstractMatrix{<:Complex}, fs::AbstractVector;
               kind::String="amp")

Extract amplitude, phase, or instantaneous frequency from wavelet transform.

# Arguments
- `wt`: Wavelet coefficients from `wavetrans()`
- `fs`: Analysis frequencies
- `kind`: `"amp"` | `"phase"` | `"freq"` | `"bandwidth"`

# References
Lilly & Olhede (2009); jRidges/instmom.m
"""
function tiredecode(wt::AbstractMatrix{<:Complex}, fs::AbstractVector;
                    kind::String="amp")

    size(wt, 2) == length(fs) || error("fs must match columns of wt")

    kind == "amp"       && return abs.(wt)
    kind == "phase"     && return angle.(wt)
    kind == "freq"      && return _instantaneous_frequency(wt)
    kind == "bandwidth" && return _instantaneous_bandwidth(wt)
    kind == "curvature" && return instmom(wt)[4]
    error("Unknown kind: $kind (use \"amp\", \"phase\", \"freq\", \"bandwidth\", \"curvature\")")
end

# N-D method (trailing dims = independent signals, matching wavetrans's
# output layout): amp/phase are pointwise and apply directly; freq/bandwidth/
# curvature run the 2-D kernel per signal slice. The Matrix method above
# stays the dispatch target for 2-D input.
function tiredecode(wt::AbstractArray{<:Complex}, fs::AbstractVector;
                    kind::String="amp")
    size(wt, 2) == length(fs) || error("fs must match dimension 2 of wt")

    kind == "amp"   && return abs.(wt)
    kind == "phase" && return angle.(wt)
    kind in ("freq", "bandwidth", "curvature") ||
        error("Unknown kind: $kind (use \"amp\", \"phase\", \"freq\", \"bandwidth\", \"curvature\")")

    N, nf = size(wt, 1), size(wt, 2)
    trailing = size(wt)[3:end]
    w3 = reshape(wt, N, nf, :)
    T = kind == "curvature" ? ComplexF64 : Float64
    out = Array{T, 3}(undef, N, nf, size(w3, 3))
    for k in axes(w3, 3)
        out[:, :, k] = tiredecode(@view(w3[:, :, k]), fs; kind=kind)
    end
    return reshape(out, N, nf, trailing...)
end

function _instantaneous_frequency(wt::AbstractMatrix)
    nt, ns = size(wt)
    inst_freq = zeros(Float64, nt, ns)

    for j in 1:ns
        phase = angle.(wt[:, j])
        # Unwrap phase
        for i in 2:nt
            dp = phase[i] - phase[i-1]
            phase[i] -= 2π * round(dp / (2π))
        end
        # Central differences (jLab's vdiff with endpoint boundary)
        for i in 2:nt-1
            inst_freq[i, j] = (phase[i+1] - phase[i-1]) / 2.0
        end
        inst_freq[1, j] = phase[2] - phase[1]
        inst_freq[nt, j] = phase[nt] - phase[nt-1]
    end
    return inst_freq
end

function _instantaneous_bandwidth(wt::AbstractMatrix)
    nt, ns = size(wt)
    bw = zeros(Float64, nt, ns)

    for j in 1:ns
        lnamp = log.(max.(abs.(wt[:, j]), 1e-300))
        for i in 2:nt-1
            bw[i, j] = (lnamp[i+1] - lnamp[i-1]) / 2.0
        end
        bw[1, j] = lnamp[2] - lnamp[1]
        bw[nt, j] = lnamp[nt] - lnamp[nt-1]
    end
    return bw
end

# ============================================================================
# RIDGE DETECTION — amplitude ridges (jLab isridgepoint.m)
# ============================================================================

"""
    ridgemap(wt::AbstractMatrix{<:Complex}, fs::AbstractVector;
             thresh::Real=0.0, quality::Bool=false)

Identify wavelet ridges: local amplitude maxima along the frequency axis
at each time step.

# References
Lilly & Olhede (2009); jRidges/isridgepoint.m, ridgemap.m
"""
function ridgemap(wt::AbstractMatrix{<:Complex}, fs::AbstractVector;
                  thresh::Real=0.0, quality::Bool=false)

    amp = abs.(wt)
    nt, ns = size(amp)

    ridge_freq = fill(NaN, nt)
    ridge_amp  = fill(NaN, nt)
    ridge_qual = fill(NaN, nt)
    ridge_idx  = zeros(Int, nt)

    for i in 1:nt
        row = @view amp[i, :]
        mx  = maximum(row)
        mx <= thresh && continue

        # Find all local maxima along frequency axis
        for j in 2:ns-1
            if row[j] >= row[j-1] && row[j] >= row[j+1] && row[j] > thresh
                # Take the strongest local maximum
                if row[j] > ridge_amp[i] || isnan(ridge_amp[i])
                    ridge_amp[i]  = row[j]
                    ridge_freq[i] = fs[j]
                    ridge_idx[i]  = j

                    if quality
                        q = 1.0
                        q *= (row[j] - row[j-1]) / (row[j] + 1e-10)
                        q *= (row[j] - row[j+1]) / (row[j] + 1e-10)
                        ridge_qual[i] = clamp(q, 0.0, 1.0)
                    end
                end
            end
        end

        # Check endpoints
        if ns >= 2
            if row[1] >= row[2] && row[1] > thresh && (isnan(ridge_amp[i]) || row[1] > ridge_amp[i])
                ridge_amp[i] = row[1]; ridge_freq[i] = fs[1]; ridge_idx[i] = 1
            end
            if row[ns] >= row[ns-1] && row[ns] > thresh && (isnan(ridge_amp[i]) || row[ns] > ridge_amp[i])
                ridge_amp[i] = row[ns]; ridge_freq[i] = fs[ns]; ridge_idx[i] = ns
            end
        end
    end

    return quality ? (ridge_freq, ridge_amp, ridge_qual) : (ridge_freq, ridge_amp)
end

# ── N-D input guards ─────────────────────────────────────────────────────────
# Ridge analysis is inherently per-signal: a ridge is a path through ONE
# signal's (time × scale) plane, so there is no meaningful joint ridge over
# a stack of independent signals. These guards turn what would otherwise be
# a MethodError (or, for transmax, a silently wrong flattened answer) into
# actionable guidance for the N-D arrays wavetrans now returns.
function _nd_ridge_error(fname::String)
    error("$fname operates on a single signal's (time × scale) wavelet " *
          "transform. For the N-D output of wavetrans (trailing dims = " *
          "independent signals), apply it per signal, e.g. " *
          "`[$fname(wt[:, :, k], fs) for k in axes(wt, 3)]`.")
end

ridgemap(wt::AbstractArray{<:Complex}, fs::AbstractVector; kwargs...) =
    _nd_ridge_error("ridgemap")
ridgechains(wt::AbstractArray{<:Complex}, fs::AbstractVector; kwargs...) =
    _nd_ridge_error("ridgechains")
ridgechains_jlab(wt::AbstractArray{<:Complex}, fs::AbstractVector; kwargs...) =
    _nd_ridge_error("ridgechains_jlab")
transmax(wt::AbstractArray{<:Complex}; kwargs...) =
    _nd_ridge_error("transmax")

# ============================================================================
# RIDGE CHAINING — ridgechains/ridgewalk (jLab algorithm)
# ============================================================================

"""
    RidgeEvent

A single coherent ridge (eddy, oscillation) extracted from the wavelet transform.
"""
struct RidgeEvent
    start::Int                    # first time index
    stop::Int                     # last time index
    duration::Int                 # number of time steps
    freq::Vector{Float64}        # instantaneous frequency along ridge
    amp::Vector{Float64}         # amplitude along ridge
    scale_idx::Vector{Int}       # scale index at each time step
end

"""
    ridgechains(wt::AbstractMatrix{<:Complex}, fs::AbstractVector;
                alpha::Real=0.25, min_length::Real=1.0,
                thresh::Real=0.0)

Extract continuous ridge chains from the wavelet transform.

Implements jLab's ridgechains.m algorithm:
1. Find all ridge points (amplitude maxima along frequency axis)
2. Chain them across time using frequency continuity
3. Filter by minimum duration

# Arguments
- `wt`: Wavelet coefficients `(N, n_freqs)`
- `fs`: Analysis frequencies
- `alpha`: Maximum normalized frequency mismatch for chaining (default 0.25)
- `min_length`: Minimum ridge duration in wavelet periods (default 1.0)
- `thresh`: Amplitude threshold (fraction of max; default 0.0)

# Returns
- `events::Vector{RidgeEvent}`: Detected ridge chains

# References
Lilly & Olhede (2009); Lilly (2011); jRidges/ridgechains.m, ridgewalk.m
"""
function ridgechains(wt::AbstractMatrix{<:Complex}, fs::AbstractVector;
                     alpha::Real=0.25, min_length::Real=1.0,
                     thresh::Real=0.0)

    amp = abs.(wt)
    nt, ns = size(wt)
    abs_thresh = thresh > 0 ? thresh * maximum(amp) : 0.0

    # Step 1: Find all ridge points at each time step
    # Multiple ridges per time step are allowed
    ridge_points = Vector{Vector{NamedTuple{(:j, :amp, :freq), Tuple{Int, Float64, Float64}}}}(undef, nt)

    for i in 1:nt
        pts = NamedTuple{(:j, :amp, :freq), Tuple{Int, Float64, Float64}}[]
        row = @view amp[i, :]
        for j in 2:ns-1
            if row[j] >= row[j-1] && row[j] >= row[j+1] && row[j] > abs_thresh
                push!(pts, (j=j, amp=row[j], freq=fs[j]))
            end
        end
        # Endpoints
        if ns >= 2
            row[1] >= row[2] && row[1] > abs_thresh && push!(pts, (j=1, amp=row[1], freq=fs[1]))
            row[ns] >= row[ns-1] && row[ns] > abs_thresh && push!(pts, (j=ns, amp=row[ns], freq=fs[ns]))
        end
        ridge_points[i] = pts
    end

    # Step 2: Chain ridge points using greedy frequency matching
    # Each active chain tracks: current frequency, predicted next frequency
    active_chains = Dict{Int, Vector{Tuple{Int, Int, Float64, Float64}}}()
    # chain_id => [(time, scale_idx, freq, amp), ...]
    chain_id_counter = 0
    all_chains = Vector{Vector{Tuple{Int, Int, Float64, Float64}}}()

    for i in 1:nt
        pts = ridge_points[i]
        isempty(pts) && continue

        used_pts = falses(length(pts))
        used_chains = Set{Int}()

        # Try to extend each active chain
        for (cid, chain) in active_chains
            isempty(chain) && continue
            last_t, last_j, last_f, last_a = chain[end]

            # Skip if chain is not at previous time step
            last_t != i - 1 && continue

            # Predict next frequency (linear extrapolation)
            if length(chain) >= 2
                prev_f = chain[end-1][3]
                dfdt = last_f - prev_f
                pred_f = last_f + dfdt
            else
                pred_f = last_f
            end

            # Find best matching point
            best_k = 0
            best_df = Inf
            for k in 1:length(pts)
                used_pts[k] && continue
                df = abs(pts[k].freq - pred_f) / abs(last_f + 1e-20)
                if df < best_df && df < alpha
                    best_df = df
                    best_k = k
                end
            end

            if best_k > 0
                push!(chain, (i, pts[best_k].j, pts[best_k].freq, pts[best_k].amp))
                used_pts[best_k] = true
                push!(used_chains, cid)
            end
        end

        # Close chains that weren't extended
        for (cid, chain) in collect(active_chains)
            if !(cid in used_chains)
                push!(all_chains, chain)
                delete!(active_chains, cid)
            end
        end

        # Start new chains for unmatched points
        for k in 1:length(pts)
            used_pts[k] && continue
            chain_id_counter += 1
            active_chains[chain_id_counter] = [(i, pts[k].j, pts[k].freq, pts[k].amp)]
        end
    end

    # Close remaining active chains
    for (_, chain) in active_chains
        push!(all_chains, chain)
    end

    # Step 3: Filter by minimum duration
    P, _, _ = morseprops(Float64(3.0), Float64(8.0))  # default for filtering
    events = RidgeEvent[]

    for chain in all_chains
        length(chain) < 2 && continue

        dur = chain[end][1] - chain[1][1] + 1
        mean_freq = mean(c[3] for c in chain)

        # Minimum length in wavelet periods: L * 2P/π
        # One period = 2π/ω, so min_steps = min_length * 2π/mean_freq
        min_steps = max(2, ceil(Int, min_length * 2π / (abs(mean_freq) + 1e-20)))

        dur < min_steps && continue

        push!(events, RidgeEvent(
            chain[1][1],                               # start
            chain[end][1],                             # stop
            dur,                                        # duration
            [c[3] for c in chain],                     # freq
            [c[4] for c in chain],                     # amp
            [c[2] for c in chain],                     # scale_idx
        ))
    end

    # Sort by start time
    sort!(events; by=e -> e.start)

    return events
end

# ============================================================================
# UTILITY: transmax
# ============================================================================

"""
    transmax(wt::AbstractMatrix{<:Complex}; n::Int=1)

Find the `n` largest-amplitude points in the wavelet time-frequency plane.
"""
function transmax(wt::AbstractMatrix{<:Complex}; n::Int=1)
    amp_flat = vec(abs.(wt))
    idx = partialsortperm(amp_flat, 1:min(n, length(amp_flat)); rev=true)

    nt = size(wt, 1)
    itime  = @. (idx - 1) % nt + 1
    iscale = @. (idx - 1) ÷ nt + 1
    return itime, iscale, amp_flat[idx]
end

# ============================================================================
# WAVELET RIDGE SIGNIFICANCE (Monte Carlo, Torrence & Compo 1998 in spirit)
# ============================================================================

"""
    wavelet_significance(x; dt=1.0, fs=nothing, nv=8, gamma=3.0, beta=8.0,
                        background=:white, confidence=0.95, n_surrogates=200,
                        gpu=false, rng=Random.default_rng())

Per-scale significance threshold for the continuous wavelet power spectrum
of `x`, for distinguishing a genuine ridge (an eddy, an inertial
oscillation) from what pure background noise could produce at each scale.

Unlike Torrence & Compo (1998)'s analytic chi-square formula — which relies
on a wavelet-specific "reconstruction factor" only worked out for the
Morlet wavelet — this estimates the threshold empirically: it generates
`n_surrogates` random realizations of `background` noise (`:white`, or
`:red` i.e. lag-1 AR(1) matched to `x`'s empirical lag-1 autocorrelation)
with the same length and variance as `x`, wavelet-transforms each with
[`wavetrans`](@ref) using the same parameters as the real signal, and takes
the per-scale `confidence` quantile of each surrogate's *time-maximum*
power (not the pointwise power) — i.e. "how strong can the single best
noise fluctuation get, at this scale, over a record this long." That
time-maximum is the right comparison for ridge detection, which itself
asks about the strongest point at each scale, not the average.

# Returns
- `sig_level::Vector{Float64}`: per-scale threshold on `abs2.(wt)`, same
  length as `fs`. A ridge is significant at scale `j` if its amplitude
  `amp[j]` satisfies `amp[j]^2 > sig_level[j]`; see [`ridge_significant`](@ref)
  for applying this directly to [`ridgemap`](@ref)/[`rotary_ridge`](@ref) output.
- `fs::Vector{Float64}`: analysis frequencies (rad/sample), matching `wavetrans`.

# References
Torrence, C. & Compo, G. P. (1998). A practical guide to wavelet analysis.
Bull. Amer. Meteor. Soc., 79(1), 61–78. (Monte Carlo alternative to their
eq. 17–18 analytic formula, used here since no equivalent closed-form
reconstruction factor is established for the generalized Morse wavelets
used by [`wavetrans`](@ref).)
"""
function wavelet_significance(x::AbstractVector{<:Real};
                              dt::Real=1.0,
                              fs::Union{AbstractVector, Nothing}=nothing,
                              nv::Int=8,
                              gamma::Real=3.0,
                              beta::Real=8.0,
                              background::Symbol=:white,
                              confidence::Real=0.95,
                              n_surrogates::Int=200,
                              gpu::Bool=false,
                              rng::Random.AbstractRNG=Random.default_rng())
    xv = collect(Float64, x)
    N = length(xv)
    N < 4 && error("x must have at least 4 samples")
    background in (:white, :red) || error("background must be :white or :red")
    n_surrogates > 1 || error("n_surrogates must be > 1")

    mu = mean(xv)
    sigma = std(xv)

    _, fs_ref = wavetrans(xv; dt=dt, fs=fs, nv=nv, gamma=gamma, beta=beta)
    n_freq = length(fs_ref)

    alpha = 0.0
    if background == :red
        xc = xv .- mu
        denom = sum(abs2, xc)
        alpha = denom > 0 ? clamp(sum(xc[1:end-1] .* xc[2:end]) / denom, 0.0, 0.95) : 0.0
    end

    # Generate all surrogates first (same RNG consumption order as the old
    # one-at-a-time loop), then transform them in a single batched N-D call
    # — one FFT pass instead of n_surrogates sequential wavetrans calls.
    surrogates = Matrix{Float64}(undef, N, n_surrogates)
    for s in 1:n_surrogates
        surrogates[:, s] = background == :white ? mu .+ sigma .* randn(rng, N) :
                                                  _ar1_surrogate(N, alpha, mu, sigma, rng)
    end
    wt_all, _ = wavetrans(surrogates; dt=dt, fs=fs_ref, nv=nv, gamma=gamma,
                          beta=beta, gpu=gpu)
    # (N, n_freq, n_surrogates) -> per-surrogate time-maximum power (n_surrogates, n_freq)
    max_power = permutedims(dropdims(maximum(abs2.(wt_all); dims=1); dims=1))

    sig_level = [quantile(@view(max_power[:, j]), confidence) for j in 1:n_freq]
    return sig_level, fs_ref
end

function _ar1_surrogate(N::Int, alpha::Real, mu::Real, sigma::Real, rng::Random.AbstractRNG)
    innov_sd = sigma * sqrt(max(1 - alpha^2, 1e-12))
    z = Vector{Float64}(undef, N)
    z[1] = sigma * randn(rng)
    for i in 2:N
        z[i] = alpha * z[i-1] + innov_sd * randn(rng)
    end
    return mu .+ z
end

"""
    ridge_significant(ridge_freq, ridge_amp, sig_level, fs)

Flag which points of a [`ridgemap`](@ref)/[`rotary_ridge`](@ref) ridge are
significant against the noise threshold from [`wavelet_significance`](@ref).

`ridge_freq[i]` is matched to the nearest entry of `fs` (they coincide
exactly when both come from the same `wavetrans` call), and the point is
flagged significant when `ridge_amp[i]^2 > sig_level[j]` at that scale.
`NaN` ridge points (undetected at that time) are always `false`.

# Returns
`BitVector`, same length as `ridge_freq`/`ridge_amp`.
"""
function ridge_significant(ridge_freq::AbstractVector{<:Real},
                           ridge_amp::AbstractVector{<:Real},
                           sig_level::AbstractVector{<:Real},
                           fs::AbstractVector{<:Real})
    length(ridge_freq) == length(ridge_amp) || error("ridge_freq and ridge_amp must have the same length")
    length(sig_level) == length(fs) || error("sig_level and fs must have the same length")

    out = falses(length(ridge_freq))
    for i in eachindex(ridge_freq)
        f = ridge_freq[i]
        isnan(f) && continue
        j = argmin(abs.(fs .- f))
        out[i] = ridge_amp[i]^2 > sig_level[j]
    end
    return out
end

# ============================================================================
# ONE-SIDED ROTARY RIDGES WITH ELLIPSE PROPERTIES
# (Lilly & Perez-Brunius 2021, Sects. 2.2, 3.5, 3.7, 4.4)
# ============================================================================

"""
    rotary_ridge_properties(u, v; dt=1.0, fs=nothing, nv=8, gamma=3.0, beta=2.0,
                            f_coriolis=1.0, min_cycles=2*sqrt(beta*gamma)/pi, alpha=0.25,
                            abs_floor=0.0, fmax_ratio=nothing, fmin_ratio=nothing,
                            D=nothing, eta=0.05)

Extract **one-sided** wavelet ridges from a bivariate signal `(u, v)` and
summarize each one by its ellipse properties, following Lilly &
Perez-Brunius (2021, NPG).

**`u`, `v` must be the eddy-induced DISPLACEMENT (position residual), not
velocity, to match jLab's actual GOMED-generating pipeline.** jLab's real
`eddyridges.m` wavelet-transforms POSITION -- for real (lat, lon)
trajectories, via `spheretrans.m:76`
(`[wx,wy,wz]=wavetrans(x,y,z,...)` on the 3D Cartesian projection of
lat/lon, then reprojected onto a local tangent plane) -- never velocity,
despite Lilly's own comment left in `spheretrans.m` ("I think that the
real answer is that I should decompose velocity, not displacement") that
this function was never revisited to actually do that. Confirmed
empirically (2026-08-01, "ValTools 7"): swapping this function's input
from GulfDriftersOpen velocity (`u`,`v`) to a simple local tangent-plane
position projection (`x=R·cos(lat₀)·(lon-lon₀)`, `y=R·(lat-lat₀)`) took
the 28-case jLab cross-check harness (`scripts/jlab_crosscheck_*`) from
mean Jaccard 0.575 (0/28 clean matches) to **0.957 (13/28 clean, 27/28
"good")** -- the single largest factor found in the whole ridge-chaining
investigation, well past any of the chaining/masking/frequency-grid fixes
that came before it (all of which are still real and still needed, just
not sufficient alone). See project_gomed_validation_results memory for
the full trail. The argument names `u`,`v` are kept for now since the
underlying math is signal-agnostic (works on any bivariate real time
series) -- callers doing GOMED-faithful analysis must pass position, not
velocity; this has NOT yet been wired into
`scripts/case_study_gomed.jl`/`scripts/validate_gulfdrifters_significant.jl`,
which still pass velocity as of this session.

"One-sided" (their Sect. 3.7) means a ridge is not permitted to switch
rotation sense partway through: ridges are extracted twice, once with
CW-dominant regions masked out and once with CCW-dominant regions masked
out, and the two sets are combined. Real eddies do not reverse their sense
of rotation, so ridges that do are artifacts of the background turbulence —
excluding them structurally suppresses false positives, without having to
impose an arbitrary eccentricity cutoff.

Per-ridge properties come from the two rotary transform amplitudes
`a⁺ = |w⁺|` and `a⁻ = |w⁻|`:

- `κ² = (a⁺)² + (a⁻)²` — squared RMS ellipse radius (their Eq. 28)
- `ξ = ((a⁺)² − (a⁻)²)/κ²` — signed circularity in `[-1, 1]` (their Eq. 9);
  `+1` = circular counter-clockwise, `-1` = circular clockwise, `0` = linear
- `ω* = sgn(ξ)·ω/f₀` — nondimensional signed frequency (their Eq. 63),
  positive for cyclonic and negative for anticyclonic motion in the
  Northern Hemisphere
- `L = (1/2π)∫ω dt` — ridge length in number of cycles executed (their Eq. 60)

Ridge-averaged quantities (`xi_bar`, `omega_ast_bar`) are **κ²-weighted**
time averages along the ridge, per their Eq. 74 — not plain means, so that
the high-amplitude portion of a ridge dominates its summary values.

# Arguments
- `u`, `v`: east-west / north-south components of a bivariate signal (equal
  length) -- displacement/position residual for GOMED-faithful analysis
  (see note above), not velocity, despite the variable names
- `dt`: sampling interval (only used to build analysis frequencies when
  `fs === nothing`; `L`/`omega` come back in radians per sample regardless)
- `f_coriolis`: local Coriolis frequency **in radians per sample**, i.e. the
  same units as the analysis frequencies, used to nondimensionalize `ω*`.
  Defaults to `1.0`, which leaves `omega_ast_bar` as a plain radian
  frequency — pass a real value to get the paper's nondimensional quantity.
- `min_cycles`: minimum ridge length in cycles (their Eq. 61 threshold,
  `2P/π` where `P=√(βγ)` is the wavelet duration — defaults to
  `2√6/π ≈ 1.6` for the default `(β,γ)=(2,3)` wavelet, and tracks whatever
  `beta`/`gamma` are actually passed rather than staying fixed at that
  number; an earlier version hardcoded `2√6/π` regardless of `beta`, which
  was silently wrong for any non-default `beta`)
- `alpha`: frequency-continuity tolerance passed to [`ridgechains`](@ref)
- `abs_floor`: absolute-frequency floor passed to
  [`ridgechains_jlab`](@ref)'s relative-error denominator (radians per
  sample, same units as `f_coriolis`). Defaults to `0.0` (jLab's own
  unfloored relative criterion); nonzero values counter fragmentation of
  slow eddies -- see [`ridgechains_jlab`](@ref) for why.
- `fmax_ratio`, `fmin_ratio`: when BOTH given (and `fs` is left `nothing`),
  restrict the analysis frequencies to jLab's own eddy band from
  `eddyridges.m`: `[fmin_ratio, fmax_ratio] * f_coriolis`, rather than the
  generic full-spectrum grid `rotary_wavetrans` otherwise builds. This is
  NOT the default because `f_coriolis` itself defaults to the `1.0`
  placeholder (see above) -- applying a Coriolis-relative band to that
  placeholder would silently mis-restrict any caller not doing real
  physical-unit analysis. Pass real values together with a real
  `f_coriolis` to opt in. Matches `eddyridges_loop`'s own construction
  (`jOceans/eddyridges.m:552-555`):
  `f_high = min(fmax_ratio*f_coriolis, morsehigh(gamma,beta,eta))`,
  `f_low = max(fmin_ratio*f_coriolis, π*P/N)` where `P=√(βγ)` -- the
  second term is a record-length floor that only binds for pathologically
  short records; for typical GulfDrifters segments `fmin_ratio*f_coriolis`
  dominates. jLab's actual Gulf-census call
  (`jFigures/makefigs_gulfcensus.m:460`,
  `onegulfeddy=eddyridges(...,2,1/64,sqrt(6),1,0)`) uses `fmax_ratio=2`,
  `fmin_ratio=1/64` -- pass those explicitly to match GOMED.
- `D`: density override for the eddy-band grid (only used when
  `fmax_ratio`/`fmin_ratio` are both given). Defaults to `16`, matching
  `eddyridges.m`'s own hardcoded `D=16` -- deliberately NOT the generic
  `morsespace`/`rotary_wavetrans` default of `4`ish (finer frequency
  resolution is what makes the eddy band usable at all for short records).
- `eta`: Nyquist-decay threshold for the eddy-band high-frequency cutoff
  (only used when `fmax_ratio`/`fmin_ratio` are both given). Defaults to
  `0.05`, matching `eddyridges.m`'s hardcoded `alpha=0.05` -- NOT
  `morsespace`'s generic default of `0.1`.

# Frequency-grid gap this fixes
Before this, `rotary_ridge_properties` always used `rotary_wavetrans`'s
generic full-spectrum grid (an N-independent-of-physics "5 wavelets must
pack into the record" heuristic, `f_low = 2√2·P·5/N`), never
`eddyridges.m`'s actual Coriolis-relative eddy band. For short drifter
segments (~80 days) this generic cutoff sits at a HIGHER frequency (~13.5d
period) than jLab's real `FMIN=1/64`-of-Coriolis band would allow (~38d
period) -- so slow, large eddies (16+ day periods) fell right at or past
the grid's structurally-excluded edge scale and fragmented into many
short, wrong-frequency ridge pieces, independent of any ridge-chaining
tolerance. Confirmed via direct comparison against jLab's own
`eddyridges.m`/`morsespace.m` source (2026-08-01) -- see
project_gomed_validation_results memory for the case-by-case numbers.

# Returns
`Vector{NamedTuple}`, one per surviving ridge, with fields `start`, `stop`,
`npoints`, `L`, `xi_bar`, `omega_ast_bar`, `kappa_bar`, `sense`
(`:ccw`/`:cw`, from the sign of `xi_bar`).

# Wavelet duration default
`beta=2.0` (with `gamma=3.0`, giving `P=√6≈2.449`) deliberately does NOT
match this file's other functions' shared `beta=8.0` default (`wavetrans`,
`rotary_wavetrans`, `wavelet_significance`, etc. — left untouched here,
scoped change). `(β,γ)=(2,3)` is the actual value used throughout Lilly,
Scott & Olhede (2011) and Lilly & Perez-Brunius (2021) for this exact
eddy-ridge application — confirmed by testing against 28 real GOMED
ridges: `beta=2` measurably tightens ridge-extent agreement over
`beta=8` (mean temporal Jaccard overlap 0.541->0.577, "similar extent"
cases 6/28->10/28) versus GOMED's own ridges, though it does not fully
resolve the remaining scatter on its own (see
project_gomed_validation_results memory).

# References
Lilly & Perez-Brunius (2021), Nonlin. Processes Geophys. 28, 181-212.
Lilly & Olhede (2010b) for the ellipse/rotary parameter relations.
Lilly, Scott & Olhede (2011), Geophys. Res. Lett. 38, L23605, for the
beta=2 gamma=3 wavelet choice (their Sect. 3: "P_psi/pi = sqrt(6)/pi ~=
0.78 in order to obtain a high degree of time concentration").
"""
# jLab-faithful eddy-band grid (eddyridges.m:552-555), shared by both the
# tangent-plane (`rotary_ridge_properties`) and spherical
# (`rotary_ridge_properties_sphere`) entry points -- depends only on record
# length `N` and the wavelet/Coriolis parameters, not on the input signal
# itself. Only kicks in when the caller hasn't already supplied an explicit
# `fs`. See `rotary_ridge_properties`'s docstring "Frequency-grid gap this
# fixes" for why this exists.
function _eddy_band_grid(N::Int, gamma::Real, beta::Real, f_coriolis::Real,
                         fmax_ratio::Real, fmin_ratio::Real,
                         D::Union{Int,Nothing}, eta::Real)
    P, _, _ = morseprops(Float64(gamma), Float64(beta))
    f_high = min(Float64(fmax_ratio) * f_coriolis, _morsehigh(gamma, beta, eta))
    f_low = max(Float64(fmin_ratio) * f_coriolis, π * P / N)
    dens = D === nothing ? 16 : D
    return morsespace(gamma, beta, N; f_high=f_high, f_low=f_low, density=dens)
end

function rotary_ridge_properties(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                 dt::Real=1.0,
                                 fs::Union{AbstractVector, Nothing}=nothing,
                                 nv::Int=8,
                                 gamma::Real=3.0,
                                 beta::Real=2.0,
                                 f_coriolis::Real=1.0,
                                 min_cycles::Real=2*sqrt(beta*gamma)/π,
                                 alpha::Real=0.25,
                                 abs_floor::Real=0.0,
                                 fmax_ratio::Union{Real,Nothing}=nothing,
                                 fmin_ratio::Union{Real,Nothing}=nothing,
                                 D::Union{Int,Nothing}=nothing,
                                 eta::Real=0.05,
                                 boundary::Symbol=:zeros)

    length(u) == length(v) || error("u and v must have the same length")
    abs(f_coriolis) > 0 || error("f_coriolis must be nonzero")

    fs_use = fs
    if fs_use === nothing && fmax_ratio !== nothing && fmin_ratio !== nothing
        fs_use = _eddy_band_grid(length(u), gamma, beta, f_coriolis, fmax_ratio, fmin_ratio, D, eta)
    end

    wt_ccw, wt_cw, fs_out = rotary_wavetrans(u, v; dt=dt, fs=fs_use, nv=nv,
                                             gamma=gamma, beta=beta, boundary=boundary)

    return _rotary_ridge_properties_core(wt_ccw, wt_cw, fs_out;
                                         f_coriolis=f_coriolis, min_cycles=min_cycles,
                                         alpha=alpha, abs_floor=abs_floor)
end

# (lat,lon) -> unit-sphere-normalized (phi,theta) in RADIANS, ported from
# xyz2latlon.m (jSphere/xyz2latlon.m): normalize to the unit sphere first
# (x,y,z need not already be on it -- callers pass a displaced position),
# then phi=asin(z), theta=atan2(y,x). theta doesn't actually need the
# normalization (atan2 is scale-invariant), only phi does.
function _xyz2latlon_rad(x::Real, y::Real, z::Real)
    Rmag = sqrt(x^2 + y^2 + z^2)
    phi = asin(clamp(z / Rmag, -1.0, 1.0))
    theta = atan(y, x)
    return phi, theta
end

"""
    rotary_wavetrans_sphere(lat, lon; dt=1.0, fs=nothing, nv=8, gamma=3.0,
                            beta=8.0, boundary=:zeros, R=6371.0)

jLab-faithful spherical counterpart of [`rotary_wavetrans`](@ref) — ported
directly from `jWavelet/spheretrans.m`/`spheretrans_xyz2xy.m`, to close the
residual ridge-fragmentation gap left by [`local_tangent_plane`](@ref)'s
FIXED local-tangent-plane approximation (mean Jaccard vs. real jLab
0.575->0.957, one remaining partial case; see
project_gomed_validation_results memory).

Where `local_tangent_plane` projects the trajectory to a flat (x,y) plane
ONCE, at a single fixed center, before ever wavelet-transforming it, jLab's
real `eddyridges.m` instead:

1. Converts `(lat,lon)` to the EXACT 3D Cartesian position on a sphere of
   radius `R` (`latlon2xyz.m`: `x=R·cos(lat)cos(lon)`, `y=R·cos(lat)sin(lon)`,
   `z=R·sin(lat)`) — never approximated as flat.
2. Wavelet-transforms EACH of `x`, `y`, `z` independently with the *same*
   wavelet bank (three real-input [`wavetrans`](@ref) calls sharing one
   frequency grid) — not a bivariate transform of a 2D projection.
3. At every `(time, scale)` point, recenters: subtracts the wavelet
   coefficient's real part from the point's own true position
   (`xyz_new = xyz_actual - real(w)`) to get the "time-varying center of
   the oscillation in each wavelet band", converts that back to
   `(lat_new, lon_new)`, and projects `(wx,wy,wz)` onto the LOCAL horizontal
   basis AT THAT RECENTERED POINT (not the trajectory's own true position) —
   `uvw2hor.m`: `uh = wy·cos(θ) - wx·sin(θ)`, `vh = wz/cos(φ)` (exact when
   `(wx,wy,wz)` is tangent to the sphere; see `uvw2hor.m`'s derivation).

This means the tangent-plane orientation itself varies per time step AND
per wavelet scale/band — the "time-varying-center-per-wavelet-band
reprojection" `local_tangent_plane`'s docstring refers to as not yet done.

# Returns
Same shape as [`rotary_wavetrans`](@ref): `(wt_ccw, wt_cw, fs)`, ready to
feed [`ridgemap`](@ref)/[`ridgechains_jlab`](@ref) or
[`rotary_ridge_properties_sphere`](@ref) directly.

# References
jWavelet/spheretrans.m, jSphere/latlon2xyz.m, jSphere/xyz2latlon.m,
jSphere/uvw2hor.m (jLab v1.7.1/v1.7.3, confirmed unchanged between the two).
"""
function rotary_wavetrans_sphere(lat::AbstractVector{<:Real}, lon::AbstractVector{<:Real};
                                 dt::Real=1.0,
                                 fs::Union{AbstractVector, Nothing}=nothing,
                                 nv::Int=8,
                                 gamma::Real=3.0,
                                 beta::Real=8.0,
                                 boundary::Symbol=:zeros,
                                 R::Real=6371.0)
    length(lat) == length(lon) || error("lat and lon must have the same length")
    N = length(lat)
    (N < 1) && error("Input must have at least 1 sample")

    latv = collect(Float64, lat)
    lonv = collect(Float64, lon)

    # latlon2xyz.m: x=R*cosd(lat)*cosd(lon), y=R*cosd(lat)*sind(lon), z=R*sind(lat).
    # (jLab unwraps lon in degrees before this, but cosd/sind are periodic so
    # it's a mathematical no-op for x,y,z -- not ported.)
    x = R .* cosd.(latv) .* cosd.(lonv)
    y = R .* cosd.(latv) .* sind.(lonv)
    z = R .* sind.(latv)

    # Three real-input transforms sharing ONE frequency grid, exactly jLab's
    # [wx,wy,wz]=wavetrans(x,y,z,...) (spheretrans.m:76) — the multi-input
    # form batches all three through a single FFT pass and guarantees the
    # shared grid structurally (no fs= re-passing needed).
    wx, wy, wz, fs_out = wavetrans(x, y, z; dt=dt, fs=fs, nv=nv, gamma=gamma,
                                   beta=beta, boundary=boundary)

    n_scales = length(fs_out)
    wxh = Matrix{ComplexF64}(undef, N, n_scales)
    wyh = Matrix{ComplexF64}(undef, N, n_scales)

    for j in 1:n_scales
        for t in 1:N
            xnew = x[t] - real(wx[t, j])
            ynew = y[t] - real(wy[t, j])
            znew = z[t] - real(wz[t, j])
            phi_new, theta_new = _xyz2latlon_rad(xnew, ynew, znew)
            wxh[t, j] = wy[t, j] * cos(theta_new) - wx[t, j] * sin(theta_new)
            wyh[t, j] = wz[t, j] / cos(phi_new)
        end
    end

    wt_ccw = wxh .+ im .* wyh
    wt_cw = wxh .- im .* wyh
    return wt_ccw, wt_cw, fs_out
end

"""
    rotary_ridge_properties_sphere(lat, lon; dt=1.0, fs=nothing, nv=8,
                                   gamma=3.0, beta=2.0, f_coriolis=1.0,
                                   min_cycles=..., alpha=0.25, abs_floor=0.0,
                                   fmax_ratio=nothing, fmin_ratio=nothing,
                                   D=nothing, eta=0.05, boundary=:zeros, R=6371.0)

Spherical counterpart of [`rotary_ridge_properties`](@ref): identical ridge
detection/chaining/significance-input logic, but the wavelet transform comes
from [`rotary_wavetrans_sphere`](@ref) (jLab's real `spheretrans.m`
time-varying-center reprojection) instead of a fixed
[`local_tangent_plane`](@ref) projection fed into [`rotary_wavetrans`](@ref).
Takes `(lat, lon)` in degrees directly (not pre-projected `(x_km, y_km)`).

All keyword arguments and the return shape (`Vector{NamedTuple}` with
`start`, `stop`, `npoints`, `L`, `xi_bar`, `omega_ast_bar`, `kappa_bar`,
`sense`) match `rotary_ridge_properties` exactly — see its docstring for
the eddy-band-grid (`fmax_ratio`/`fmin_ratio`/`D`/`eta`) and wavelet-duration
(`beta=2.0` default) rationale, both of which apply identically here.
"""
function rotary_ridge_properties_sphere(lat::AbstractVector{<:Real}, lon::AbstractVector{<:Real};
                                        dt::Real=1.0,
                                        fs::Union{AbstractVector, Nothing}=nothing,
                                        nv::Int=8,
                                        gamma::Real=3.0,
                                        beta::Real=2.0,
                                        f_coriolis::Real=1.0,
                                        min_cycles::Real=2*sqrt(beta*gamma)/π,
                                        alpha::Real=0.25,
                                        abs_floor::Real=0.0,
                                        fmax_ratio::Union{Real,Nothing}=nothing,
                                        fmin_ratio::Union{Real,Nothing}=nothing,
                                        D::Union{Int,Nothing}=nothing,
                                        eta::Real=0.05,
                                        boundary::Symbol=:zeros,
                                        R::Real=6371.0)

    length(lat) == length(lon) || error("lat and lon must have the same length")
    abs(f_coriolis) > 0 || error("f_coriolis must be nonzero")

    fs_use = fs
    if fs_use === nothing && fmax_ratio !== nothing && fmin_ratio !== nothing
        fs_use = _eddy_band_grid(length(lat), gamma, beta, f_coriolis, fmax_ratio, fmin_ratio, D, eta)
    end

    wt_ccw, wt_cw, fs_out = rotary_wavetrans_sphere(lat, lon; dt=dt, fs=fs_use, nv=nv,
                                                    gamma=gamma, beta=beta, boundary=boundary, R=R)

    return _rotary_ridge_properties_core(wt_ccw, wt_cw, fs_out;
                                         f_coriolis=f_coriolis, min_cycles=min_cycles,
                                         alpha=alpha, abs_floor=abs_floor)
end

# Shared ridge-detection/chaining core for both rotary_ridge_properties
# (tangent-plane input) and rotary_ridge_properties_sphere (spherical
# input) -- everything downstream of the wavelet transform itself doesn't
# care where wt_ccw/wt_cw came from.
function _rotary_ridge_properties_core(wt_ccw::AbstractMatrix{ComplexF64}, wt_cw::AbstractMatrix{ComplexF64},
                                       fs_out::AbstractVector{Float64};
                                       f_coriolis::Real, min_cycles::Real, alpha::Real, abs_floor::Real)
    a_plus = abs.(wt_ccw)
    a_minus = abs.(wt_cw)
    kappa2 = a_plus .^ 2 .+ a_minus .^ 2
    delta = a_plus .- a_minus          # >0 where CCW (wp) dominates, matches jLab's abs(wp)>abs(wn)
    mag = sqrt.(kappa2)                # joint amplitude kappa=||w(t,s)||, jLab's `rq`/`a` (isridgepoint.m:56,62)

    # jLab-faithful joint instantaneous frequency (instmom(w,1,3) on the
    # Cartesian (wx,wy) pair, ridgewalk.m:349 -> isridgepoint.m:56) --
    # shared by both sides below, since it does not depend on the mask.
    om_joint, dfdt_joint = _instfreq_joint(wt_ccw, wt_cw, 1.0)
    mag_c = complex.(mag)  # ridgechains_jlab only ever reads abs.(w); embedding as complex satisfies its type

    ridges = NamedTuple[]

    for side in (:ccw, :cw)
        # Ridge points are local maxima of the UNMASKED joint amplitude
        # `mag`, exactly like jLab: isridgepoint.m finds local maxima on the
        # full joint quantity first, and only THEN applies the one-sided
        # mask as a post-hoc point filter (`bool=bool.*mask`, isridgepoint.m:127)
        # -- masked-out points are simply ineligible ridge-point candidates,
        # not zeroed out of the amplitude field used to locate maxima. An
        # earlier version zeroed wt_ccw/wt_cw before finding local maxima,
        # which distorts exactly where those maxima sit near every mask
        # boundary and was a real, jLab-crosscheck-confirmed source of
        # ridge fragmentation (see project_gomed_validation_results memory,
        # 2026-08-01) -- fixed by passing `mask` through to
        # ridgechains_jlab instead.
        #
        # Note the exactly-zero case (|w⁺| == |w⁻|, i.e. perfectly linear
        # polarization) is excluded from BOTH sides and so yields no ridge at
        # all. That is deliberate: a purely rectilinear oscillation is not an
        # eddy, and it would in any case score X = L·|ξ̄|^α = 0 in
        # `density_ratio_significance` and be rejected. Assigning it to one
        # side instead would double-count it. In real (noisy) data `delta` is
        # never identically zero, so this only bites on synthetic input.
        mask_side = side === :ccw ? (delta .> 0) : (delta .< 0)

        events = ridgechains_jlab(mag_c, fs_out; alpha=alpha, min_cycles=min_cycles, dt=1.0,
                                  abs_floor=abs_floor, om=om_joint, dfdt=dfdt_joint, mask=mask_side)

        for e in events
            times = collect(e.times)
            n = length(times)

            # Interpolate a_plus^2/a_minus^2 to the same sub-bin scale
            # location (jfrac) the frequency was refined to, via the same
            # quadinterp fit ridgechains_jlab used for amplitude/frequency
            # (ridgeinterp.m's "interpolate any other ridge quantity the
            # same way" -- here applied to kappa/xi rather than the raw
            # transform, since that's what downstream needs).
            ap2 = Vector{Float64}(undef, n)
            am2 = Vector{Float64}(undef, n)
            for k in 1:n
                t, j, je = times[k], e.jcenter[k], e.jfrac[k]
                if je == j
                    ap2[k] = a_plus[t, j]^2
                    am2[k] = a_minus[t, j]^2
                else
                    ap2[k] = quadinterp(j - 1, j, j + 1, a_plus[t, j-1]^2, a_plus[t, j]^2, a_plus[t, j+1]^2, je)
                    am2[k] = quadinterp(j - 1, j, j + 1, a_minus[t, j-1]^2, a_minus[t, j]^2, a_minus[t, j+1]^2, je)
                end
            end
            k2 = ap2 .+ am2
            sk2 = sum(k2)
            sk2 > 0 || continue

            xi = (ap2 .- am2) ./ k2
            omega = e.freq                              # rad/sample, jLab-style phase-derived
            omega_ast = sign.(xi) .* omega ./ f_coriolis

            # Eq. 60: ridge length in cycles (one sample per time step).
            L = sum(abs.(omega)) / (2π)
            L > min_cycles || continue

            # Eq. 74: kappa^2-weighted ridge averages.
            xi_bar = sum(k2 .* xi) / sk2
            omega_ast_bar = sum(k2 .* omega_ast) / sk2
            kappa_bar = sqrt(sk2 / n)

            push!(ridges, (start=times[1], stop=times[end], npoints=n,
                           L=L, xi_bar=xi_bar, omega_ast_bar=omega_ast_bar,
                           kappa_bar=kappa_bar,
                           sense=(xi_bar >= 0 ? :ccw : :cw)))
        end
    end

    sort!(ridges; by=r -> r.start)
    return ridges
end

# ============================================================================
# DENSITY-RATIO SIGNIFICANCE CRITERION
# (Lilly & Perez-Brunius 2021, Sect. 4.6, Eqs. 76-80)
# ============================================================================

"""
    density_ratio_significance(data_ridges, noise_ridges, n_data_points, n_noise_points;
                               alpha=4, delta_omega=0.25, n_bands=1000,
                               n_bins=2000, x_max=20.0)

Compute the density-ratio significance level `ρ_X` of each ridge, following
Lilly & Perez-Brunius (2021) Sect. 4.6.

Rather than thresholding on a physical property directly (radius, velocity,
Rossby number — all of which are what we actually want to *study*, so
cutting on them biases the result), significance is assessed on the combined
parameter

    X = L · |ξ̄|^alpha

which rewards ridges that are both long-lived (`L`) and close to circularly
polarized (`|ξ̄|`) — the two things coherent vortices are and background
turbulence is not. `alpha = 4` is the paper's own choice, found by sweeping
the exponent to maximize the number of oscillations judged significant.

For each ridge, the frequency-local **survival functions** (fraction of
ridge points, per data point, having a significance parameter greater than
this ridge's `X`) are formed for the real data and for the noise, in
overlapping bands of nondimensional frequency of width `delta_omega`. Their
ratio

    ρ_X = S^ε_X / S_X

estimates the probability that a detection with these properties is a false
positive: `ρ_X = 0.1` means such events occur only a tenth as often in the
noise as in the data, so 1 in 10 is attributable to the null hypothesis.
The paper accepts ridges with `ρ_X < 0.1`.

Note this is a **global, two-pass** statistic: the survival functions pool
ridges across the entire dataset, so significance cannot be decided one
record at a time. Both `data_ridges` and `noise_ridges` must be the full
pooled collections, and `n_data_points`/`n_noise_points` the corresponding
total sample counts (the per-data-point normalization of their Eq. 77 is
what makes two differently-sized collections comparable).

# Arguments
- `data_ridges`, `noise_ridges`: collections of NamedTuples as returned by
  [`rotary_ridge_properties`](@ref) (need `L`, `xi_bar`, `omega_ast_bar`,
  `npoints`)
- `n_data_points`, `n_noise_points`: total number of time samples analyzed
  in each collection (summed over all records, and over all noise
  realizations for the noise)
- `alpha`: exponent on `|ξ̄|` in the significance parameter
- `delta_omega`: width of the nondimensional-frequency bands
- `n_bands`, `n_bins`: resolution of the frequency / significance-parameter grids
- `x_max`: upper edge of the `X` grid (values above are clamped into the top bin)

# Returns
`Vector{Float64}` of `ρ_X`, one per entry of `data_ridges`. Ridges whose
data survival function is empty return `Inf` (rejected), never `NaN`.

# References
Lilly & Perez-Brunius (2021), Sect. 4.6, Eqs. 76-80.
"""
function density_ratio_significance(data_ridges, noise_ridges,
                                    n_data_points::Real, n_noise_points::Real;
                                    alpha::Real=4,
                                    delta_omega::Real=0.25,
                                    n_bands::Int=1000,
                                    n_bins::Int=2000,
                                    x_max::Real=20.0)

    isempty(data_ridges) && return Float64[]
    n_data_points > 0 || error("n_data_points must be positive")
    n_noise_points > 0 || error("n_noise_points must be positive")

    _X(r) = r.L * abs(r.xi_bar)^alpha

    Xd = [_X(r) for r in data_ridges]
    wd = [Float64(r.npoints) for r in data_ridges]
    od = [r.omega_ast_bar for r in data_ridges]

    Xn = isempty(noise_ridges) ? Float64[] : [_X(r) for r in noise_ridges]
    wn = isempty(noise_ridges) ? Float64[] : [Float64(r.npoints) for r in noise_ridges]
    on = isempty(noise_ridges) ? Float64[] : [r.omega_ast_bar for r in noise_ridges]

    omega_lo = minimum(vcat(od, isempty(on) ? od : on)) - delta_omega
    omega_hi = maximum(vcat(od, isempty(on) ? od : on)) + delta_omega
    band_centers = range(omega_lo, omega_hi; length=n_bands)
    band_step = n_bands > 1 ? step(band_centers) : one(Float64)

    # Histogram of ridge POINTS (not ridge counts) over (frequency band, X bin),
    # normalized per data point -- Eq. 77's ridge density.
    function _accumulate(X, w, o, npts)
        H = zeros(Float64, n_bands, n_bins)
        isempty(X) && return H
        for i in eachindex(X)
            bin = clamp(1 + floor(Int, (X[i] / x_max) * n_bins), 1, n_bins)
            # Bands are overlapping: a ridge contributes to every band whose
            # center lies within delta_omega/2 of its frequency.
            lo = max(1, 1 + ceil(Int, (o[i] - delta_omega/2 - omega_lo) / band_step))
            hi = min(n_bands, 1 + floor(Int, (o[i] + delta_omega/2 - omega_lo) / band_step))
            for b in lo:hi
                H[b, bin] += w[i]
            end
        end
        return H ./ npts
    end

    Hd = _accumulate(Xd, wd, od, n_data_points)
    Hn = _accumulate(Xn, wn, on, n_noise_points)

    # Eq. 79: survival function = reverse cumulative sum along X.
    Sd = cumsum(Hd[:, end:-1:1]; dims=2)[:, end:-1:1]
    Sn = cumsum(Hn[:, end:-1:1]; dims=2)[:, end:-1:1]

    rho = Vector{Float64}(undef, length(data_ridges))
    for i in eachindex(Xd)
        bin = clamp(1 + floor(Int, (Xd[i] / x_max) * n_bins), 1, n_bins)
        b = clamp(1 + round(Int, (od[i] - omega_lo) / band_step), 1, n_bands)
        sd = Sd[b, bin]
        # A data ridge always contributes to its own bin, so sd > 0 in
        # practice; guard anyway and reject rather than emit NaN, which
        # would compare false against any threshold and read as "significant".
        rho[i] = sd > 0 ? Sn[b, bin] / sd : Inf
    end
    return rho
end

# ============================================================================
# JLAB-FAITHFUL RIDGE CHAINING
# (ported from jLab v1.7.1 jRidges/{quadinterp,instmom,ridgechains}.m,
#  the exact version that generated GOMED -- confirmed algorithmically
#  unchanged through jLab's current v1.7.3, 2026-08-01)
# ============================================================================

"""
    quadinterp(t1, t2, t3, x1, x2, x3, t=nothing)

Closed-form quadratic (parabola) interpolation/extrapolation through three
exact points `(t1,x1), (t2,x2), (t3,x3)`.

Solves `x = a*t^2 + b*t + c` for `(a,b,c)` via the exact Lagrange-style
algebraic solution (no linear system, no loops — vectorizes elementwise
over arrays), then either evaluates the fitted parabola at `t` or, if `t`
is omitted, locates its vertex `t = -b/2a`.

# Returns
- `t !== nothing`: `x` — the fitted parabola evaluated at `t`
- `t === nothing`: `(xe, te)` — the vertex value and its location

# References
Ported from jLab's `quadinterp.m` (Lilly, jLab v1.7.1/v1.7.3, identical).
"""
function quadinterp(t1, t2, t3, x1, x2, x3, t=nothing)
    denom = @. (t1 - t2) * (t1 - t3) * (t2 - t3)
    a = @. (x1 * (t2 - t3) + x2 * (t3 - t1) + x3 * (t1 - t2)) / denom
    b = @. -(x1 * (t2^2 - t3^2) + x2 * (t3^2 - t1^2) + x3 * (t1^2 - t2^2)) / denom
    c = @. (x1 * t2 * t3 * (t2 - t3) + x2 * t3 * t1 * (t3 - t1) + x3 * t1 * t2 * (t1 - t2)) / denom

    if t === nothing
        te = @. -b / (2 * a)
        xe = @. a * te^2 + b * te + c
        return xe, te
    else
        return @. a * t^2 + b * t + c
    end
end

# --- Stage B: instantaneous frequency via unwrapped-phase differencing ----

function _unwrap(p::AbstractVector{<:Real})
    n = length(p)
    out = similar(p, Float64)
    n == 0 && return out
    out[1] = p[1]
    correction = 0.0
    @inbounds for i in 2:n
        d = p[i] - p[i-1]
        if d > π
            correction -= 2π
        elseif d < -π
            correction += 2π
        end
        out[i] = p[i] + correction
    end
    return out
end

# jLab's vdiff(...,'endpoint'): central difference in the interior,
# one-sided forward/backward difference at the first/last sample.
function _vdiff_endpoint(x::AbstractVector{<:Real}, dt::Real)
    n = length(x)
    d = similar(x, Float64)
    n == 0 && return d
    n == 1 && (d[1] = 0.0; return d)
    d[1] = (x[2] - x[1]) / dt
    d[n] = (x[n] - x[n-1]) / dt
    @inbounds for i in 2:n-1
        d[i] = (x[i+1] - x[i-1]) / (2dt)
    end
    return d
end

# Instantaneous frequency (Eq. from instmom.m:184) and its own time
# derivative, evaluated on the FULL (t,s) transform matrix -- not just at
# already-found ridge points -- one column (scale) at a time since phase
# unwrapping only makes sense along the time axis at fixed scale.
function _instfreq_matrix(w::AbstractMatrix{<:Complex}, dt::Real)
    nt, ns = size(w)
    om = Matrix{Float64}(undef, nt, ns)
    for j in 1:ns
        om[:, j] = _vdiff_endpoint(_unwrap(angle.(@view(w[:, j]))), dt)
    end
    dfdt = Matrix{Float64}(undef, nt, ns)
    for j in 1:ns
        dfdt[:, j] = _vdiff_endpoint(@view(om[:, j]), dt)
    end
    return om, dfdt
end

# --- Stage C: mutual-nearest-neighbor ridge chaining ----------------------

struct _RidgePt
    j::Int
    om::Float64
    dfdt::Float64
end

"""
    ridgechains_jlab(w, fs; alpha=0.25, min_cycles=2*sqrt(6)/pi, dt=1.0, abs_floor=0.0,
                     om=nothing, dfdt=nothing, mask=nothing)

One-branch wavelet ridge chaining faithfully following jLab v1.7.1/v1.7.3's
`ridgechains.m`/`isridgepoint.m`/`ridgeinterp.m`/`quadinterp.m` (the exact,
algorithmically-unchanged version that produced GOMED) -- as opposed to
[`ridgechains`](@ref)'s independent greedy-forward implementation.

Differs from `ridgechains` in two ways, both ported directly rather than
approximated:

1. **Frequency estimate**: each ridge point's frequency comes from
   [`_instfreq_matrix`](@ref) (finite-differenced unwrapped phase of `w`
   itself, evaluated on the full matrix), not the nominal analysis-grid
   value `fs[j]`. A second difference gives a LOCAL prediction
   `fr_next=om+dfdt`, `fr_prev=om-dfdt` at every point, independent of any
   chain's own history (`ridgechains.m:52-53`).
2. **Matching**: two candidate points at consecutive time steps are linked
   only if each is the OTHER's best match under the averaged forward/
   backward relative frequency error (`ridgechains.m:73-105`) -- a mutual-
   nearest-neighbor bipartite match, not a greedy forward search extending
   whichever chain got there first. The relative error's denominator is
   `max(|ω|, abs_floor)` rather than jLab's plain `|ω|`: for slow eddies
   (`|ω|` near zero), the plain relative criterion makes ordinary
   finite-difference frequency noise look like a huge fractional jump,
   fragmenting one true ridge into many short spurious pieces. `abs_floor`
   defaults to `0.0` (exactly jLab's own, unfloored criterion).

Ridge points additionally get their scale index refined to sub-bin
precision via [`quadinterp`](@ref) fit through the amplitude² at the three
scales around each point (`ridgeinterp.m:76-92`), with frequency
re-evaluated at that refined location.

Excludes the first/last scale bins from ever being ridge points
(`isridgepoint.m:99`, `bool(:,[1 end])=0`), unlike `ridgechains`.

# Returns
`Vector{NamedTuple}`, one per surviving ridge, with fields `times`
(`UnitRange` into the time axis), `freq` (interpolated instantaneous
frequency along the ridge, rad/sample), `jfrac` (fractional scale index
along the ridge, for looking up other quantities via [`quadinterp`](@ref)).

# References
Ported from jLab v1.7.1 (confirmed algorithmically identical through
v1.7.3, 2026-08-01) `jRidges/ridgechains.m`, `isridgepoint.m`,
`ridgeinterp.m`, `quadinterp.m`. Lilly, jLab, https://github.com/jonathanlilly/jLab.
"""
function ridgechains_jlab(w::AbstractMatrix{<:Complex}, fs::AbstractVector;
                          alpha::Real=0.25, min_cycles::Real=2*sqrt(6)/π,
                          dt::Real=1.0, abs_floor::Real=0.0,
                          om::Union{Nothing,AbstractMatrix}=nothing,
                          dfdt::Union{Nothing,AbstractMatrix}=nothing,
                          mask::Union{Nothing,AbstractMatrix{Bool}}=nothing)
    nt, ns = size(w)
    ns >= 3 || return NamedTuple[]
    amp = abs.(w)
    # Experimental override: jLab's own eddyridges.m actually derives the
    # instantaneous frequency from a joint POWER-WEIGHTED AVERAGE of the raw
    # Cartesian (wx,wy) channels' own phase derivatives (instmom's
    # multivariate path), not from a single rotary branch's own phase --
    # a simplification flagged when this function was first written. Pass
    # precomputed (om,dfdt) to test that more faithful estimate without
    # duplicating the point-finding/chaining/interpolation logic below.
    if om === nothing
        om, dfdt = _instfreq_matrix(w, dt)
    end

    # Ridge points: local maxima of amplitude along scale, excluding the
    # first/last scale bin (isridgepoint.m:99), matching jLab exactly. When
    # `mask` is given, it is applied HERE -- as a filter on which local
    # maxima are eligible ridge-point candidates -- exactly matching
    # isridgepoint.m:127 (`bool=bool.*mask`), which zeroes the point-eligibility
    # array AFTER local maxima are found on the full (unmasked) amplitude
    # field, not before. Critically, `amp` itself must therefore be the
    # UNMASKED joint amplitude (the caller no longer zeroes the transform
    # before calling this) -- masking the amplitude field itself would
    # distort where the true local maxima sit, which is exactly the bug
    # this parameter replaces (see rotary_ridge_properties).
    pts = Vector{Vector{_RidgePt}}(undef, nt)
    for i in 1:nt
        row = @view amp[i, :]
        p = _RidgePt[]
        for j in 2:ns-1
            if row[j] >= row[j-1] && row[j] >= row[j+1] && row[j] > 0 &&
               (mask === nothing || mask[i, j])
                push!(p, _RidgePt(j, om[i, j], dfdt[i, j]))
            end
        end
        pts[i] = p
    end

    # Mutual-nearest-neighbor matching between consecutive time steps.
    # link[i][k] = index into pts[i+1] that point k of pts[i] links to, or 0.
    link = [zeros(Int, length(pts[i])) for i in 1:nt]
    for i in 1:nt-1
        a, b = pts[i], pts[i+1]
        (isempty(a) || isempty(b)) && continue
        cost = fill(Inf, length(a), length(b))
        for (ka, pa) in enumerate(a), (kb, pb) in enumerate(b)
            pred_next = pa.om + pa.dfdt         # forward prediction from a[ka]
            pred_prev = pb.om - pb.dfdt         # backward prediction from b[kb]
            df1 = abs(pred_next - pb.om) / (max(abs(pa.om), abs_floor) + 1e-20)
            df2 = abs(pred_prev - pa.om) / (max(abs(pb.om), abs_floor) + 1e-20)
            d = (df1 + df2) / 2
            d <= alpha && (cost[ka, kb] = d)
        end
        for ka in 1:length(a)
            row = @view cost[ka, :]
            all(isinf, row) && continue
            kb = argmin(row)
            # Mutual best: kb's own best match over column kb must be ka too.
            col = @view cost[:, kb]
            argmin(col) == ka || continue
            # jLab's own second, separate mask application (ridgechains.m:138-176,
            # "Keep from chaining ridges through mask"): even though both
            # endpoints individually passed the point-level mask in
            # isridgepoint.m, the LINK connecting them is only valid if mask
            # holds across every scale index it implicitly spans, evaluated
            # at the earlier time step (`jjindex=min(jj,jjnext):max(jj,jjnext);
            # bool=all(mask(ii,jjindex))`) -- prevents a chain from jumping
            # clean over a masked (wrong-rotation-sense) region between two
            # otherwise-valid points.
            if mask !== nothing
                j1, j2 = a[ka].j, b[kb].j
                all(@view mask[i, min(j1, j2):max(j1, j2)]) || continue
            end
            link[i][ka] = kb
        end
    end

    # Walk links into chains of (time, scale-index) points.
    visited = [falses(length(pts[i])) for i in 1:nt]
    chains = Vector{Vector{Tuple{Int,Int}}}()
    for i in 1:nt, k in 1:length(pts[i])
        visited[i][k] && continue
        chain = Tuple{Int,Int}[(i, k)]
        visited[i][k] = true
        ci, ck = i, k
        while ci < nt && link[ci][ck] != 0
            nk = link[ci][ck]
            ci += 1
            visited[ci][nk] && break
            push!(chain, (ci, nk))
            visited[ci][nk] = true
            ck = nk
        end
        push!(chains, chain)
    end

    # Drop isolated single-point ridges (ridgechains.m:134-136).
    filter!(c -> length(c) > 1, chains)

    ridges = NamedTuple[]
    for chain in chains
        # chain entries are (time_index, position-within-pts[time_index]) --
        # NOT (time_index, scale_index) -- convert to the real scale index
        # via pts[.].j before using it to index amp/om (a bug caught here by
        # a BoundsError during verification: a position index of 1 was used
        # directly as a scale bin, so amp[t, j-1] underflowed to index 0).
        times = [c[1] for c in chain]
        js = [pts[c[1]][c[2]].j for c in chain]
        (times[end] - times[1] + 1 == length(times)) || continue  # must be contiguous in time

        freq = Vector{Float64}(undef, length(chain))
        jfrac = Vector{Float64}(undef, length(chain))
        jcenter = Vector{Int}(undef, length(chain))
        for n in 1:length(chain)
            t, j = times[n], js[n]
            jcenter[n] = j
            # Sub-bin scale refinement (ridgeinterp.m:76-92): fit a parabola
            # through amplitude^2 at the 3 scales around this point.
            xe, je = quadinterp(j - 1, j, j + 1, amp[t, j-1]^2, amp[t, j]^2, amp[t, j+1]^2)
            if !(j - 1 < je < j + 1) || !isfinite(je)
                je = Float64(j)  # quadratic fit failed -- fall back to the integer bin
            end
            jfrac[n] = je
            # Re-evaluate the instantaneous frequency at the refined location
            # (ridgeinterp.m:154), same three-point parabola through om.
            freq[n] = quadinterp(j - 1, j, j + 1, om[t, j-1], om[t, j], om[t, j+1], je)
        end

        L = sum(abs.(freq)) / (2π)
        L > min_cycles || continue

        push!(ridges, (times=times[1]:times[end], freq=freq, jfrac=jfrac, jcenter=jcenter))
    end

    sort!(ridges; by=r -> r.times.start)
    return ridges
end

# Experimental: joint (Cartesian wx,wy) power-weighted instantaneous
# frequency, matching jLab's actual instmom(w,1,3)/jointmom formula, as an
# alternative to a single rotary branch's own phase (the simplification
# ridgechains_jlab's om/dfdt use by default). w_plus/w_minus are the rotary
# transforms already computed by rotary_wavetrans; reconstructs wx,wy from
# them algebraically (no extra wavetrans call needed).
function _instfreq_joint(w_plus::AbstractMatrix{<:Complex}, w_minus::AbstractMatrix{<:Complex}, dt::Real)
    wx = (w_plus .+ w_minus) ./ sqrt(2)
    wy = .-im .* (w_plus .- w_minus) ./ sqrt(2)
    nt, ns = size(wx)
    om_x = Matrix{Float64}(undef, nt, ns)
    om_y = Matrix{Float64}(undef, nt, ns)
    for j in 1:ns
        om_x[:, j] = _vdiff_endpoint(_unwrap(angle.(@view(wx[:, j]))), dt)
        om_y[:, j] = _vdiff_endpoint(_unwrap(angle.(@view(wy[:, j]))), dt)
    end
    wx2 = abs2.(wx)
    wy2 = abs2.(wy)
    om = (wx2 .* om_x .+ wy2 .* om_y) ./ (wx2 .+ wy2 .+ 1e-300)
    dfdt = Matrix{Float64}(undef, nt, ns)
    for j in 1:ns
        dfdt[:, j] = _vdiff_endpoint(@view(om[:, j]), dt)
    end
    return om, dfdt
end

# ============================================================================
# MULTIVARIATE INSTANTANEOUS MOMENTS & WAVELET RIDGE ANALYSIS
# (Lilly & Olhede 2012, "Analysis of Modulated Multivariate Oscillations",
# IEEE Trans. Signal Process. 60(2), 600-612; jRidges/instmom.m, ridgewalk.m,
# isridgepoint.m)
#
# Unlike the rotary (CCW/CW) path above -- which is specific to the N=2
# elliptical/rotary special case and does NOT generalize past N=2 -- this
# section ports jLab's genuinely N-general joint-moment machinery: a
# power-weighted average across N raw channels, with no cross-channel
# covariance/tensor terms. Confirmed by reading jRidges/instmom.m's nested
# `jointmom` and jRidges/isridgepoint.m directly (2026-08-03): jLab's own
# real-data examples never exercise N>2, but the code and the underlying
# theory (Lilly & Olhede 2012) are honestly N-agnostic, so this is a real
# port, not a renamed bivariate special case.
# ============================================================================

"""
    instmom(w::AbstractMatrix{<:Complex}; dt=1.0)

Univariate instantaneous moments of a wavelet transform (or any per-column
analytic signal), ported from jLab's `instmom.m`.

# Returns
`(a, om, upsilon, xi)`:
- `a`: instantaneous amplitude, `abs.(w)`
- `om`: instantaneous radian frequency, `d/dt arg(w)`
- `upsilon`: instantaneous bandwidth, `d/dt ln|w|`
- `xi`: instantaneous curvature (complex), `upsilon^2 + d/dt(upsilon) + i*d/dt(om)`

`om`/`upsilon`/`xi` are identical to `tiredecode(w,fs;kind="freq"/"bandwidth")`
computed with the `dt`-aware `_vdiff_endpoint` kernel; `instmom` bundles all
four moments in one call since the joint (multivariate) formulas below need
all of them at once.

# References
Lilly & Olhede (2010), IEEE Trans. Signal Process. 58(2), 591-603 (a, om, upsilon);
Lilly & Olhede (2012), IEEE Trans. Signal Process. 60(2), 600-612 (xi, the
*instantaneous curvature*). jRidges/instmom.m.
"""
function instmom(w::AbstractMatrix{<:Complex}; dt::Real=1.0)
    nt, ns = size(w)
    a = abs.(w)
    om = Matrix{Float64}(undef, nt, ns)
    upsilon = Matrix{Float64}(undef, nt, ns)
    xi = Matrix{ComplexF64}(undef, nt, ns)
    for j in 1:ns
        om[:, j] = _vdiff_endpoint(_unwrap(angle.(@view(w[:, j]))), dt)
        upsilon[:, j] = _vdiff_endpoint(log.(max.(@view(a[:, j]), 1e-300)), dt)
        dupsilon = _vdiff_endpoint(@view(upsilon[:, j]), dt)
        domega = _vdiff_endpoint(@view(om[:, j]), dt)
        xi[:, j] = upsilon[:, j] .^ 2 .+ dupsilon .+ im .* domega
    end
    return a, om, upsilon, xi
end

"""
    instmom(W::AbstractArray{<:Complex,3}; dt=1.0)

Joint instantaneous moments of N channels stacked along the trailing
dimension (jLab's `[JA,JOMEGA,JUPSILON,JXI]=INSTMOM(X1,...,XN,DIM)`), i.e.
the multivariate generalization of [`instmom`](@ref)`(w)` above. Computes
each channel's own univariate moments first, then combines them via jLab's
`jointmom` formula -- a power-weighted average across channels, with no
cross-channel covariance terms:

- `ja = sqrt(mean_n |w_n|^2)` (RMS joint amplitude -- `instmom`'s own
  convention; note [`multivariate_ridges`](@ref) instead uses the Euclidean
  *sum* convention `sqrt(Σ_n|w_n|^2)` for ridge detection, matching
  `isridgepoint.m`; the two differ by a factor of `sqrt(N)`, a
  normalization-convention difference between jLab's own two use sites,
  not a discrepancy in this port)
- `jomega = Σ_n(a_n^2*om_n) / Σ_n(a_n^2)` (Lilly & Olhede 2012, Eq. 13)
- `jupsilon = sqrt(Σ_n(a_n^2*|upsilon_n + i(om_n-jomega)|^2) / Σ_n(a_n^2))`
- `jxi = sqrt(Σ_n(a_n^2*|xi_n + 2i*upsilon_n*(om_n-jomega) - (om_n-jomega)^2|^2) / Σ_n(a_n^2))`

# Returns
`(ja, jomega, jupsilon, jxi)`, each `(nt, ns)` -- one entry per (time,scale),
collapsed across the channel dimension.

# References
Lilly & Olhede (2012), IEEE Trans. Signal Process. 60(2), 600-612, Eq. 13
and Sect. III.D (deviation vectors). jRidges/instmom.m's nested `jointmom`.
"""
function instmom(W::AbstractArray{<:Complex,3}; dt::Real=1.0)
    nt, ns, n_ch = size(W)
    a = Array{Float64,3}(undef, nt, ns, n_ch)
    om = Array{Float64,3}(undef, nt, ns, n_ch)
    upsilon = Array{Float64,3}(undef, nt, ns, n_ch)
    xi = Array{ComplexF64,3}(undef, nt, ns, n_ch)
    for c in 1:n_ch
        a[:, :, c], om[:, :, c], upsilon[:, :, c], xi[:, :, c] = instmom(@view(W[:, :, c]); dt=dt)
    end

    a2 = a .^ 2
    sa2 = dropdims(sum(a2; dims=3); dims=3) .+ 1e-300

    ja = sqrt.(dropdims(sum(a2; dims=3); dims=3) ./ n_ch)
    jomega = dropdims(sum(a2 .* om; dims=3); dims=3) ./ sa2

    jomega_rep = reshape(jomega, nt, ns, 1)
    dev = upsilon .+ im .* (om .- jomega_rep)
    jupsilon = sqrt.(dropdims(sum(a2 .* abs2.(dev); dims=3); dims=3) ./ sa2)

    dom = om .- jomega_rep
    dev2 = xi .+ (2 .* im) .* upsilon .* dom .- dom .^ 2
    jxi = sqrt.(dropdims(sum(a2 .* abs2.(dev2); dims=3); dims=3) ./ sa2)

    return ja, jomega, jupsilon, jxi
end

# General N-channel generalization of _instfreq_joint's power-weighted joint
# instantaneous frequency (jLab's instmom(w,1,3)/jointmom formula), applied
# directly to N raw wavetrans channels rather than via the 2-channel rotary
# w+/w- reconstruction _instfreq_joint uses (which does not generalize past
# N=2). On the N=2 case (wx,wy stacked along dim 3), this agrees with
# _instfreq_joint(w_plus,w_minus,dt) to floating-point precision (tested).
function _instfreq_joint_nd(W::AbstractArray{<:Complex,3}, dt::Real)
    nt, ns, n_ch = size(W)
    om_ch = Array{Float64,3}(undef, nt, ns, n_ch)
    amp2_ch = Array{Float64,3}(undef, nt, ns, n_ch)
    for c in 1:n_ch, j in 1:ns
        om_ch[:, j, c] = _vdiff_endpoint(_unwrap(angle.(@view(W[:, j, c]))), dt)
        amp2_ch[:, j, c] = abs2.(@view(W[:, j, c]))
    end
    num = dropdims(sum(amp2_ch .* om_ch; dims=3); dims=3)
    den = dropdims(sum(amp2_ch; dims=3); dims=3) .+ 1e-300
    om = num ./ den
    dfdt = Matrix{Float64}(undef, nt, ns)
    for j in 1:ns
        dfdt[:, j] = _vdiff_endpoint(@view(om[:, j]), dt)
    end
    return om, dfdt
end

"""
    multivariate_ridges(X::AbstractMatrix{<:Real}; dt=1.0, fs=nothing, nv=8,
                        gamma=3.0, beta=2.0, boundary=:zeros,
                        min_cycles=2*sqrt(beta*gamma)/pi, alpha=0.25, abs_floor=0.0)
    multivariate_ridges(x1, x2, xs...; kwargs...)

Multivariate wavelet ridge analysis (Lilly & Olhede 2012): extracts a
modulated oscillation common to `N >= 2` simultaneously observed real
channels (columns of `X`, or passed as separate vectors). Generalizes
[`rotary_ridge_properties`](@ref) beyond its `N=2` rotary/elliptical special
case to arbitrary `N`, using jLab's genuinely N-general joint-amplitude/
frequency machinery (power-weighted averages and Euclidean vector norms
across channels -- no cross-channel covariance/tensor terms; see
[`instmom`](@ref) for the same idea applied to instantaneous moments
directly).

Unlike the rotary path, there is no notion of rotation "sense" for `N > 2`
independent channels, so no `xi`/`sense` fields are returned -- only the
joint amplitude/frequency and the ridge-based signal estimate itself.

# Returns
`Vector{NamedTuple}`, one per ridge, sorted by start time, with fields:
- `start`, `stop`, `npoints`: ridge extent (time-index bookkeeping)
- `L`: ridge length in cycles (`Σ|omega|/(2π)`)
- `omega_bar`: kappa²-weighted mean joint instantaneous frequency (rad/sample)
- `kappa_bar`: kappa²-weighted RMS joint amplitude (Euclidean-norm convention,
  `sqrt(Σ_n|w_n|^2)`, matching `isridgepoint.m` -- NOT `instmom`'s RMS
  convention, which differs by `sqrt(N)`; see [`instmom`](@ref)'s docstring)
- `wt_ridge::Matrix{ComplexF64}` (`npoints × n_channels`): the wavelet
  transform vector along the ridge -- the ridge-based estimate of the
  analytic joint oscillation (paper Eq. 40, `x̂₊(t) = w_x,ψ(t, ŝ(t))`)

# References
Lilly, J. M. & Olhede, S. C. (2012). Analysis of modulated multivariate
oscillations. IEEE Trans. Signal Process., 60(2), 600-612.
jRidges/ridgewalk.m, isridgepoint.m, instmom.m.
"""
function multivariate_ridges(X::AbstractMatrix{<:Real};
                             dt::Real=1.0,
                             fs::Union{AbstractVector, Nothing}=nothing,
                             nv::Int=8,
                             gamma::Real=3.0,
                             beta::Real=2.0,
                             boundary::Symbol=:zeros,
                             min_cycles::Real=2 * sqrt(beta * gamma) / π,
                             alpha::Real=0.25,
                             abs_floor::Real=0.0)
    n_ch = size(X, 2)
    n_ch >= 2 || error("multivariate_ridges requires at least 2 channels (columns of X)")

    wt3, fs_out = wavetrans(X; dt=dt, fs=fs, nv=nv, gamma=gamma, beta=beta, boundary=boundary)

    mag = sqrt.(dropdims(sum(abs2, wt3; dims=3); dims=3))   # joint amplitude, Euclidean-norm convention (isridgepoint.m)
    om_joint, dfdt_joint = _instfreq_joint_nd(wt3, 1.0)
    mag_c = complex.(mag)  # ridgechains_jlab only ever reads abs.(w); embedding as complex satisfies its type

    events = ridgechains_jlab(mag_c, fs_out; alpha=alpha, min_cycles=min_cycles, dt=1.0,
                              abs_floor=abs_floor, om=om_joint, dfdt=dfdt_joint)

    ridges = NamedTuple[]
    for e in events
        times = collect(e.times)
        n = length(times)

        # Sub-bin interpolation of the raw complex transform itself (real/imag
        # parts separately), via the same quadinterp fit ridgechains_jlab used
        # for amplitude/frequency (ridgeinterp.m's "interpolate any other
        # ridge quantity the same way") -- here applied directly to the
        # transform, since wt_ridge (the ridge-based signal estimate, Eq. 40)
        # IS the deliverable of multivariate ridge analysis.
        wt_ridge = Matrix{ComplexF64}(undef, n, n_ch)
        k2 = Vector{Float64}(undef, n)
        for k in 1:n
            t, j, je = times[k], e.jcenter[k], e.jfrac[k]
            for c in 1:n_ch
                if je == j
                    wt_ridge[k, c] = wt3[t, j, c]
                else
                    re = quadinterp(j - 1, j, j + 1, real(wt3[t, j-1, c]), real(wt3[t, j, c]), real(wt3[t, j+1, c]), je)
                    imv = quadinterp(j - 1, j, j + 1, imag(wt3[t, j-1, c]), imag(wt3[t, j, c]), imag(wt3[t, j+1, c]), je)
                    wt_ridge[k, c] = complex(re, imv)
                end
            end
            k2[k] = sum(abs2, @view(wt_ridge[k, :]))
        end
        sk2 = sum(k2)
        sk2 > 0 || continue

        omega = e.freq   # rad/sample, jLab-style phase-derived (ridgechains_jlab)
        L = sum(abs.(omega)) / (2π)
        L > min_cycles || continue

        omega_bar = sum(k2 .* omega) / sk2
        kappa_bar = sqrt(sk2 / n)

        push!(ridges, (start=times[1], stop=times[end], npoints=n,
                       L=L, omega_bar=omega_bar, kappa_bar=kappa_bar,
                       wt_ridge=wt_ridge))
    end

    sort!(ridges; by=r -> r.start)
    return ridges
end

function multivariate_ridges(x1::AbstractVector{<:Real}, x2::AbstractVector{<:Real},
                             xs::AbstractVector{<:Real}...; kwargs...)
    multivariate_ridges(hcat(x1, x2, xs...); kwargs...)
end

# ============================================================================
# TYPED ENTRY POINTS — TimeSeries* containers -> Types.WaveletTransform
# (These live in src, not an extension: the wavelet machinery has no
# weak dependency, unlike the Multitaper-gated spectral overloads.)
# ============================================================================

# Shared body for the Vector/Matrix container methods: derive dt from the
# time axis (hard error on irregular sampling, via
# Types._dt_hours_from_time), strip units on the way in per the house
# convention, record everything needed for reinterpretation in params.
function _wavetrans_typed(value::AbstractVecOrMat, time::Vector{Dates.DateTime},
                          channels::Vector{String}, name::String; kwargs...)
    isempty(value) && error("wavetrans: empty time series")
    haskey(kwargs, :dt) &&
        error("wavetrans(::TimeSeries...): dt is derived from the time axis; do not pass dt=")
    dt_hours = Types._dt_hours_from_time(time)
    u = Unitful.unit(first(value))
    x = Unitful.ustrip.(value)
    wt, fs_v = wavetrans(x; dt=dt_hours, kwargs...)
    params = (; dt_hours=dt_hours, unit=u, name=name, kwargs...)
    return Types.WaveletTransform(wt, fs_v, copy(time), channels, params)
end

"""
    wavetrans(ts::TimeSeriesVector; kwargs...)
    wavetrans(ts::TimeSeriesMatrix; kwargs...)
    wavetrans(tc::TimeSeriesCollection; kwargs...)

Typed overloads of [`wavetrans`](@ref): the sampling interval is derived
from the container's time axis (in hours; irregular sampling errors
clearly rather than silently averaging a `dt` — do not also pass `dt=`),
values are stripped of their Unitful unit on the way in, and the result is
a [`Types.WaveletTransform`](@ref) carrying the time axis, channel names
(for `TimeSeriesMatrix`), and the original unit in `params.unit`
(re-applied by `tiredecode(W; kind="amp")`).

A `TimeSeriesCollection` (ragged records, no shared clock) cannot share
one batched FFT, so it returns a `Vector{WaveletTransform}`, one per
series — each on its own default frequency grid unless an explicit `fs=`
is given (mirroring the batch-spectral `Vector{<:SpectralEstimate}`
convention).
"""
wavetrans(ts::Types.TimeSeriesVector; kwargs...) =
    _wavetrans_typed(ts.value, ts.time, String[], ts.name; kwargs...)

wavetrans(ts::Types.TimeSeriesMatrix; kwargs...) =
    _wavetrans_typed(ts.value, ts.time, copy(ts.channels), ts.name; kwargs...)

wavetrans(tc::Types.TimeSeriesCollection; kwargs...) =
    [wavetrans(s; kwargs...) for s in tc.series]

"""
    tiredecode(W::Types.WaveletTransform; kind="amp")

Typed overload of [`tiredecode`](@ref): decodes `W.wt` against `W.fs`.
For `kind="amp"` the source series' unit (`W.params.unit`, recorded by the
typed [`wavetrans`](@ref) methods) is re-applied to the amplitude; the
other kinds (`"phase"`, `"freq"`, `"bandwidth"`) are returned as bare
`Float64` (radians / rad-per-sample quantities, unit-independent).
"""
function tiredecode(W::Types.WaveletTransform; kind::String="amp")
    out = tiredecode(W.wt, W.fs; kind=kind)
    if kind == "amp" && haskey(W.params, :unit)
        return out .* W.params.unit
    end
    return out
end
