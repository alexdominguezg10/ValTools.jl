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
    wavetrans(x::AbstractVector; dt=1.0, fs=nothing, nv=8,
              mother="morse", gamma=3.0, beta=8.0, gpu=false)

Continuous wavelet transform using generalized Morse wavelets,
calibrated to match jLab's wavetrans.m.

Key differences from a naive CWT:
- Morse wavelet is bandpass-normalized (peak of ψ̂ = 1)
- Real-valued input is multiplied by 2 (one-sided analytic wavelet)
- Frequencies auto-generated via morsespace (log-spaced, proper cutoffs)
- Linear detrend applied by default

# Arguments
- `x`: Input time series
- `dt`: Sampling interval (default 1.0)
- `fs`: Analysis frequencies in rad/sample (auto via morsespace if nothing)
- `nv`: Voices per octave (used only if fs=nothing; mapped to density)
- `mother`: "morse" (default)
- `gamma`, `beta`: Morse wavelet parameters (default 3.0, 8.0)
- `gpu`: Use GPU (default false)

# Returns
- `wt::Matrix{ComplexF64}`: Wavelet coefficients `(N, n_freqs)`
- `fs::Vector{Float64}`: Analysis frequencies (rad/sample)

# References
Lilly & Olhede (2009); jWavelet/wavetrans.m
"""
function wavetrans(x::AbstractVector;
                   dt::Real=1.0,
                   fs::Union{AbstractVector, Nothing}=nothing,
                   scales::Union{AbstractVector, Nothing}=nothing,
                   nv::Int=8,
                   mother::String="morse",
                   gamma::Real=3.0,
                   beta::Real=8.0,
                   gpu::Bool=false)

    xv = collect(Float64, x)
    N  = length(xv)
    (N < 1) && error("Input must have at least 1 sample")

    g = Float64(gamma)
    b = Float64(beta)

    # Linear detrend (jLab default)
    xv = _detrend_signal(xv)

    # Pad to next power of 2
    N_fft = 2^ceil(Int, log2(2*N - 1))

    # Generate analysis frequencies
    if fs === nothing && scales === nothing
        P, _, _ = morseprops(g, b)
        density = max(1, round(Int, nv * P / 4))
        fs = morsespace(g, b, N; density=density)
    elseif scales !== nothing
        # Backward compatibility: convert scales to radian frequencies
        fs = 2π .* collect(Float64, scales) .* dt
    end
    fs = collect(Float64, fs)

    # Build filter bank (calibrated Morse wavelet)
    bank = _build_filter_bank(N_fft, fs, g, b)

    # Pad signal
    x_pad = vcat(xv, zeros(N_fft - N))

    # Dispatch
    if gpu
        wt = _wavetrans_gpu(x_pad, bank, N, N_fft, length(fs))
    else
        wt = _wavetrans_cpu(x_pad, bank, N, N_fft, length(fs))
    end

    # Factor of 2 for real-valued input (jLab convention)
    if eltype(x) <: Real
        wt .*= 2
    end

    return wt, fs
end

function _detrend_signal(x::Vector{Float64})
    n = length(x)
    n < 2 && return x
    t = collect(1.0:n)
    A = [ones(n) t]
    coeffs = A \ x
    return x .- A * coeffs
end

# ── CPU path ────────────────────────────────────────────────────────────────

function _wavetrans_cpu(x_pad::Vector{<:Number}, bank::Matrix{Float64},
                        N::Int, N_fft::Int, n_scales::Int)
    X  = fft(x_pad)
    wt = zeros(ComplexF64, N, n_scales)

    for j in 1:n_scales
        Y = X .* conj.(@view(bank[:, j]))      # conj(ψ̂) · X̂ (jLab convention)
        wt[:, j] .= ifft(Y)[1:N]
    end
    return wt
end

# ── GPU path (stub — implemented by ValToolsCUDAExt) ────────────────────────

function _wavetrans_gpu(x_pad, bank, N, N_fft, n_scales)
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
- `u`, `v`: east-west / north-south velocity components (equal length)
- `dt`, `fs`, `nv`, `gamma`, `beta`: as in [`wavetrans`](@ref)

# Returns
- `wt_ccw::Matrix{ComplexF64}`: CCW-rotating wavelet coefficients `(N, n_freqs)`
- `wt_cw::Matrix{ComplexF64}`: CW-rotating wavelet coefficients `(N, n_freqs)`
- `fs::Vector{Float64}`: analysis frequencies (rad/sample), shared by both

Both `wt_ccw` and `wt_cw` are ordinary `wavetrans`-shaped outputs, so
[`ridgemap`](@ref), [`ridgechains`](@ref), and [`tiredecode`](@ref) all work
on them directly — or use [`rotary_ridge`](@ref) for a one-call ridge
summary of both components.

# References
Lilly, J. M. & Olhede, S. C. (2010). Bivariate instantaneous frequency and
bandwidth. IEEE Trans. Signal Process., 58(2), 591–603.
"""
function rotary_wavetrans(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                          dt::Real=1.0,
                          fs::Union{AbstractVector, Nothing}=nothing,
                          nv::Int=8,
                          gamma::Real=3.0,
                          beta::Real=8.0)
    length(u) == length(v) || error("u and v must have the same length")
    N = length(u)
    (N < 1) && error("Input must have at least 1 sample")

    g, b = Float64(gamma), Float64(beta)
    uf = _detrend_signal(collect(Float64, u))
    vf = _detrend_signal(collect(Float64, v))

    N_fft = 2^ceil(Int, log2(2 * N - 1))

    if fs === nothing
        P, _, _ = morseprops(g, b)
        density = max(1, round(Int, nv * P / 4))
        fs = morsespace(g, b, N; density=density)
    end
    fs = collect(Float64, fs)

    bank = _build_filter_bank(N_fft, fs, g, b)

    w = uf .+ im .* vf
    wc = conj.(w)
    w_pad = vcat(w, zeros(ComplexF64, N_fft - N))
    wc_pad = vcat(wc, zeros(ComplexF64, N_fft - N))

    wt_ccw = _wavetrans_cpu(w_pad, bank, N, N_fft, length(fs))
    wt_cw = _wavetrans_cpu(wc_pad, bank, N, N_fft, length(fs))

    return wt_ccw, wt_cw, fs
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
                    gamma=3.0, beta=8.0, gpu=false)

Wavelet transform of multiple signals simultaneously.
Each column of `X` is an independent time series.

# Returns
- `wt::Array{ComplexF64, 3}`: `(N, n_freqs, n_signals)`
- `fs::Vector{Float64}`
"""
function wavetrans_batch(X::AbstractMatrix;
                         dt::Real=1.0,
                         fs::Union{AbstractVector, Nothing}=nothing,
                         scales::Union{AbstractVector, Nothing}=nothing,
                         nv::Int=8,
                         gamma::Real=3.0,
                         beta::Real=8.0,
                         gpu::Bool=false)

    N, n_sig = size(X)
    N_fft = 2^ceil(Int, log2(2*N - 1))
    g = Float64(gamma); b = Float64(beta)

    if fs === nothing && scales === nothing
        P, _, _ = morseprops(g, b)
        density = max(1, round(Int, nv * P / 4))
        fs = morsespace(g, b, N; density=density)
    elseif scales !== nothing
        fs = 2π .* collect(Float64, scales) .* dt
    end
    fs = collect(Float64, fs)
    n_fs = length(fs)

    bank = _build_filter_bank(N_fft, fs, g, b)

    # Detrend all signals before dispatch (CPU and GPU get same input)
    X_det = collect(Float64, X)
    for s in 1:n_sig
        X_det[:, s] = _detrend_signal(X_det[:, s])
    end

    if gpu
        wt = _wavetrans_batch_gpu(X_det, bank, N, N_fft, n_fs, n_sig)
    else
        wt = _wavetrans_batch_cpu(X_det, bank, N, N_fft, n_fs, n_sig)
    end

    # Factor of 2 for real input
    if eltype(X) <: Real
        wt .*= 2
    end

    return wt, fs
end

function _wavetrans_batch_cpu(X::Matrix{Float64}, bank::Matrix{Float64},
                              N::Int, N_fft::Int, n_scales::Int, n_sig::Int)
    wt = zeros(ComplexF64, N, n_scales, n_sig)
    for s in 1:n_sig
        x_pad = vcat(X[:, s], zeros(N_fft - N))
        Xf = fft(x_pad)
        for j in 1:n_scales
            Y = Xf .* conj.(@view(bank[:, j]))
            wt[:, j, s] .= ifft(Y)[1:N]
        end
    end
    return wt
end

function _wavetrans_batch_gpu(X, bank, N, N_fft, n_scales, n_sig)
    error("""GPU batch wavelet transform requires CUDA.jl.
    Load it first:  `using CUDA`""")
end

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
    error("Unknown kind: $kind (use \"amp\", \"phase\", \"freq\", \"bandwidth\")")
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
                        rng=Random.default_rng())

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

    max_power = Matrix{Float64}(undef, n_surrogates, n_freq)
    for s in 1:n_surrogates
        surrogate = background == :white ? mu .+ sigma .* randn(rng, N) :
                                            _ar1_surrogate(N, alpha, mu, sigma, rng)
        wt_s, _ = wavetrans(surrogate; dt=dt, fs=fs_ref, nv=nv, gamma=gamma, beta=beta)
        max_power[s, :] = vec(maximum(abs2.(wt_s); dims=1))
    end

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
                            f_coriolis=1.0, min_cycles=2*sqrt(beta*gamma)/pi, alpha=0.25)

Extract **one-sided** wavelet ridges from a bivariate velocity series and
summarize each one by its ellipse properties, following Lilly &
Perez-Brunius (2021, NPG).

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
- `u`, `v`: east-west / north-south velocity components (equal length)
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
function rotary_ridge_properties(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                 dt::Real=1.0,
                                 fs::Union{AbstractVector, Nothing}=nothing,
                                 nv::Int=8,
                                 gamma::Real=3.0,
                                 beta::Real=2.0,
                                 f_coriolis::Real=1.0,
                                 min_cycles::Real=2*sqrt(beta*gamma)/π,
                                 alpha::Real=0.25)

    length(u) == length(v) || error("u and v must have the same length")
    abs(f_coriolis) > 0 || error("f_coriolis must be nonzero")

    wt_ccw, wt_cw, fs_out = rotary_wavetrans(u, v; dt=dt, fs=fs, nv=nv,
                                             gamma=gamma, beta=beta)

    a_plus = abs.(wt_ccw)
    a_minus = abs.(wt_cw)
    kappa2 = a_plus .^ 2 .+ a_minus .^ 2
    delta = a_plus .- a_minus          # >0 where CCW dominates
    mag = sqrt.(kappa2)                # ||w(t,s)||, the total transform magnitude

    ridges = NamedTuple[]

    for side in (:ccw, :cw)
        # Mask the opposite-rotation half-plane, then chain ridges on what
        # remains. Zeroed entries can never be ridge points (ridgechains
        # requires amplitude strictly above its threshold, default 0).
        # Note the exactly-zero case (|w⁺| == |w⁻|, i.e. perfectly linear
        # polarization) is excluded from BOTH sides and so yields no ridge at
        # all. That is deliberate: a purely rectilinear oscillation is not an
        # eddy, and it would in any case score X = L·|ξ̄|^α = 0 in
        # `density_ratio_significance` and be rejected. Assigning it to one
        # side instead would double-count it. In real (noisy) data `delta` is
        # never identically zero, so this only bites on synthetic input.
        masked = copy(mag)
        if side === :ccw
            masked[delta .<= 0] .= 0.0
        else
            masked[delta .>= 0] .= 0.0
        end

        events = ridgechains(complex.(masked), fs_out; alpha=alpha, min_length=1.0)

        for e in events
            times = e.start:e.stop
            js = e.scale_idx
            length(js) == length(times) || continue   # defensive: chains are contiguous

            k2 = [kappa2[t, j] for (t, j) in zip(times, js)]
            ap = [a_plus[t, j] for (t, j) in zip(times, js)]
            am = [a_minus[t, j] for (t, j) in zip(times, js)]
            sk2 = sum(k2)
            sk2 > 0 || continue

            xi = (ap .^ 2 .- am .^ 2) ./ k2
            omega = e.freq                              # rad/sample
            omega_ast = sign.(xi) .* omega ./ f_coriolis

            # Eq. 60: ridge length in cycles (one sample per time step).
            L = sum(abs.(omega)) / (2π)
            L > min_cycles || continue

            # Eq. 74: kappa^2-weighted ridge averages.
            xi_bar = sum(k2 .* xi) / sk2
            omega_ast_bar = sum(k2 .* omega_ast) / sk2
            kappa_bar = sqrt(sk2 / length(k2))

            push!(ridges, (start=e.start, stop=e.stop, npoints=length(times),
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
