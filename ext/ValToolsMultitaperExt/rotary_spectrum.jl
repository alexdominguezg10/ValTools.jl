# Met.rotary_spectrum / rotary_coherence — Multitaper.jl-backed implementations.
# Moved verbatim from src/Metrics/spectral.jl when Multitaper.jl became a
# weakdep; see the stub docstrings there for the public API documentation.
# Uses Met._detrend_linear / Met.fftfreq (shared helpers that stay in the
# main Metrics module since non-Multitaper functions there need them too).

function Met.rotary_spectrum(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                         dt_hours::Real=1.0,
                         detrend::String="linear",
                         nw::Real=4.0,
                         ntapers::Int=0,
                         ci::Bool=true,
                         confidence::Real=0.95,
                         ftest::Bool=true)
    length(u) == length(v) || error("u and v must have the same length")
    any(!isfinite, u) && error("u must not contain NaN/Inf")
    any(!isfinite, v) && error("v must not contain NaN/Inf")

    uf = Float64.(u)
    vf = Float64.(v)
    n = length(uf)

    if detrend == "linear"
        uf = Met._detrend_linear(uf)
        vf = Met._detrend_linear(vf)
    elseif detrend == "constant"
        uf = uf .- mean(uf)
        vf = vf .- mean(vf)
    elseif detrend != "none"
        error("detrend must be \"none\", \"constant\", or \"linear\"")
    end

    K = ntapers > 0 ? ntapers : max(1, 2 * floor(Int, nw) - 1)
    tapers, _lambdas = dpss_tapers(n, Float64(nw), K, :both)

    freqs_all = Met.fftfreq(n, dt_hours)
    pos_mask = freqs_all .> 0
    neg_mask = freqs_all .< 0
    freqs_pos = freqs_all[pos_mask]
    freqs_neg_raw = -freqs_all[neg_mask]
    order = sortperm(freqs_neg_raw)
    freqs_neg = freqs_neg_raw[order]

    S_ccw_k = Matrix{Float64}(undef, length(freqs_pos), K)
    S_cw_k = Matrix{Float64}(undef, length(freqs_pos), K)
    W_ccw_k = Matrix{ComplexF64}(undef, length(freqs_pos), K)
    W_cw_native_k = Matrix{ComplexF64}(undef, length(freqs_neg), K)
    Hk0 = Vector{Float64}(undef, K)

    for k in 1:K
        taper = @view tapers[:, k]
        Hk0[k] = sum(taper)
        w_k = (uf .* taper) .+ im .* (vf .* taper)
        W_k = fft(w_k)
        psd_k = (abs.(W_k) .^ 2) .* dt_hours ./ n

        S_ccw_k[:, k] = psd_k[pos_mask]
        W_ccw_k[:, k] = W_k[pos_mask]
        S_cw_raw = psd_k[neg_mask][order]
        S_cw_k[:, k] = _interp1(freqs_neg, S_cw_raw, freqs_pos)
        W_cw_native_k[:, k] = W_k[neg_mask][order]
    end

    S_ccw = vec(mean(S_ccw_k; dims=2))
    S_cw = vec(mean(S_cw_k; dims=2))

    ci_ccw = nothing
    ci_cw = nothing
    if ci && K > 1
        ci_ccw = _jackknife_ci(S_ccw_k, confidence)
        ci_cw = _jackknife_ci(S_cw_k, confidence)
    end

    ftest_ccw = nothing
    ftest_cw = nothing
    if ftest && K > 1
        Fstat_ccw = _harmonic_fstat(W_ccw_k, Hk0)
        ftest_ccw = _harmonic_fpval.(Fstat_ccw, K)

        Fstat_cw_native = _harmonic_fstat(W_cw_native_k, Hk0)
        Fstat_cw = _interp1(freqs_neg, Fstat_cw_native, freqs_pos)
        ftest_cw = _harmonic_fpval.(Fstat_cw, K)
    end

    rotary_coefficient = (S_ccw .- S_cw) ./ (S_ccw .+ S_cw)

    params = (nw=Float64(nw), ntapers=K, dt_hours=Float64(dt_hours),
              detrend=detrend, confidence=Float64(confidence), N=n)

    return ValTools.Types.RotarySpectralEstimate(freqs_pos, S_ccw, S_cw, ci_ccw, ci_cw,
                                        rotary_coefficient, ftest_ccw, ftest_cw, params)
end

# The `_dt_hours_from_time` helper the typed methods below use was hoisted
# to Types (src/Types/types_def.jl) when JLab's typed wavetrans methods
# became a second consumer -- imported at the top of this extension.

"""
    rotary_spectrum(u::Types.TimeSeriesVector, v::Types.TimeSeriesVector; kwargs...)

Typed overload of [`rotary_spectrum`](@ref): `dt_hours` is derived from
`u.time` instead of being a required keyword. `u` and `v` must share the
same time axis (`u.time == v.time`) and be regularly sampled — multitaper
spectral estimation assumes uniform sampling, so irregular timestamps
error clearly (via [`_dt_hours_from_time`](@ref)) rather than silently
averaging a `dt`. `v` is converted to `u`'s unit before both are stripped
(same auto-convert convention as `Dispatch._stripped_common_unit`). All
other keyword arguments (`detrend`, `nw`, `ntapers`, `ci`, `confidence`,
`ftest`) pass straight through to the untyped method.

No same-signature stub exists in the main package for this method (see
the NOTE in `Metrics/spectral.jl`) — without `using Multitaper`, calling
this with `TimeSeriesVector` arguments raises a plain `MethodError`.
"""
function Met.rotary_spectrum(u::ValTools.Types.TimeSeriesVector, v::ValTools.Types.TimeSeriesVector; kwargs...)
    u.time == v.time || error("rotary_spectrum: u and v must share the same time axis")
    dt_hours = _dt_hours_from_time(u.time)
    uv = Unitful.ustrip.(u.value)
    common_unit = Unitful.unit(first(u.value))
    vv = Unitful.ustrip.(Unitful.uconvert.(common_unit, v.value))
    return Met.rotary_spectrum(uv, vv; dt_hours=dt_hours, kwargs...)
end

# Thomson (1982) harmonic F-test for a line component, applied to the
# per-taper complex eigencoefficients `W_k` (nfreq x K) of a single rotary
# branch. `Hk0[k] = sum(taper_k)` is the k-th DPSS taper's zero-frequency
# response, shared between the CCW and CW branches (same taper set).
function _harmonic_fstat(W_k::AbstractMatrix{<:Complex}, Hk0::AbstractVector{<:Real})
    nfreq, K = size(W_k)
    sum_Hk0_sq = sum(abs2, Hk0)
    Fstat = Vector{Float64}(undef, nfreq)
    for i in 1:nfreq
        Chat = sum(Hk0[k] * W_k[i, k] for k in 1:K) / sum_Hk0_sq
        resid = sum(abs2(W_k[i, k] - Chat * Hk0[k]) for k in 1:K)
        Fstat[i] = resid > 0 ? (K - 1) * abs2(Chat) * sum_Hk0_sq / resid : Inf
    end
    return Fstat
end

# p-value for F ~ F(2, 2K-2) via the closed form for a Beta(1, K-1) CDF
# (avoids a Distributions.jl dependency): if X ~ F(2, ν) then
# 2X/(2X+ν) ~ Beta(1, ν/2), whose CDF is 1-(1-y)^(ν/2) for y in [0,1].
function _harmonic_fpval(F::Real, K::Int)
    isinf(F) && return 0.0
    return (1 + F / (K - 1))^(-(K - 1))
end

function _jackknife_ci(S_k::AbstractMatrix{<:Real}, confidence::Real)
    nfreq, K = size(S_k)
    logS_delete = Matrix{Float64}(undef, nfreq, K)
    for k in 1:K
        idx = [j for j in 1:K if j != k]
        logS_delete[:, k] = log.(mean(S_k[:, idx]; dims=2))
    end

    mbar = vec(mean(logS_delete; dims=2))
    jkvar = ((K - 1) / K) .* vec(sum((logS_delete .- mbar) .^ 2; dims=2))

    alpha = 1 - confidence
    zval = sqrt(2) * erfinv(2 * (1 - alpha / 2) - 1)
    half_width = zval .* sqrt.(jkvar)

    lower = exp.(mbar .- half_width)
    upper = exp.(mbar .+ half_width)
    return (lower, upper)
end

function Met.rotary_coherence(u1::AbstractVector{<:Real}, v1::AbstractVector{<:Real},
                          u2::AbstractVector{<:Real}, v2::AbstractVector{<:Real};
                          dt_hours::Real=1.0,
                          detrend::String="linear",
                          nw::Real=4.0,
                          ntapers::Int=0,
                          confidence::Real=0.95)
    n = length(u1)
    length(v1) == n && length(u2) == n && length(v2) == n ||
        error("u1, v1, u2, v2 must all have the same length")
    for x in (u1, v1, u2, v2)
        any(!isfinite, x) && error("inputs must not contain NaN/Inf")
    end

    function _prep(x)
        xf = Float64.(x)
        if detrend == "linear"
            return Met._detrend_linear(xf)
        elseif detrend == "constant"
            return xf .- mean(xf)
        elseif detrend == "none"
            return xf
        else
            error("detrend must be \"none\", \"constant\", or \"linear\"")
        end
    end
    u1f, v1f, u2f, v2f = _prep(u1), _prep(v1), _prep(u2), _prep(v2)

    K = ntapers > 0 ? ntapers : max(1, 2 * floor(Int, nw) - 1)
    tapers, _lambdas = dpss_tapers(n, Float64(nw), K, :both)

    freqs_all = Met.fftfreq(n, dt_hours)
    pos_mask = freqs_all .> 0
    neg_mask = freqs_all .< 0
    freqs_pos = freqs_all[pos_mask]
    freqs_neg_raw = -freqs_all[neg_mask]
    order = sortperm(freqs_neg_raw)
    freqs_neg = freqs_neg_raw[order]

    nfreq = length(freqs_pos)
    C_ccw_k = Matrix{ComplexF64}(undef, nfreq, K)
    C_cw_k = Matrix{ComplexF64}(undef, nfreq, K)
    P1_ccw_k = Matrix{Float64}(undef, nfreq, K)
    P2_ccw_k = Matrix{Float64}(undef, nfreq, K)
    P1_cw_k = Matrix{Float64}(undef, nfreq, K)
    P2_cw_k = Matrix{Float64}(undef, nfreq, K)

    for k in 1:K
        taper = @view tapers[:, k]
        w1_k = fft((u1f .* taper) .+ im .* (v1f .* taper))
        w2_k = fft((u2f .* taper) .+ im .* (v2f .* taper))

        W1_ccw = w1_k[pos_mask]
        W2_ccw = w2_k[pos_mask]
        C_ccw_k[:, k] = W1_ccw .* conj.(W2_ccw)
        P1_ccw_k[:, k] = abs.(W1_ccw) .^ 2
        P2_ccw_k[:, k] = abs.(W2_ccw) .^ 2

        W1_cw_raw = w1_k[neg_mask][order]
        W2_cw_raw = w2_k[neg_mask][order]
        C_cw_raw = W1_cw_raw .* conj.(W2_cw_raw)
        C_cw_k[:, k] = _interp1_complex(freqs_neg, C_cw_raw, freqs_pos)
        P1_cw_k[:, k] = _interp1(freqs_neg, abs.(W1_cw_raw) .^ 2, freqs_pos)
        P2_cw_k[:, k] = _interp1(freqs_neg, abs.(W2_cw_raw) .^ 2, freqs_pos)
    end

    C_ccw = vec(mean(C_ccw_k; dims=2))
    P1_ccw = vec(mean(P1_ccw_k; dims=2))
    P2_ccw = vec(mean(P2_ccw_k; dims=2))
    coh_ccw = (abs.(C_ccw) .^ 2) ./ (P1_ccw .* P2_ccw)
    phase_ccw = angle.(C_ccw)

    C_cw = vec(mean(C_cw_k; dims=2))
    P1_cw = vec(mean(P1_cw_k; dims=2))
    P2_cw = vec(mean(P2_cw_k; dims=2))
    coh_cw = (abs.(C_cw) .^ 2) ./ (P1_cw .* P2_cw)
    phase_cw = angle.(C_cw)

    significance_level = K > 1 ? 1 - (1 - confidence)^(1 / (K - 1)) : NaN

    params = (nw=Float64(nw), ntapers=K, dt_hours=Float64(dt_hours),
              detrend=detrend, confidence=Float64(confidence), N=n)

    return ValTools.Types.RotaryCoherenceEstimate(freqs_pos, coh_ccw, coh_cw, phase_ccw, phase_cw,
                                         significance_level, params)
end

function _interp1(xp, yp, xi)
    n = length(xp)
    out = similar(xi)
    for (i, x) in enumerate(xi)
        if x <= xp[1]
            out[i] = yp[1]
        elseif x >= xp[end]
            out[i] = yp[end]
        else
            j = searchsortedlast(xp, x)
            j = clamp(j, 1, n - 1)
            t = (x - xp[j]) / (xp[j+1] - xp[j])
            out[i] = yp[j] + t * (yp[j+1] - yp[j])
        end
    end
    return out
end

function _interp1_complex(xp, yp::AbstractVector{<:Complex}, xi)
    re = _interp1(xp, real.(yp), xi)
    im_ = _interp1(xp, imag.(yp), xi)
    return re .+ im .* im_
end
