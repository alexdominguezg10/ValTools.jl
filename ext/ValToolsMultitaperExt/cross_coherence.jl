# Met.cross_coherence — Multitaper.jl-backed implementation.
# The non-rotary counterpart of rotary_coherence (rotary_spectrum.jl in this
# same extension): real signals, one-sided FFT, no CW/CCW split. Uses
# Met._detrend_linear / Met.fftfreq (shared helpers that stay in the main
# Metrics module since non-Multitaper functions there need them too).

function Met.cross_coherence(x::AbstractVector{<:Real}, y::AbstractVector{<:Real};
                             dt::Real=1.0,
                             detrend::String="linear",
                             nw::Real=4.0,
                             ntapers::Int=0,
                             confidence::Real=0.95)
    n = length(x)
    length(y) == n || error("x and y must have the same length")
    any(!isfinite, x) && error("x must not contain NaN/Inf")
    any(!isfinite, y) && error("y must not contain NaN/Inf")

    function _prep(v)
        vf = Float64.(v)
        if detrend == "linear"
            return Met._detrend_linear(vf)
        elseif detrend == "constant"
            return vf .- mean(vf)
        elseif detrend == "none"
            return vf
        else
            error("detrend must be \"none\", \"constant\", or \"linear\"")
        end
    end
    xf, yf = _prep(x), _prep(y)

    K = ntapers > 0 ? ntapers : max(1, 2 * floor(Int, nw) - 1)
    tapers, _lambdas = dpss_tapers(n, Float64(nw), K, :both)

    freqs_all = Met.fftfreq(n, dt)
    pos_mask = freqs_all .> 0
    freqs_pos = freqs_all[pos_mask]
    nfreq = length(freqs_pos)

    Cxy_k = Matrix{ComplexF64}(undef, nfreq, K)
    Pxx_k = Matrix{Float64}(undef, nfreq, K)
    Pyy_k = Matrix{Float64}(undef, nfreq, K)

    for k in 1:K
        taper = @view tapers[:, k]
        Xk = fft(xf .* taper)[pos_mask]
        Yk = fft(yf .* taper)[pos_mask]
        Cxy_k[:, k] = Xk .* conj.(Yk)
        Pxx_k[:, k] = abs.(Xk) .^ 2
        Pyy_k[:, k] = abs.(Yk) .^ 2
    end

    Cxy = vec(mean(Cxy_k; dims=2))
    Pxx = vec(mean(Pxx_k; dims=2))
    Pyy = vec(mean(Pyy_k; dims=2))

    coherence = (abs.(Cxy) .^ 2) ./ (Pxx .* Pyy)
    phase = angle.(Cxy)

    significance_level = K > 1 ? 1 - (1 - confidence)^(1 / (K - 1)) : NaN

    params = (nw=Float64(nw), ntapers=K, dt=Float64(dt),
              detrend=detrend, confidence=Float64(confidence), N=n)

    return ValTools.Types.CrossSpectralEstimate(freqs_pos, Cxy, coherence, phase,
                                                significance_level, params)
end

"""
    cross_coherence(x::Types.TimeSeriesVector, y::Types.TimeSeriesVector; kwargs...)

Typed overload of [`cross_coherence`](@ref): `dt` (in hours) is derived
from `x.time` instead of being a required keyword. `x` and `y` must share
the same time axis and be regularly sampled. Unlike the
`rotary_spectrum`/`ellipse_polarization` typed overloads, `x` and `y` are
stripped of units independently (not converted to a common one) —
coherence/phase are unit-agnostic ratios and `x`/`y` need not even be the
same physical quantity (e.g. wind stress vs. a current component).

No same-signature stub exists in the main package for this method (see
the NOTE in `Metrics/spectral.jl`) — without `using Multitaper`, calling
this with `TimeSeriesVector` arguments raises a plain `MethodError`.
"""
function Met.cross_coherence(x::ValTools.Types.TimeSeriesVector, y::ValTools.Types.TimeSeriesVector; kwargs...)
    x.time == y.time || error("cross_coherence: x and y must share the same time axis")
    dt = _dt_hours_from_time(x.time)
    xv = Unitful.ustrip.(x.value)
    yv = Unitful.ustrip.(y.value)
    return Met.cross_coherence(xv, yv; dt=dt, kwargs...)
end
