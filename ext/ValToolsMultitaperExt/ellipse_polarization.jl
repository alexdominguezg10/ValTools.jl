# Met.ellipse_polarization — Multitaper.jl-backed implementation.
# Port of jLab's jSpectral/polparams.m + specdiag.m (Lilly), applied to the
# same per-taper spectral-matrix machinery as cross_coherence.jl (this same
# extension). Uses Met._detrend_linear / Met.fftfreq (shared helpers that
# stay in the main Metrics module) and _jackknife_ci (defined in
# rotary_spectrum.jl, same module — reused here for d1/d2 CIs).
#
# NOTE on terminology (verified by hand-tracing the formulas against known
# cases, not just porting symbols): `P` ("total polarization") is 1 for
# BOTH a purely rectilinear (back-and-forth on a line) signal AND a purely
# circular single-sense rotary signal — both are "fully polarized" states
# in the optics-polarization sense this is borrowed from. `P` is near 0
# only for genuinely isotropic 2-D noise (u, v independent, equal
# variance). Linear vs. circular is distinguished by `alpha` (real,
# Cartesian anisotropy) vs. `imag(beta)` (rotary sense) — NOT by `P`
# itself. Do not confuse this with "circular" meaning "unpolarized."

function Met.ellipse_polarization(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                  dt_hours::Real=1.0,
                                  detrend::String="linear",
                                  nw::Real=4.0,
                                  ntapers::Int=0,
                                  ci::Bool=true,
                                  confidence::Real=0.95)
    n = length(u)
    length(v) == n || error("u and v must have the same length")
    any(!isfinite, u) && error("u must not contain NaN/Inf")
    any(!isfinite, v) && error("v must not contain NaN/Inf")

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
    uf, vf = _prep(u), _prep(v)

    K = ntapers > 0 ? ntapers : max(1, 2 * floor(Int, nw) - 1)
    tapers, _lambdas = dpss_tapers(n, Float64(nw), K, :both)

    freqs_all = Met.fftfreq(n, dt_hours)
    pos_mask = freqs_all .> 0
    freqs_pos = freqs_all[pos_mask]
    nfreq = length(freqs_pos)

    Sxx_k = Matrix{Float64}(undef, nfreq, K)
    Syy_k = Matrix{Float64}(undef, nfreq, K)
    Sxy_k = Matrix{ComplexF64}(undef, nfreq, K)

    # `.* dt_hours` matches jLab's own mspec.m normalization exactly
    # (jSpectral/mspec.m: `cellout{2}=avgspec(mmatx,mmatx,N).*dt;`, confirmed
    # by reading the real source) -- without it, d1/d2 (the spectral-matrix
    # eigenvalues, i.e. major/minor-axis POWER in absolute physical units)
    # are off by a factor of 1/dt_hours from jLab's convention. theta/nu/P/
    # alpha/beta are unaffected (dt cancels in those ratios), which is why
    # this was invisible to every existing test: they all use the default
    # dt_hours=1.0, where 1/dt_hours=1. Found 2026-08-06 by real-jLab
    # crosscheck on dt=0.5h ARE mooring data
    # (scripts/jlab_crosscheck_ellipse_polarization.{jl,m}), where it showed
    # up as a clean, constant d1/d2 ratio of ~2.0 = 1/0.5.
    for k in 1:K
        taper = @view tapers[:, k]
        Uk = fft(uf .* taper)[pos_mask]
        Vk = fft(vf .* taper)[pos_mask]
        Sxx_k[:, k] = abs.(Uk) .^ 2 .* dt_hours
        Syy_k[:, k] = abs.(Vk) .^ 2 .* dt_hours
        Sxy_k[:, k] = Uk .* conj.(Vk) .* dt_hours
    end

    Sxx = vec(mean(Sxx_k; dims=2))
    Syy = vec(mean(Syy_k; dims=2))
    Sxy = vec(mean(Sxy_k; dims=2))

    d1, d2, theta, nu, P, alpha, beta = _ellipse_from_spectral_matrix(Sxx, Syy, Sxy)

    ci_d1 = nothing
    ci_d2 = nothing
    ci_P = nothing
    ci_theta = nothing
    if ci && K > 1
        d1_del = Matrix{Float64}(undef, nfreq, K)
        d2_del = Matrix{Float64}(undef, nfreq, K)
        P_del = Matrix{Float64}(undef, nfreq, K)
        theta_del = Matrix{Float64}(undef, nfreq, K)
        for k in 1:K
            idx = [j for j in 1:K if j != k]
            Sxx_del = vec(mean(view(Sxx_k, :, idx); dims=2))
            Syy_del = vec(mean(view(Syy_k, :, idx); dims=2))
            Sxy_del = vec(mean(view(Sxy_k, :, idx); dims=2))
            d1_k, d2_k, theta_k, _, P_k, _, _ = _ellipse_from_spectral_matrix(Sxx_del, Syy_del, Sxy_del)
            d1_del[:, k] = d1_k
            d2_del[:, k] = d2_k
            P_del[:, k] = P_k
            theta_del[:, k] = theta_k
        end
        # Linear (non-log) jackknife CI, not the log-transformed `_jackknife_ci`
        # from rotary_spectrum.jl: d1/d2 are spectral-matrix EIGENVALUES, not
        # raw per-taper auto-spectra, and the minor eigenvalue d2 is exactly 0
        # for any perfectly rectilinear (fully linearly polarized) signal at
        # its own frequency -- log(0) is a real, not just theoretical, case
        # here (hit it in testing: a pure v=0 line signal). A log-CI is the
        # wrong tool whenever the quantity can legitimately be zero.
        ci_d1 = _jackknife_ci_linear(d1_del, confidence, 0.0, Inf)
        ci_d2 = _jackknife_ci_linear(d2_del, confidence, 0.0, Inf)
        ci_P = _jackknife_ci_linear(P_del, confidence, 0.0, 1.0)
        ci_theta = _jackknife_ci_circular_half_angle(theta, theta_del, confidence)
    end

    params = (nw=Float64(nw), ntapers=K, dt_hours=Float64(dt_hours),
              detrend=detrend, confidence=Float64(confidence), N=n)

    return ValTools.Types.EllipsePolarizationEstimate(freqs_pos, d1, d2, theta, nu, P, alpha, beta,
                                                       ci_d1, ci_d2, ci_P, ci_theta, params)
end

"""
    ellipse_polarization(u::Types.TimeSeriesVector, v::Types.TimeSeriesVector; kwargs...)

Typed overload of [`ellipse_polarization`](@ref), same `dt_hours`-from-
timestamps and common-unit-conversion conventions as the typed
`rotary_spectrum` overload in `rotary_spectrum.jl` (which also defines the
shared [`_dt_hours_from_time`](@ref) helper this uses).

No same-signature stub exists in the main package for this method (see
the NOTE in `Metrics/spectral.jl`) — without `using Multitaper`, calling
this with `TimeSeriesVector` arguments raises a plain `MethodError`.
"""
function Met.ellipse_polarization(u::ValTools.Types.TimeSeriesVector, v::ValTools.Types.TimeSeriesVector; kwargs...)
    u.time == v.time || error("ellipse_polarization: u and v must share the same time axis")
    dt_hours = _dt_hours_from_time(u.time)
    uv = Unitful.ustrip.(u.value)
    common_unit = Unitful.unit(first(u.value))
    vv = Unitful.ustrip.(Unitful.uconvert.(common_unit, v.value))
    return Met.ellipse_polarization(uv, vv; dt_hours=dt_hours, kwargs...)
end

# Core polparams.m + specdiag.m formulas, vectorized over frequency.
# Sxx, Syy real and positive (auto-spectra); Sxy complex (cross-spectrum).
# `atan(y, x)` (two-argument, Julia's atan2) already resolves the quadrant
# that jLab's single-argument `atan` + explicit sign-flip achieves by hand
# (verified equivalent by tracing both paths) — so no extra quadrant
# correction is needed here beyond the two-argument call itself.
function _ellipse_from_spectral_matrix(Sxx::AbstractVector{<:Real}, Syy::AbstractVector{<:Real},
                                       Sxy::AbstractVector{<:Complex})
    trS = Sxx .+ Syy
    detS = Sxx .* Syy .- abs2.(Sxy)  # real: det of a Hermitian 2x2 matrix

    P = sqrt.(max.(1.0 .- 4.0 .* detS ./ (trS .^ 2), 0.0))
    alpha = (Sxx .- Syy) ./ trS
    beta = 2.0 .* Sxy ./ trS

    alphamax = sqrt.(alpha .^ 2 .+ real.(beta) .^ 2)
    theta = atan.(real.(beta), alpha) ./ 2
    nu = atan.(imag.(beta), alphamax) ./ 2

    disc = max.(trS .^ 2 .- 4.0 .* detS, 0.0)  # clamp: Hermitian S has real, non-negative discriminant up to fp noise
    d1 = (trS .+ sqrt.(disc)) ./ 2
    d2 = (trS .- sqrt.(disc)) ./ 2

    return d1, d2, theta, nu, P, alpha, beta
end

# Linear (non-log) jackknife CI for a bounded quantity like P in [0,1].
# Unlike `_jackknife_ci` (log-transformed, for positive power spectra),
# P can be exactly 0, so a log transform is invalid here.
function _jackknife_ci_linear(X_delete::AbstractMatrix{<:Real}, confidence::Real, lo::Real, hi::Real)
    nfreq, K = size(X_delete)
    mbar = vec(mean(X_delete; dims=2))
    jkvar = ((K - 1) / K) .* vec(sum((X_delete .- mbar) .^ 2; dims=2))

    alpha = 1 - confidence
    zval = sqrt(2) * erfinv(2 * (1 - alpha / 2) - 1)
    half_width = zval .* sqrt.(jkvar)

    lower = clamp.(mbar .- half_width, lo, hi)
    upper = clamp.(mbar .+ half_width, lo, hi)
    return (lower, upper)
end

# Circular jackknife CI for the ellipse orientation `theta`. A plain linear
# jackknife (mean + sum-of-squared-deviations) is wrong for an angle: theta
# wraps at the branch cut of atan2/2, so two delete-one estimates that are
# physically nearly identical (e.g. theta=1.55 and theta=-1.55, ~0.04 rad
# apart going the short way around) get treated as ~pi apart, producing a
# huge, meaningless jackknife variance whenever the true orientation sits
# near that boundary.
#
# Fix: theta's natural period is pi (an ellipse's axis, not a directed
# vector), so `twoth = 2*theta` is the genuinely 2*pi-periodic quantity --
# exactly the value atan2 already computes internally before halving in
# `_ellipse_from_spectral_matrix`. Jackknife THAT on the circle (resultant-
# vector circular mean; deviations as the signed shortest arc via
# atan2(sin(d), cos(d)), not plain subtraction) using the same jackknife
# variance formula as `_jackknife_ci_linear`, then halve back to theta's
# domain. This is a standard circular-statistics jackknife (Fisher 1993,
# "Statistical Analysis of Circular Data") -- same formula shape as the
# ordinary jackknife, only the "distance" operator changes to respect the
# periodic domain.
#
# Bounds are NOT re-wrapped into theta's canonical (-pi/2, pi/2] range after
# halving -- a half-width approaching pi/2 (in theta's domain) genuinely
# means orientation is poorly constrained (e.g. a near-isotropic signal, or
# any noise-dominated frequency far from the actual spectral peak), and
# forcing the reported interval back into a fixed window would silently
# hide that. Interpret a very wide ci_theta as "orientation not resolved,"
# not as an error.
#
# Centered on `theta_point` (the actual full-sample point estimate, from
# ALL K tapers), not on the delete-one replicates' own circular mean --
# found by testing: those two differ enough at some frequencies (nearly
# degenerate/near-isotropic bins especially) that centering on the
# delete-one mean occasionally produced a reported interval that didn't
# bracket the point estimate it's supposedly a CI *for*. Centering on the
# point estimate directly guarantees bracketing by construction (the
# half-width is >= 0), while the delete-one replicates still supply an
# honest circular-dispersion estimate for the width.
function _jackknife_ci_circular_half_angle(theta_point::AbstractVector{<:Real},
                                           theta_delete::AbstractMatrix{<:Real}, confidence::Real)
    nfreq, K = size(theta_delete)
    phi_point = 2.0 .* theta_point  # the genuinely 2*pi-periodic doubled angle
    phi = 2.0 .* theta_delete

    d = atan.(sin.(phi .- phi_point), cos.(phi .- phi_point))  # signed shortest-arc deviation from the point estimate
    jkvar = ((K - 1) / K) .* vec(sum(d .^ 2; dims=2))

    alpha = 1 - confidence
    zval = sqrt(2) * erfinv(2 * (1 - alpha / 2) - 1)
    half_width = zval .* sqrt.(jkvar)

    lower = (phi_point .- half_width) ./ 2
    upper = (phi_point .+ half_width) ./ 2
    return (lower, upper)
end
