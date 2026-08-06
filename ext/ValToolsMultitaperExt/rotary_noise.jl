# Isotropic rotary noise model of Lilly & Perez-Brunius (2021, NPG) Sect. 4.3.
# Public docstrings live on the stubs in src/Metrics/spectral.jl.
#
# NOTE: these take `AbstractVector{<:Real}` while the src/ stubs take plain
# `AbstractVector` -- deliberately NARROWER, so Julia sees two distinct
# methods. Matching the stub's signature exactly makes this an illegal
# same-method redefinition across modules, rejected at precompilation
# ("Method overwriting is not permitted"). Same trick rotary_spectrum
# already uses, and the same trap the GPU spectral functions hit.

function Met.rotary_noise_spectrum(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                   dt_hours::Real=1.0,
                                   nw::Real=4.0,
                                   ntapers::Int=0,
                                   detrend::String="linear")
    length(u) == length(v) || error("u and v must have the same length")
    uf = Float64.(u)
    vf = Float64.(v)
    n = length(uf)
    n < 4 && error("u and v must have at least 4 samples")

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

    # Two-sided multitaper PSD of the COMPLEX signal w = u + iv, kept on the
    # native FFT grid. Deliberately NOT via Met.rotary_spectrum: that
    # interpolates the CW branch onto the positive-frequency grid and drops
    # DC/Nyquist, whereas Eq. 69 is a pointwise min over the native grid and
    # the surrogate's ifft needs every bin.
    # NOTE (checked 2026-08-05, do NOT "fix" this to match rotary_spectrum.jl):
    # this looks like the same missing-taper-normalization bug found in
    # Met.rotary_spectrum (extra /n on top of the tapers' own unit-energy
    # normalization), but it is not the same situation here. This function's
    # only consumer, rotary_noise_surrogate below, builds its complex noise
    # realization as `eps_t = ifft(amp.*z).*n` with `amp=sqrt(S_iso/dt)` --
    # a raw (unweighted-by-Δf) sum of S_iso over all n bins, which by
    # discrete Parseval's own factor-of-n exactly cancels this /n. Removing
    # it here without also reworking that formula would silently break the
    # surrogate's variance-matching (see "rotary_noise_surrogate — variance
    # and isotropy" in test/jlab/test_ellipse.jl, which checks realized
    # surrogate variance against the real input's own sample variance --
    # an externally-anchored check, not a self-consistency one, and it
    # currently passes). Individual S_full[i]/S_iso[i] values ARE off by
    # 1/n if read directly at a single frequency (same as the
    # rotary_spectrum bug), just not in the one place this module uses them.
    S_k = Matrix{Float64}(undef, n, K)
    for k in 1:K
        taper = @view tapers[:, k]
        w_k = (uf .* taper) .+ im .* (vf .* taper)
        W_k = fft(w_k)
        S_k[:, k] = (abs.(W_k) .^ 2) .* dt_hours ./ n
    end
    S_full = vec(mean(S_k; dims=2))

    freq = Met.fftfreq(n, dt_hours)

    # Eq. 69: S_iso(w) = min{S(w), S(-w)}, pointwise on the native grid.
    # Index of -f for FFT bin ordering: bin 1 is DC (self-conjugate
    # frequency), bins 2..n mirror as n+2-i.
    S_min = similar(S_full)
    for i in 1:n
        mirror = i == 1 ? 1 : n + 2 - i
        S_min[i] = min(S_full[i], S_full[mirror])
    end

    # Eq. 70: rescale so the isotropic spectrum carries the same total
    # variance as the original (a pointwise minimum always undershoots).
    denom = sum(S_min)
    c_eps = denom > 0 ? sum(S_full) / denom : 1.0
    S_iso = c_eps .* S_min

    params = (nw=Float64(nw), ntapers=K, dt_hours=Float64(dt_hours),
              detrend=detrend, N=n)
    return (freq=freq, S_iso=S_iso, S_full=S_full, c_eps=c_eps, params=params)
end

function Met.rotary_noise_surrogate(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                    dt_hours::Real=1.0,
                                    nw::Real=4.0,
                                    ntapers::Int=0,
                                    detrend::String="linear",
                                    rng=Random.default_rng())
    rng === nothing && (rng = Random.default_rng())
    spec = Met.rotary_noise_spectrum(u, v; dt_hours=dt_hours, nw=nw,
                                     ntapers=ntapers, detrend=detrend)
    n = spec.params.N
    S_iso = spec.S_iso

    # Complex Gaussian white noise with E|z|^2 = 1 per bin, shaped by
    # sqrt(S_iso). Scaling derivation (verified numerically, don't "simplify"):
    # want var(eps) = sum(S_iso)/dt. With E = fft(eps), Parseval gives
    # var(eps) = sum(|E|^2)/n^2, so we need |E| = n*sqrt(S_iso/dt). Since
    # eps = ifft(amp.*z)*n implies E = amp.*z*n, that means amp =
    # sqrt(S_iso/dt) -- with NO extra sqrt(n) (an earlier version had one and
    # inflated the surrogate variance by exactly a factor of n).
    amp = sqrt.(S_iso ./ dt_hours)
    z = (randn(rng, n) .+ im .* randn(rng, n)) ./ sqrt(2)
    eps_t = ifft(amp .* z) .* n

    # Restore the record's mean velocity (surrogate is built from the
    # detrended/demeaned spectrum, so it is zero-mean by construction).
    mu_u = mean(Float64.(u))
    mu_v = mean(Float64.(v))
    return (real.(eps_t) .+ mu_u, imag.(eps_t) .+ mu_v)
end
