"""
    rotary_spectrum(u, v; dt_hours=1.0, detrend="linear", nw=4.0, ntapers=0, ci=true, confidence=0.95)

Multitaper rotary (CW/CCW) spectral decomposition of a velocity time series
`w = u + iv`, following Gonella (1972): positive frequencies of `w`'s FFT
correspond to counter-clockwise (CCW) rotation, negative frequencies to
clockwise (CW) rotation.

This is the single, unified implementation of rotary spectral analysis in
ValTools.jl (previously duplicated as `JLab.rotary`, which is now a thin
deprecated wrapper around this function).

# Arguments
- `u`, `v`: east-west / north-south velocity components (equal length)
- `dt_hours`: sampling interval (hours by default; any consistent time unit)
- `detrend`: `"none"`, `"constant"`, or `"linear"` (default `"linear"`)
- `nw`: DPSS time-bandwidth product (default 4.0)
- `ntapers`: number of tapers (default: `2*floor(nw) - 1`)
- `ci`: compute jackknife-over-tapers confidence intervals (default `true`)
- `confidence`: confidence level for `ci_ccw`/`ci_cw` (default 0.95)
- `ftest`: compute a Thomson (1982) harmonic F-test for a line component in
  each rotary branch (default `true`; see `ftest_ccw`/`ftest_cw` below)

# Returns
A [`Types.RotarySpectralEstimate`](@ref) with fields `freq`, `S_ccw`, `S_cw`,
`ci_ccw`, `ci_cw`, `rotary_coefficient`, `ftest_ccw`, `ftest_cw`, `params`.
Also supports tuple destructuring `freqs, S_ccw, S_cw = rotary_spectrum(u, v)`
for backward compatibility.

`ftest_ccw`/`ftest_cw` are per-frequency p-values under the null hypothesis
of no coherent line component (pure background noise) in that rotary
branch — small values (e.g. `< 0.05`) flag genuine spectral peaks (an
inertial oscillation, a tidal line) as distinct from noise, which
`ci_ccw`/`ci_cw` alone cannot do (a wide confidence interval doesn't tell
you whether a bump is signal or fluctuation).

# References
Gonella, J. (1972). A rotary-component method for analysing meteorological
and oceanographic vector time series. Deep Sea Res., 19(12), 833–846.
Thomson, D. J. (1982). Spectrum estimation and harmonic analysis.
Proc. IEEE, 70(9), 1055–1096.

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function rotary_spectrum(u::AbstractVector, v::AbstractVector;
                         dt_hours::Real=1.0,
                         detrend::String="linear",
                         nw::Real=4.0,
                         ntapers::Int=0,
                         ci::Bool=true,
                         confidence::Real=0.95,
                         ftest::Bool=true)
    error("rotary_spectrum requires Multitaper.jl. Load it first: `using Multitaper`")
end

"""
    rotary_noise_spectrum(u, v; dt_hours=1.0, nw=4.0, ntapers=0, detrend="linear")

Isotropic ("no preferred rotation sense") noise spectrum of a velocity time
series `w = u + iv`, following Lilly & Perez-Brunius (2021, NPG) Sect. 4.3,
Eqs. 69-70.

The multitaper power spectrum of `w` is estimated on the **native two-sided
FFT frequency grid**, then made rotationally isotropic by taking, at each
frequency, the smaller of the CCW (`+ω`) and CW (`-ω`) power:

    S_εε(ω) = c_ε · min{Ŝ(ω), Ŝ(-ω)}

Because eddies and inertial oscillations raise the spectrum on *one* rotary
side at a given frequency, taking the elementwise minimum strips out
rotationally-anisotropic features (exactly the ones an eddy detector should
be testing against a null) while retaining the record's real broadband
spectral *shape* — unlike a parametric AR(1)/white null, which cannot
represent an inertial peak at all. `c_ε` (Eq. 70) rescales so the isotropic
spectrum integrates to the same total velocity variance as the original,
compensating the fact that a pointwise minimum always underestimates.

Use [`rotary_noise_surrogate`](@ref) to draw stochastic realizations from
this spectrum for Monte Carlo significance testing.

# Arguments
- `u`, `v`: east-west / north-south velocity components (equal length)
- `dt_hours`: sampling interval (hours by default; any consistent time unit)
- `nw`: DPSS time-bandwidth product (default 4.0, the paper's choice)
- `ntapers`: number of tapers (default: `2*floor(nw) - 1`)
- `detrend`: `"none"`, `"constant"`, or `"linear"` (default `"linear"`)

# Returns
NamedTuple `(freq, S_iso, S_full, c_eps, params)`:
- `freq`: two-sided FFT frequency grid (unshifted, i.e. `FFTW` bin order —
  the same order `S_iso` is in, so it can be fed straight to `ifft`)
- `S_iso`: the isotropic noise spectrum `S_εε`
- `S_full`: the un-symmetrized multitaper spectrum `Ŝ` it was built from
- `c_eps`: the Eq. 70 normalization constant actually applied

# References
Lilly, J. M., & Perez-Brunius, P. (2021). Extracting statistically
significant eddy signals from large Lagrangian datasets using wavelet ridge
analysis, with application to the Gulf of Mexico. Nonlin. Processes
Geophys., 28, 181-212. Sect. 4.3, Eqs. 69-70.

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function rotary_noise_spectrum(u::AbstractVector, v::AbstractVector;
                               dt_hours::Real=1.0,
                               nw::Real=4.0,
                               ntapers::Int=0,
                               detrend::String="linear")
    error("rotary_noise_spectrum requires Multitaper.jl. Load it first: `using Multitaper`")
end

"""
    rotary_noise_surrogate(u, v; dt_hours=1.0, nw=4.0, ntapers=0, detrend="linear", rng=Random.default_rng())

Draw one stochastic realization of the isotropic noise process defined by
[`rotary_noise_spectrum`](@ref) — the null-hypothesis "no eddies" velocity
series of Lilly & Perez-Brunius (2021) Sect. 4.3.

The surrogate is built by multiplying complex Gaussian white noise by
`sqrt(S_εε(ω))` at each frequency and inverse-transforming, giving a series
with the prescribed isotropic spectrum but random phase — so it matches the
original record's spectral shape and total variance while containing no
organized (rotationally anisotropic) oscillations.

# Returns
`(εx, εy)`: two real velocity series (east/north components), same length
as the input, with the input's temporal mean velocity added back.

# References
Lilly & Perez-Brunius (2021), Sect. 4.3. See [`rotary_noise_spectrum`](@ref).

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function rotary_noise_surrogate(u::AbstractVector, v::AbstractVector;
                                dt_hours::Real=1.0,
                                nw::Real=4.0,
                                ntapers::Int=0,
                                detrend::String="linear",
                                rng=nothing)
    error("rotary_noise_surrogate requires Multitaper.jl. Load it first: `using Multitaper`")
end

# NOTE: a `rotary_spectrum(u::Types.TimeSeriesVector, v::Types.TimeSeriesVector; ...)`
# typed-stub method deliberately does NOT exist here. Unlike the
# `AbstractVector` vs `AbstractVector{<:Real}` stub/extension split above
# (two genuinely different signatures, so both can coexist), a stub taking
# exactly `Types.TimeSeriesVector` would have the IDENTICAL signature to
# the extension's real implementation -- Julia treats that as redefining
# the same method, which errors during precompilation ("Method
# overwriting is not permitted"). The typed method is defined only in
# ValToolsMultitaperExt/rotary_spectrum.jl; without Multitaper loaded,
# calling it raises a plain MethodError (no custom message, but not wrong).

"""
    rotary_coherence(u1, v1, u2, v2; dt_hours=1.0, detrend="linear", nw=4.0, ntapers=0, confidence=0.95)

Multitaper rotary cross-spectral coherence between two velocity time series
`w1 = u1 + iv1` and `w2 = u2 + iv2`. The CW/CCW decomposition of Gonella
(1972) is applied to each series separately, then coherence and phase are
computed between the two CCW components and, independently, the two CW
components (Mooers 1973; Kundu 1976). This lets you ask, e.g., whether a
model and observed current agree specifically in their inertial (CW, NH)
content versus their tidal/wind-driven (CCW) content.

# Arguments
- `u1`, `v1`: velocity components of the first series
- `u2`, `v2`: velocity components of the second series (same length as series 1)
- `dt_hours`: sampling interval (hours by default; any consistent time unit)
- `detrend`: `"none"`, `"constant"`, or `"linear"` (default `"linear"`)
- `nw`: DPSS time-bandwidth product (default 4.0)
- `ntapers`: number of tapers (default: `2*floor(nw) - 1`)
- `confidence`: confidence level for `significance_level` (default 0.95)

# Returns
A [`Types.RotaryCoherenceEstimate`](@ref) with fields `freq`, `coh_ccw`,
`coh_cw`, `phase_ccw`, `phase_cw`, `significance_level`, `params`.

The significance level follows the standard multitaper result for the null
distribution of magnitude-squared coherence with `K` tapers (Thomson & Chave
1991; Percival & Walden 1993, §8.13): critical value `1 - (1-confidence)^(1/(K-1))`.
Coherence values above this line are unlikely to arise from independent
(zero true coherence) signals.

# References
Gonella, J. (1972). Deep Sea Res., 19(12), 833–846.
Mooers, C. N. K. (1973). Deep Sea Res., 20(12), 1129–1141.
Kundu, P. K. (1976). J. Phys. Oceanogr., 6(2), 238–242.

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function rotary_coherence(u1::AbstractVector, v1::AbstractVector,
                          u2::AbstractVector, v2::AbstractVector;
                          dt_hours::Real=1.0,
                          detrend::String="linear",
                          nw::Real=4.0,
                          ntapers::Int=0,
                          confidence::Real=0.95)
    error("rotary_coherence requires Multitaper.jl. Load it first: `using Multitaper`")
end

"""
    cross_coherence(x, y; dt=1.0, detrend="linear", nw=4.0, ntapers=0, confidence=0.95)

Multitaper cross-spectral coherence between two ordinary real time series
`x` and `y` — the non-rotary counterpart of [`rotary_coherence`](@ref) for
scalar records (e.g. two SST time series, a wind stress and a current
component, two current-meter records at different depths) rather than
complex velocity. Uses the same DPSS-tapered cross-periodogram averaging
as `rotary_coherence`, without the CW/CCW split.

# Arguments
- `x`, `y`: real time series of equal length
- `dt`: sampling interval (any consistent time unit)
- `detrend`: `"none"`, `"constant"`, or `"linear"` (default `"linear"`)
- `nw`: DPSS time-bandwidth product (default 4.0)
- `ntapers`: number of tapers (default: `2*floor(nw) - 1`)
- `confidence`: confidence level for `significance_level` (default 0.95)

# Returns
A [`Types.CrossSpectralEstimate`](@ref) with fields `freq`, `cross_power`,
`coherence`, `phase`, `significance_level`, `params`.

The significance level follows the standard multitaper result for the null
distribution of magnitude-squared coherence with `K` tapers (Thomson & Chave
1991; Percival & Walden 1993, §8.13): critical value `1 - (1-confidence)^(1/(K-1))`.
Coherence values above this line are unlikely to arise from independent
(zero true coherence) signals.

# References
Percival, D. B. & Walden, A. T. (1993). Spectral Analysis for Physical
Applications. Cambridge University Press, §8.13.

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function cross_coherence(x::AbstractVector, y::AbstractVector;
                         dt::Real=1.0,
                         detrend::String="linear",
                         nw::Real=4.0,
                         ntapers::Int=0,
                         confidence::Real=0.95)
    error("cross_coherence requires Multitaper.jl. Load it first: `using Multitaper`")
end

# See the NOTE above rotary_coherence's stub: no same-signature typed
# stub here either, for the identical reason. Defined only in
# ValToolsMultitaperExt/cross_coherence.jl.

"""
    ellipse_polarization(u, v; dt_hours=1.0, detrend="linear", nw=4.0, ntapers=0, ci=true, confidence=0.95)

Frequency-domain ellipse polarization decomposition of a bivariate (u,v)
velocity time series. Port of jLab's `polparams`/`specdiag` (Lilly),
applied to the same multitaper spectral matrix machinery used by
[`cross_coherence`](@ref). See [`Types.EllipsePolarizationEstimate`](@ref)
for the returned fields.

`u` and `v` must be the same length and contain no NaN/Inf. Requires
`using Multitaper` to be loaded — the real implementation lives in the
ValToolsMultitaperExt package extension.
"""
function ellipse_polarization(u::AbstractVector, v::AbstractVector;
                              dt_hours::Real=1.0,
                              detrend::String="linear",
                              nw::Real=4.0,
                              ntapers::Int=0,
                              ci::Bool=true,
                              confidence::Real=0.95)
    error("ellipse_polarization requires Multitaper.jl. Load it first: `using Multitaper`")
end

# Same NOTE as rotary_spectrum/cross_coherence above: no same-signature
# typed stub here. Defined only in
# ValToolsMultitaperExt/ellipse_polarization.jl.


"""
    alongtrack_wavenumber_spectrum(ssh_track, dx_km; detrend="linear", window="hann")

1-D power spectral density of an along-track field using Welch's method.

Returns `(k, psd)` where `k` is in cycles/km.
"""
function alongtrack_wavenumber_spectrum(ssh_track::AbstractVector{<:Real},
                                        dx_km::Real;
                                        detrend::String="linear",
                                        window::String="hann")
    x = Float64.(ssh_track)
    any(!isfinite, x) && error("ssh_track must not contain NaN/Inf")

    n = length(x)
    nperseg = min(n, 256)
    noverlap = nperseg ÷ 2
    fs = 1.0 / dx_km

    k, psd = _welch(x, fs, nperseg, noverlap; detrend=detrend, window=window)

    valid = k .> 0
    return k[valid], psd[valid]
end


"""
    isotropic_2d_spectrum(field, dx_km, dy_km; detrend="linear", window="hann", n_bins=nothing)

Radially-averaged 2-D power spectral density.

Returns `(k_iso, psd_iso)` where `k_iso` is in cycles/km.
"""
function isotropic_2d_spectrum(field::AbstractMatrix{<:Real},
                               dx_km::Real, dy_km::Real;
                               detrend::String="linear",
                               window::String="hann",
                               n_bins::Union{Int,Nothing}=nothing)
    f = Float64.(field)
    any(!isfinite, f) && error("field must not contain NaN/Inf")

    ny, nx = size(f)

    if detrend == "linear"
        f = detrend_2d_linear(f)
    elseif detrend == "constant"
        f = f .- mean(f)
    end

    if window == "hann"
        wx = _hann_window(nx)
        wy = _hann_window(ny)
        win2d = wy * wx'
        win_norm = win2d ./ sqrt(mean(win2d .^ 2))
        f = f .* win_norm
    end

    F = fftshift(fft(f))
    psd2d = (abs.(F) .^ 2) .* dx_km .* dy_km ./ (nx * ny)

    kx = fftshift(fftfreq(nx, dx_km))
    ky = fftshift(fftfreq(ny, dy_km))
    K = [sqrt(kx_i^2 + ky_j^2) for ky_j in ky, kx_i in kx]

    nbins = n_bins !== nothing ? n_bins : min(ny, nx) ÷ 2
    k_max = maximum(K)
    bin_edges = range(0, k_max; length=nbins + 1)
    bin_centers = 0.5 .* (bin_edges[1:end-1] .+ bin_edges[2:end])

    psd_iso = fill(NaN, nbins)
    for i in 1:nbins
        mask = (K .>= bin_edges[i]) .& (K .< bin_edges[i+1])
        if any(mask)
            psd_iso[i] = mean(psd2d[mask])
        end
    end

    valid = bin_centers .> 0
    return collect(bin_centers[valid]), psd_iso[valid]
end


"""
    cross_spectrum_kx_ky(field1, field2, dx_km, dy_km; detrend="linear", window="hann", n_bins=nothing)

Isotropic cross-spectrum of two co-located 2-D fields.

Returns `(k_iso, coherence, phase, cross_psd)`.
"""
function cross_spectrum_kx_ky(field1::AbstractMatrix{<:Real},
                              field2::AbstractMatrix{<:Real},
                              dx_km::Real, dy_km::Real;
                              detrend::String="linear",
                              window::String="hann",
                              n_bins::Union{Int,Nothing}=nothing)
    size(field1) == size(field2) || error("field1 and field2 must have the same shape")

    ny, nx = size(field1)

    function _prep(f)
        f = Float64.(f)
        if detrend == "linear"
            f = detrend_2d_linear(f)
        elseif detrend == "constant"
            f = f .- mean(f)
        end
        if window == "hann"
            wx = _hann_window(nx)
            wy = _hann_window(ny)
            win2d = wy * wx'
            win_norm = win2d ./ sqrt(mean(win2d .^ 2))
            f = f .* win_norm
        end
        return f
    end

    F1 = fftshift(fft(_prep(field1)))
    F2 = fftshift(fft(_prep(field2)))

    norm = dx_km * dy_km / (nx * ny)
    P11 = (abs.(F1) .^ 2) .* norm
    P22 = (abs.(F2) .^ 2) .* norm
    P12 = (F1 .* conj.(F2)) .* norm

    kx = fftshift(fftfreq(nx, dx_km))
    ky = fftshift(fftfreq(ny, dy_km))
    K = [sqrt(kx_i^2 + ky_j^2) for ky_j in ky, kx_i in kx]

    nbins = n_bins !== nothing ? n_bins : min(ny, nx) ÷ 2
    k_max = maximum(K)
    bin_edges = range(0, k_max; length=nbins + 1)
    bin_centers = 0.5 .* (bin_edges[1:end-1] .+ bin_edges[2:end])

    coherence = fill(NaN, nbins)
    phase_arr = fill(NaN, nbins)
    cross_psd = fill(NaN, nbins)

    for i in 1:nbins
        mask = (K .>= bin_edges[i]) .& (K .< bin_edges[i+1])
        if !any(mask)
            continue
        end
        p11 = mean(P11[mask])
        p22 = mean(P22[mask])
        p12 = mean(P12[mask])
        denom = p11 * p22
        coherence[i] = denom > 0 ? abs(p12)^2 / denom : NaN
        phase_arr[i] = angle(p12)
        cross_psd[i] = real(p12)
    end

    valid = bin_centers .> 0
    return collect(bin_centers[valid]), coherence[valid], phase_arr[valid], cross_psd[valid]
end


"""
    detrend_2d_linear(f)

Remove a least-squares 2-D linear plane fit from `f`.
"""
function detrend_2d_linear(f::AbstractMatrix{<:Real})
    ny, nx = size(f)
    y_grid = repeat(0:ny-1, 1, nx)
    x_grid = repeat((0:nx-1)', ny, 1)
    A = hcat(vec(x_grid), vec(y_grid), ones(ny * nx))
    coeffs = A \ vec(f)
    plane = reshape(A * coeffs, ny, nx)
    return f .- plane
end


# ── Internal helpers ──────────────────────────────────────────────

function _hann_window(n::Int)
    return @. 0.5 * (1 - cos(2π * (0:n-1) / (n - 1)))
end

function _detrend_linear(x::AbstractVector)
    n = length(x)
    t = collect(0.0:n-1)
    A = hcat(t, ones(n))
    coeffs = A \ x
    return x .- A * coeffs
end

function _welch(x::AbstractVector, fs::Real, nperseg::Int, noverlap::Int;
                detrend::String="linear", window::String="hann")
    n = length(x)
    step = nperseg - noverlap
    n_segments = max(1, (n - nperseg) ÷ step + 1)

    win = window == "hann" ? _hann_window(nperseg) : ones(nperseg)
    win_ss = sum(win .^ 2)

    nfreqs = nperseg ÷ 2 + 1
    psd = zeros(nfreqs)

    for s in 1:n_segments
        i0 = (s - 1) * step + 1
        i1 = i0 + nperseg - 1
        seg = Float64.(x[i0:i1])

        if detrend == "linear"
            seg = _detrend_linear(seg)
        elseif detrend == "constant"
            seg = seg .- mean(seg)
        end

        seg = seg .* win
        F = fft(seg)

        for k in 1:nfreqs
            psd[k] += abs(F[k])^2
        end
    end

    psd ./= (n_segments * fs * win_ss)
    # double one-sided (except DC and Nyquist)
    psd[2:end-1] .*= 2

    freqs = collect(range(0, fs / 2; length=nfreqs))
    return freqs, psd
end

function fftfreq(n::Int, d::Real)
    f = Vector{Float64}(undef, n)
    if iseven(n)
        for i in 0:n÷2-1
            f[i+1] = i / (n * d)
        end
        for i in n÷2:n-1
            f[i+1] = (i - n) / (n * d)
        end
    else
        for i in 0:(n-1)÷2
            f[i+1] = i / (n * d)
        end
        for i in (n+1)÷2:n-1
            f[i+1] = (i - n) / (n * d)
        end
    end
    return f
end

function fftshift(x::AbstractVector)
    n = length(x)
    mid = n ÷ 2
    return vcat(x[mid+1:end], x[1:mid])
end

function fftshift(x::AbstractMatrix)
    ny, nx = size(x)
    my, mx = ny ÷ 2, nx ÷ 2
    return vcat(
        hcat(x[my+1:end, mx+1:end], x[my+1:end, 1:mx]),
        hcat(x[1:my, mx+1:end],     x[1:my, 1:mx])
    )
end

# ============================================================================
# SVD-BASED POLARIZATION ANALYSIS (Park, Vernon & Lindberg 1987)
# ============================================================================

"""
    msvd(W::AbstractArray{<:Complex,3})

Multiple-transform (SVD-based) polarization analysis. Ported from jLab's
`jSpectral/msvd.m`. Pure linear algebra — no dependency on how `W` was
computed (multitaper eigencoefficients, multi-order wavelet coefficients,
...), so unlike [`rotary_spectrum`](@ref)/[`ellipse_polarization`](@ref)
this does NOT require `using Multitaper`.

`W` holds complex "eigentransform" coefficients: `size(W) == (J, N, K)`
where `J` is the number of frequency/scale bands, `N` is the number of
simultaneous data-set channels (e.g. `N=2` for a bivariate `u,v` velocity
record), and `K` is the number of eigentransforms (multitaper tapers, or
multi-order wavelets) sharing that band. At each band `j`, the `N x K`
sub-matrix `W[j,:,:]` (rescaled by `1/sqrt(K*N)`, matching jLab's own
normalization — see the note on `scale` in the implementation for why it's
`K*N` and not just `K`) is singular-value-decomposed:
`W[j,:,:]/sqrt(K*N) = U*Diagonal(d)*V'`.

Unlike forming the spectral matrix `S = W*W'/K` and eigendecomposing it
directly (the `specdiag`-based route `ellipse_polarization` and
[`JLab.ellpol`](@ref) both use, averaged/pooled over eigentransforms first),
MSVD works on the raw per-eigentransform matrix directly, giving BOTH:
- `u1`: the leading LEFT singular vector at each band — the eigenvector of
  the (implicit) `N x N` spectral matrix, i.e. the dominant polarization
  direction across channels. Carries the same information
  `ellipse_polarization`'s `theta`/`d1`/`d2` give via a different route
  (`|d[:,1]|^2` here is that route's `d1`).
- `v1`: the leading RIGHT singular vector — the eigenvector of the `K x K`
  "structure matrix", the pattern of agreement ACROSS the `K`
  eigentransforms themselves. This has no analog in the spectral-matrix
  route and is what makes MSVD useful for signal detection: a genuinely
  polarized/coherent signal has `v1` close to a constant vector (every
  eigentransform agrees on it), while incoherent noise does not.

`d[j,1]^2 / trS[j]` is the fraction of power at band `j` explained by the
single dominant polarized signal. `u2`/`v2` (the SECOND singular vectors)
support two-signal/signal-plus-noise separation, per jLab's own test —
e.g. projecting a known polarization structure onto `u1` vs. `u2` to
estimate per-band SNR (see `msvd_test2` in `jSpectral/msvd.m`).

Note on singular-vector sign/phase: as with any SVD, `(u1[j,:], v1[j,:])`
for a given band are only determined up to a common unit-phase factor —
`u1[j,:]*d[j,1]*v1[j,:]'` (the rank-1 reconstruction) is the
phase-invariant quantity, not `u1`/`v1` individually. Two correct
implementations (e.g. this one vs. real jLab MATLAB) can therefore return
`u1`/`v1` differing by an overall phase per band while still agreeing
exactly on `d` and on that reconstruction — confirmed directly against
real jLab output (2026-08-02).

# Returns
`NamedTuple` with:
- `d`: `(J, min(N,K))` singular values
- `u1`, `u2`: `(J, N)` first/second left singular vectors
- `v1`, `v2`: `(J, K)` first/second right singular vectors
- `trS`: `(J,)` trace of the implicit spectral matrix `W*W'/K` at each band

# References
Park, J., Vernon, F. L., & Lindberg, C. R. (1987). Frequency dependent
polarization analysis of high-frequency seismograms. J. Geophys. Res.,
92(B12), 12664–12674. Note jLab's own convention for `u1`/`v1` is the
TRANSPOSE of Park et al.'s original (per `msvd.m`'s own comment) — this
port matches jLab, not the original paper directly.
"""
function msvd(W::AbstractArray{<:Complex,3})
    J, N, K = size(W)
    minNK = min(N, K)
    # msvd.m normalizes by 1/sqrt(K*M) where, confusingly, its internal `M`
    # (reassigned via `[N,J,M,K]=size(mmat)` on the reshaped 4-D array) is
    # the CHANNEL count -- i.e. this file's `N` -- not an outer batch
    # dimension as the docstring's optional `M x J x N x K` form might
    # suggest. Verified directly against real MATLAB output (2026-08-02):
    # using 1/sqrt(K) alone reproduced u1/v1 (scale-invariant) but was off
    # from real d/trS by exactly sqrt(N) in every band.
    scale = 1 / sqrt(K * N)

    d = Matrix{Float64}(undef, J, minNK)
    u1 = Matrix{ComplexF64}(undef, J, N)
    v1 = Matrix{ComplexF64}(undef, J, K)
    u2 = Matrix{ComplexF64}(undef, J, N)
    v2 = Matrix{ComplexF64}(undef, J, K)
    trS = Vector{Float64}(undef, J)

    for j in 1:J
        Wj = scale .* @view(W[j, :, :])   # N x K
        F = svd(Wj)                        # Wj = F.U * Diagonal(F.S) * F.V'
        d[j, :] = F.S
        u1[j, :] = F.U[:, 1]
        v1[j, :] = F.V[:, 1]
        if minNK >= 2
            u2[j, :] = F.U[:, 2]
            v2[j, :] = F.V[:, 2]
        else
            u2[j, :] .= 0
            v2[j, :] .= 0
        end
        trS[j] = sum(abs2, Wj)
    end

    return (d=d, u1=u1, v1=v1, u2=u2, v2=v2, trS=trS)
end

"""
    msvd(W::AbstractMatrix{<:Complex})

Single-band convenience form of [`msvd`](@ref): `W` is `N x K` (one
frequency/scale band's eigentransform matrix directly, no leading `J`
axis) — matches jLab's own dual-mode `msvd` input handling. Returns the
same fields as the 3-argument form but with the `J` axis squeezed out
(`d`/`u1`/`v1`/`u2`/`v2` become plain vectors, `trS` a scalar).
"""
function msvd(W::AbstractMatrix{<:Complex})
    N, K = size(W)
    r = msvd(reshape(W, 1, N, K))
    return (d=vec(r.d), u1=vec(r.u1), v1=vec(r.v1), u2=vec(r.u2), v2=vec(r.v2), trS=r.trS[1])
end
