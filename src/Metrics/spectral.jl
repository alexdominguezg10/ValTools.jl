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

# Returns
A [`Types.RotarySpectralEstimate`](@ref) with fields `freq`, `S_ccw`, `S_cw`,
`ci_ccw`, `ci_cw`, `rotary_coefficient`, `params`. Also supports tuple
destructuring `freqs, S_ccw, S_cw = rotary_spectrum(u, v)` for backward
compatibility.

# References
Gonella, J. (1972). A rotary-component method for analysing meteorological
and oceanographic vector time series. Deep Sea Res., 19(12), 833–846.

Requires `using Multitaper` to be loaded — the real implementation lives in
the ValToolsMultitaperExt package extension.
"""
function rotary_spectrum(u::AbstractVector, v::AbstractVector;
                         dt_hours::Real=1.0,
                         detrend::String="linear",
                         nw::Real=4.0,
                         ntapers::Int=0,
                         ci::Bool=true,
                         confidence::Real=0.95)
    error("rotary_spectrum requires Multitaper.jl. Load it first: `using Multitaper`")
end

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
