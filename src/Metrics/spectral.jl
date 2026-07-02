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
"""
function rotary_spectrum(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                         dt_hours::Real=1.0,
                         detrend::String="linear",
                         nw::Real=4.0,
                         ntapers::Int=0,
                         ci::Bool=true,
                         confidence::Real=0.95)
    length(u) == length(v) || error("u and v must have the same length")
    any(!isfinite, u) && error("u must not contain NaN/Inf")
    any(!isfinite, v) && error("v must not contain NaN/Inf")

    uf = Float64.(u)
    vf = Float64.(v)
    n = length(uf)

    if detrend == "linear"
        uf = _detrend_linear(uf)
        vf = _detrend_linear(vf)
    elseif detrend == "constant"
        uf = uf .- mean(uf)
        vf = vf .- mean(vf)
    elseif detrend != "none"
        error("detrend must be \"none\", \"constant\", or \"linear\"")
    end

    K = ntapers > 0 ? ntapers : max(1, 2 * floor(Int, nw) - 1)
    tapers, _lambdas = dpss_tapers(n, Float64(nw), K, :both)

    freqs_all = fftfreq(n, dt_hours)
    pos_mask = freqs_all .> 0
    neg_mask = freqs_all .< 0
    freqs_pos = freqs_all[pos_mask]
    freqs_neg_raw = -freqs_all[neg_mask]
    order = sortperm(freqs_neg_raw)
    freqs_neg = freqs_neg_raw[order]

    S_ccw_k = Matrix{Float64}(undef, length(freqs_pos), K)
    S_cw_k = Matrix{Float64}(undef, length(freqs_pos), K)

    for k in 1:K
        taper = @view tapers[:, k]
        w_k = (uf .* taper) .+ im .* (vf .* taper)
        W_k = fft(w_k)
        psd_k = (abs.(W_k) .^ 2) .* dt_hours ./ n

        S_ccw_k[:, k] = psd_k[pos_mask]
        S_cw_raw = psd_k[neg_mask][order]
        S_cw_k[:, k] = _interp1(freqs_neg, S_cw_raw, freqs_pos)
    end

    S_ccw = vec(mean(S_ccw_k; dims=2))
    S_cw = vec(mean(S_cw_k; dims=2))

    ci_ccw = nothing
    ci_cw = nothing
    if ci && K > 1
        ci_ccw = _jackknife_ci(S_ccw_k, confidence)
        ci_cw = _jackknife_ci(S_cw_k, confidence)
    end

    rotary_coefficient = (S_ccw .- S_cw) ./ (S_ccw .+ S_cw)

    params = (nw=Float64(nw), ntapers=K, dt_hours=Float64(dt_hours),
              detrend=detrend, confidence=Float64(confidence), N=n)

    return Types.RotarySpectralEstimate(freqs_pos, S_ccw, S_cw, ci_ccw, ci_cw,
                                        rotary_coefficient, params)
end

"""
    _jackknife_ci(S_k, confidence)

Delete-one jackknife confidence interval across tapers `k`, computed on the
log-power scale (Thomson & Chave 1991), for a `(nfreq, K)` matrix of
single-taper power estimates. Returns `(lower, upper)` vectors.

Uses a normal approximation for the critical value (adequate for the small
taper counts typical here, `K` ~ 3–9); avoids adding a hard dependency on
Distributions.jl just for a Student-t quantile.
"""
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
"""
function rotary_coherence(u1::AbstractVector{<:Real}, v1::AbstractVector{<:Real},
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
            return _detrend_linear(xf)
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

    freqs_all = fftfreq(n, dt_hours)
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

    return Types.RotaryCoherenceEstimate(freqs_pos, coh_ccw, coh_cw, phase_ccw, phase_cw,
                                         significance_level, params)
end

"""
    _interp1_complex(xp, yp, xi)

Linear interpolation for complex-valued `yp`, applied independently to the
real and imaginary parts.
"""
function _interp1_complex(xp, yp::AbstractVector{<:Complex}, xi)
    re = _interp1(xp, real.(yp), xi)
    im_ = _interp1(xp, imag.(yp), xi)
    return re .+ im .* im_
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
