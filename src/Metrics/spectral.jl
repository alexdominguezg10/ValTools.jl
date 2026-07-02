"""
    rotary_spectrum(u, v; dt_hours=1.0, detrend="linear", window="hann")

Rotary (CW/CCW) spectral decomposition of a velocity time series.

Returns `(freqs, S_ccw, S_cw)` where `freqs` are positive frequencies
in cycles per hour.
"""
function rotary_spectrum(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                         dt_hours::Real=1.0,
                         detrend::String="linear",
                         window::String="hann")
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
    end

    if window == "hann"
        win = _hann_window(n)
        win_norm = win ./ sqrt(mean(win .^ 2))
        uf = uf .* win_norm
        vf = vf .* win_norm
    end

    w = uf .+ im .* vf
    W = fft(w)
    freqs_all = fftfreq(n, dt_hours)

    psd_all = (abs.(W) .^ 2) .* dt_hours ./ n

    pos_mask = freqs_all .> 0
    neg_mask = freqs_all .< 0

    freqs_pos = freqs_all[pos_mask]
    S_ccw = psd_all[pos_mask]

    freqs_neg = -freqs_all[neg_mask]
    S_cw_raw = psd_all[neg_mask]
    order = sortperm(freqs_neg)
    freqs_neg = freqs_neg[order]
    S_cw_raw = S_cw_raw[order]

    S_cw = _interp1(freqs_neg, S_cw_raw, freqs_pos)

    return freqs_pos, S_ccw, S_cw
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
