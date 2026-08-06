# Reproduce Fig. 2 of Jonathan Lilly's unfunded Flare NSF proposal: the
# rotary multitaper spectrum of w = u + iv for GDP drifter 44000 (WMO
# 4100571), K=15 tapers, with the raw periodogram overlaid in grey.
#
# Uses ValTools' unified `Metrics.rotary_spectrum` (re-exported as
# `ValTools.rotary_spectrum`) -- NOT the deprecated `JLab.rotary` thin
# wrapper, per this project's own documentation
# (src/Metrics/spectral.jl:9-11). No new library code.
#
# Data: the clean, gap-free 2005 calendar-year segment (8761 hourly rows),
# per memory: project_wavenumber_mooring_plan. ve/vn are m/s in the source
# CSV; the proposal's figures are in cm/s, so both are multiplied by 100
# before analysis (see that memory's "Unit note").
#
# Also saves the computed spectrum (freq, S_ccw, S_cw, periodogram) to CSV
# in results/flare_reproduction/, so flare_fig3_rotary_wavelet_ridge.jl can
# cross-reference the Kunze-effect frequency shift against the actual
# location of spectral power in the CW (negative-rotary) branch, without
# recomputing the spectrum twice.
#
# Usage (from ValTools.jl root, in an env with Multitaper.jl available):
#   julia --project=envs/cpu scripts/flare_fig2_rotary_spectrum.jl

using ValTools, ValTools.JLab
using Multitaper       # loads ValToolsMultitaperExt -- rotary_spectrum backend
using CairoMakie
using FFTW
using Dates, Statistics, Printf

# ── Config ───────────────────────────────────────────────────────────────
const CSV_PATH = "/Volumes/DATA_SSD/gdp_44000_2005.csv"
const OUT_PNG = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_fig2_rotary_spectrum.png")
const OUT_CSV = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_fig2_spectrum_data.csv")
const DT_HOURS = 1.0
const K_TAPERS = 15                 # Flare Fig. 2: K=15 tapers
const NW = 8.0                      # time-bandwidth product giving K=2*NW-1=15
const DETREND = "linear"
const FIGSIZE = (900, 550)
const DPI = 2

function read_gdp_csv(path::String)
    lines = readlines(path)
    header = split(lines[1], ",")
    col = Dict(name => i for (i, name) in enumerate(header))
    n = length(lines) - 2
    time = Vector{DateTime}(undef, n)
    lat = Vector{Float64}(undef, n)
    lon = Vector{Float64}(undef, n)
    ve = Vector{Float64}(undef, n)
    vn = Vector{Float64}(undef, n)
    for (k, line) in enumerate(@view lines[3:end])
        f = split(line, ",")
        tstr = f[col["time"]]
        tstr = endswith(tstr, "Z") ? tstr[1:end-1] : tstr
        time[k] = DateTime(tstr)
        lat[k] = parse(Float64, f[col["latitude"]])
        lon[k] = parse(Float64, f[col["longitude"]])
        ve[k] = parse(Float64, f[col["ve"]])
        vn[k] = parse(Float64, f[col["vn"]])
    end
    return (; time, lat, lon, ve, vn)
end

# ── Raw (boxcar, untapered) periodogram of w=u+iv, split into CCW/CW
# branches on the SAME frequency grid convention as Met.rotary_spectrum, so
# it is directly, apples-to-apples comparable on one log-log axis.
#
# S(f_k) = (dt/N)*|W_k|^2 is the standard periodogram normalization (no
# taper): by Parseval (unnormalized FFT: sum_n|w[n]|^2 = (1/N)sum_k|W_k|^2,
# i.e. sum_k|W_k|^2 = N^2*var(w)), sum_k S(f_k)*df with df=1/(N*dt) collapses
# to exactly var(w) -- i.e. integrating the periodogram over frequency
# recovers the input variance, the same convention the multitaper estimator
# satisfies for its own (unit-energy-taper) per-taper spectra. This is
# exactly what jLab's mspec.m computes for `MSPEC(Z,[])` ("boxcar taper ...
# returns the periodogram").
function rotary_periodogram(u::Vector{Float64}, v::Vector{Float64};
                             dt_hours::Real=1.0, detrend::String="linear")
    n = length(u)
    uf, vf = copy(u), copy(v)
    if detrend == "linear"
        for x in (uf, vf)
            t = collect(1.0:n)
            A = hcat(ones(n), t)
            coef = A \ x
            x .-= A * coef
        end
    elseif detrend == "constant"
        uf .-= mean(uf); vf .-= mean(vf)
    end
    w = ComplexF64.(uf) .+ im .* ComplexF64.(vf)
    W = fft(w)

    # Same frequency-grid construction as Metrics.fftfreq(n, dt_hours)
    freqs_all = Vector{Float64}(undef, n)
    if iseven(n)
        for i in 0:(n ÷ 2 - 1); freqs_all[i+1] = i / (n * dt_hours); end
        for i in (n ÷ 2):(n - 1); freqs_all[i+1] = (i - n) / (n * dt_hours); end
    else
        for i in 0:((n - 1) ÷ 2); freqs_all[i+1] = i / (n * dt_hours); end
        for i in ((n + 1) ÷ 2):(n - 1); freqs_all[i+1] = (i - n) / (n * dt_hours); end
    end
    pos_mask = freqs_all .> 0
    neg_mask = freqs_all .< 0
    freqs_pos = freqs_all[pos_mask]

    S_full = (abs.(W) .^ 2) .* dt_hours ./ n
    S_ccw = S_full[pos_mask]

    freqs_neg_raw = -freqs_all[neg_mask]
    order = sortperm(freqs_neg_raw)
    freqs_neg = freqs_neg_raw[order]
    S_cw_native = S_full[neg_mask][order]

    # Linear-interpolate the (slightly longer, includes Nyquist) negative
    # grid onto freqs_pos -- same purpose as the library's private
    # `_interp1`, reimplemented here standalone (script-local, no src/
    # dependency) since this function intentionally bypasses the tapered
    # multitaper code path.
    S_cw = similar(freqs_pos)
    for (i, f) in enumerate(freqs_pos)
        j = searchsortedlast(freqs_neg, f)
        if j < 1
            S_cw[i] = S_cw_native[1]
        elseif j >= length(freqs_neg)
            S_cw[i] = S_cw_native[end]
        else
            f0, f1 = freqs_neg[j], freqs_neg[j+1]
            frac = f1 > f0 ? (f - f0) / (f1 - f0) : 0.0
            S_cw[i] = S_cw_native[j] + frac * (S_cw_native[j+1] - S_cw_native[j])
        end
    end
    return freqs_pos, S_ccw, S_cw
end

function main()
    data = read_gdp_csv(CSV_PATH)
    n = length(data.time)
    mean_lat = mean(data.lat)
    println("Loaded $n rows; mean latitude = $(round(mean_lat, digits=3)) N")

    # m/s -> cm/s (Flare's figures are in cm/s)
    u = data.ve .* 100.0
    v = data.vn .* 100.0

    println("Computing multitaper rotary spectrum: K=$K_TAPERS tapers, nw=$NW, detrend=$DETREND ...")
    rse = rotary_spectrum(u, v; dt_hours=DT_HOURS, detrend=DETREND, nw=NW,
                           ntapers=K_TAPERS, ci=true, confidence=0.95, ftest=true)
    println("  ntapers actually used = $(rse.params.ntapers)")

    println("Computing raw (boxcar) periodogram for grey overlay ...")
    freqs_pg, S_ccw_pg, S_cw_pg = rotary_periodogram(u, v; dt_hours=DT_HOURS, detrend=DETREND)
    @assert freqs_pg ≈ rse.freq "periodogram and multitaper frequency grids must match"

    f0_cph = inertial_frequency(mean_lat)   # cycles/hour, local planetary inertial frequency
    @printf("Local inertial frequency f0 = %.6f cycles/hour (period %.2f h = %.2f d)\n",
            f0_cph, 1 / f0_cph, 1 / f0_cph / 24)

    # ── Save spectrum data for Fig. 3's Kunze cross-check ──────────────────
    mkpath(dirname(OUT_CSV))
    open(OUT_CSV, "w") do io
        println(io, "freq_cph,S_ccw,S_cw,periodogram_ccw,periodogram_cw")
        for i in eachindex(rse.freq)
            @printf(io, "%.10f,%.8e,%.8e,%.8e,%.8e\n",
                    rse.freq[i], rse.S_ccw[i], rse.S_cw[i], S_ccw_pg[i], S_cw_pg[i])
        end
    end
    println("Saved spectrum data: $OUT_CSV")
    # Also drop f0 / mean_lat for Fig. 3 to reuse without recomputation
    open(joinpath(dirname(OUT_CSV), "flare_fig2_f0.txt"), "w") do io
        @printf(io, "mean_lat=%.6f\nf0_cph=%.10f\n", mean_lat, f0_cph)
    end

    # ── Plot: jLab style (Flare Fig. 2) — two side-by-side panels, one per
    # rotary sense, sharing a y-axis; periodogram (grey) and multitaper
    # (colored) on the SAME axes at the SAME absolute scale (the point of
    # today's normalization fix: they should now share an envelope, not
    # sit ~n apart); frequency in rad/hour (jLab's own convention, not
    # cycles/hour); the negative-rotary panel's x-axis is reversed so both
    # panels' low-frequency ends meet at the shared center, matching the
    # real Flare Fig. 2 layout. See feedback_plotting_standards.md rule 7.
    freq_radph = 2π .* rse.freq                 # same grid, both panels
    # S(f_cyc)*df_cyc = S(f_rad)*df_rad, df_rad = 2π*df_cyc -> S_rad = S_cyc/2π
    to_radpsd(S) = S ./ (2π)
    S_ccw_rad, S_cw_rad = to_radpsd(rse.S_ccw), to_radpsd(rse.S_cw)
    S_ccw_pg_rad, S_cw_pg_rad = to_radpsd(S_ccw_pg), to_radpsd(S_cw_pg)
    f0_radph = 2π * f0_cph

    fig = Figure(size=(1000, 550), fontsize=14)
    common = (; yscale=log10, xscale=log10,
                titlesize=15, xlabelsize=13, ylabelsize=13, xticklabelsize=11, yticklabelsize=11)

    ax_neg = Axis(fig[1, 1]; common..., xreversed=true,
        xlabel="Frequency [rad/hour]", ylabel="Power Spectral Density [cm²·s⁻²·hour·rad⁻¹]",
        title="Negative Rotary Spectrum")
    lines!(ax_neg, freq_radph, max.(S_cw_pg_rad, 1e-12); color=(:gray60, 0.7), linewidth=1.0)
    lines!(ax_neg, freq_radph, max.(S_cw_rad, 1e-12); color=:steelblue, linewidth=2.0)
    vlines!(ax_neg, [f0_radph]; color=:black, linestyle=:dot, linewidth=1.5)
    i_pk = argmax(S_cw_rad); vlines!(ax_neg, [freq_radph[i_pk]]; color=:black, linewidth=1.2)

    ax_pos = Axis(fig[1, 2]; common...,
        xlabel="Frequency [rad/hour]",
        title="Positive Rotary Spectrum")
    lines!(ax_pos, freq_radph, max.(S_ccw_pg_rad, 1e-12); color=(:gray60, 0.7), linewidth=1.0)
    lines!(ax_pos, freq_radph, max.(S_ccw_rad, 1e-12); color=:steelblue, linewidth=2.0)
    j_pk = argmax(S_ccw_rad); vlines!(ax_pos, [freq_radph[j_pk]]; color=:magenta, linewidth=1.5)

    linkyaxes!(ax_neg, ax_pos)
    Label(fig[0, 1:2], "GDP 44000 (2005) — rotary multitaper spectrum, K=$K_TAPERS " *
                       "(Flare Fig. 2 reproduction)"; fontsize=16, font=:bold)

    mkpath(dirname(OUT_PNG))
    save(OUT_PNG, fig; px_per_unit=DPI)
    println("Saved: $OUT_PNG")
end

main()
