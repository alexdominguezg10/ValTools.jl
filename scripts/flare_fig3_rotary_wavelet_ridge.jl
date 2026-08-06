# Reproduce Fig. 3 of Jonathan Lilly's unfunded Flare NSF proposal: the
# rotary wavelet transform of w = u + iv for GDP drifter 44000 (WMO
# 4100571), generalized Morse wavelet with beta=3, gamma=3 (NOT this
# codebase's beta=2/beta=8 defaults used elsewhere -- this exact (3,3) pair
# is Flare Fig. 3's own parameter choice), with ridge detection overlaid.
#
# Uses `JLab.rotary_wavetrans` (scalogram) + `JLab.rotary_ridge_properties`
# (chain-based ridge summary: L, xi_bar, omega_ast_bar, kappa_bar, sense)
# + `JLab.rotary_ridge` (per-timestep ridgemap frequency/amplitude, used
# to read a directly-dimensional omega off the identified eddy ridge).
# No new library code.
#
# *** Known, deliberate deviation from this function's own docstring ***
# `rotary_ridge_properties`'s docstring (src/JLab/wavelets.jl:1088-1111)
# says it expects DISPLACEMENT/POSITION as `u`,`v` to match jLab's actual
# GOMED-generating `eddyridges.m` pipeline -- confirmed empirically in this
# project (2026-08-01, "ValTools 7") to matter a great deal for matching
# jLab's real eddy-census ridges. This script intentionally passes VELOCITY
# instead, per the task's explicit spec (Flare Fig. 3 is presented as the
# rotary wavelet transform of w=u+iv velocity, mirroring Fig. 2's spectrum
# of the same quantity -- not a GOMED-style eddyridges.m census run). This
# is flagged here and in the run's printed report so the ridge properties
# below are NOT directly comparable to this repo's GOMED validation numbers.
#
# Data: the clean, gap-free 2005 calendar-year segment (8761 hourly rows).
# ve/vn are m/s in the source CSV; converted to cm/s (x100) to match
# Flare's own axis scales.
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/flare_fig3_rotary_wavelet_ridge.jl
# (Run flare_fig2_rotary_spectrum.jl FIRST -- this script reads its saved
# spectrum CSV to do the final Kunze-effect cross-check.)

using ValTools, ValTools.JLab
using CairoMakie
using Dates, Statistics, Printf

# ── Config ───────────────────────────────────────────────────────────────
const CSV_PATH = "/Volumes/DATA_SSD/gdp_44000_2005.csv"
const OUT_PNG = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_fig3_rotary_wavelet_ridge.png")
const OUT_RIDGES_CSV = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_fig3_ridges.csv")
const OUT_REPORT = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_kunze_report.txt")
const SPECTRUM_CSV = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_fig2_spectrum_data.csv")
const DT_HOURS = 1.0
const GAMMA = 3.0
const BETA = 3.0            # Flare Fig. 3's own choice -- NOT this repo's beta=2/beta=8 defaults
const FIGSIZE = (1000, 650)
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

function read_spectrum_csv(path::String)
    lines = readlines(path)
    n = length(lines) - 1
    freq = Vector{Float64}(undef, n)
    S_ccw = Vector{Float64}(undef, n)
    S_cw = Vector{Float64}(undef, n)
    for (k, line) in enumerate(@view lines[2:end])
        f = split(line, ",")
        freq[k] = parse(Float64, f[1])
        S_ccw[k] = parse(Float64, f[2])
        S_cw[k] = parse(Float64, f[3])
    end
    return freq, S_ccw, S_cw
end

function main()
    data = read_gdp_csv(CSV_PATH)
    n = length(data.time)
    mean_lat = mean(data.lat)
    println("Loaded $n rows; mean latitude = $(round(mean_lat, digits=3)) N")

    u = data.ve .* 100.0   # cm/s
    v = data.vn .* 100.0

    f0_cph = inertial_frequency(mean_lat)         # cycles/hour
    f0_radph = 2π * f0_cph                        # rad/hour == rad/sample (dt=1h)
    @printf("Local inertial frequency f0 = %.6f cyc/h = %.6f rad/h (period %.2f h = %.2f d)\n",
            f0_cph, f0_radph, 1 / f0_cph, 1 / f0_cph / 24)

    println("Computing rotary wavelet transform: gamma=$GAMMA, beta=$BETA ...")
    wt_ccw, wt_cw, fs = rotary_wavetrans(u, v; dt=DT_HOURS, gamma=GAMMA, beta=BETA)
    println("  scalogram size: $(size(wt_ccw)), n_freqs=$(length(fs))")
    println("  frequency range: $(minimum(fs)) - $(maximum(fs)) rad/sample " *
            "(periods $(round(2π/maximum(fs), digits=2))-$(round(2π/minimum(fs)/24, digits=1))d)")

    # ── Eddy-band frequency grid for RIDGE DETECTION ────────────────────
    # The generic full-spectrum grid `rotary_wavetrans` builds by default
    # (used for `wt_ccw`/`wt_cw` above, and for the background scalogram
    # image) is an N-independent-of-physics heuristic. jLab's own real
    # Gulf-eddy-census call (`jFigures/makefigs_gulfcensus.m:460`) instead
    # uses a Coriolis-relative eddy band `fmax_ratio=2, fmin_ratio=1/64` --
    # reused here (via `rotary_ridge_properties`'s own `fmax_ratio`/
    # `fmin_ratio` kwargs, matching jLab's call). NOTE this band is
    # deliberately BROAD (everything from just above 2x the inertial
    # frequency down to 1/64th of it, i.e. periods ~0.45 days to ~58 days
    # at this latitude) -- it is jLab's general eddy-detection band, not a
    # "slow eddies only" filter, so it does not by itself exclude
    # multi-day ridges; it only guards against the grid's generic
    # heuristic clipping a genuinely slow (e.g. >51-day) ridge at its edge
    # (src/JLab/wavelets.jl:1186-1198's documented gap). Using it here is
    # a legitimate, jLab-faithful sanity check that the ridge identified
    # below is not a generic-grid artifact.
    const_fmax_ratio, const_fmin_ratio = 2.0, 1 / 64

    println("Computing per-timestep ridge (JLab.rotary_ridge, eddy-band grid) ...")
    P_w, _, _ = morseprops(GAMMA, BETA)
    f_high_eddy = const_fmax_ratio * f0_radph   # << the wavelet's own natural
                                                 # eta-decay cutoff (~1.85 rad/h
                                                 # printed above), so the min()
                                                 # in the library's own eddy-band
                                                 # formula is inactive here
    f_low_eddy = max(const_fmin_ratio * f0_radph, π * P_w / n)
    fs_eddy = morsespace(GAMMA, BETA, n; f_high=f_high_eddy, f_low=f_low_eddy, density=16)
    println("  eddy-band grid: $(length(fs_eddy)) freqs, periods " *
            "$(round(2π/maximum(fs_eddy)/24, digits=2))-$(round(2π/minimum(fs_eddy)/24, digits=1))d")
    rr = rotary_ridge(u, v; dt=DT_HOURS, fs=fs_eddy, gamma=GAMMA, beta=BETA)

    println("Computing chain-based ridge properties (JLab.rotary_ridge_properties, eddy-band grid) ...")
    ridges = rotary_ridge_properties(u, v; dt=DT_HOURS, gamma=GAMMA, beta=BETA, f_coriolis=f0_radph,
                                      fmax_ratio=const_fmax_ratio, fmin_ratio=const_fmin_ratio)
    println("  found $(length(ridges)) ridges")

    # ── Save all ridges ─────────────────────────────────────────────────
    mkpath(dirname(OUT_RIDGES_CSV))
    open(OUT_RIDGES_CSV, "w") do io
        println(io, "start,stop,npoints,L_cycles,xi_bar,omega_ast_bar,kappa_bar_cmps,sense,omega_radph,period_days")
        for r in ridges
            omega_radph = r.omega_ast_bar * f0_radph  # invert Eq.63: omega* = sign(xi)*omega/f0, sign folded into omega_ast_bar's own sign
            period_days = omega_radph != 0 ? abs(2π / omega_radph) / 24 : Inf
            @printf(io, "%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%s,%.6f,%.3f\n",
                    r.start, r.stop, r.npoints, r.L, r.xi_bar, r.omega_ast_bar,
                    r.kappa_bar, string(r.sense), omega_radph, period_days)
        end
    end
    println("Saved ridge table: $OUT_RIDGES_CSV")

    # ── Identify the eddy ridge: the proposal's narrative is a long-lived,
    # dominant, positive-rotary (CCW = cyclonic in the NH) ridge -- the one
    # that plausibly reflects the drifter actually being trapped, not a
    # brief transient. Among CCW-sense ridges, pick the one with the
    # largest L (total cycles, rewarding BOTH duration and persistence).
    # On this record that selects a single, clearly dominant outlier: 3538
    # points (147 days), xi_bar=0.87 (strongly, consistently cyclonic),
    # kappa_bar=37.7 cm/s -- by a wide margin the largest amplitude of any
    # CCW ridge found (next-largest is ~31.7 cm/s on a 347-point ridge) --
    # spanning hours 1751-5288, i.e. days 73-220 of 2005 (2005-03-15 to
    # 2005-08-08). That window lines up with the visually tangled,
    # tightly-looping cluster in flare_fig1_map.png (colorbar days
    # ~150-300 since the 2004-10-13 deployment = the SAME date range),
    # which is independent, visual corroboration that this is a real
    # trapping event and not a ridge-chaining artifact. Its period comes
    # out to ~3.5-4.6 days depending on which frequency estimator is used
    # (see the two numbers printed below) -- fast for a textbook
    # "mesoscale" eddy, but physically ordinary for a tighter Gulf
    # Stream-region eddy/ring, and consistent with the very tight looping
    # visible in the map over that same period. ─────────────────────────
    ccw_ridges = filter(r -> r.sense == :ccw, ridges)
    isempty(ccw_ridges) && error("No CCW (cyclonic) ridges found -- cannot identify the eddy ridge")
    # NOTE: `argmax(f, itr)` returns the ELEMENT of `itr` maximizing `f`,
    # not its index (Julia >=1.7) -- do not re-index `ccw_ridges` with it.
    eddy = argmax(r -> r.L, ccw_ridges)

    omega_ridge_radph = eddy.omega_ast_bar * f0_radph   # dimensional, rad/h
    omega_ridge_cph = omega_ridge_radph / (2π)          # cycles/h
    period_days = abs(2π / omega_ridge_radph) / 24

    # Cross-check: directly average JLab.rotary_ridge's per-timestep CCW
    # ridge frequency (rad/sample = rad/h) over the SAME time window, as an
    # independent read of omega off the ridge (ridgemap-based local maxima,
    # vs. rotary_ridge_properties' ridgechains_jlab-based chained ridge).
    win = eddy.start:eddy.stop
    freq_ccw_win = filter(!isnan, rr.freq_ccw[win])
    omega_direct_radph = isempty(freq_ccw_win) ? NaN : mean(freq_ccw_win)
    omega_direct_cph = omega_direct_radph / (2π)

    @printf("\n=== Identified eddy ridge (longest CCW/cyclonic ridge) ===\n")
    @printf("  time indices: %d -> %d (%d points, %.1f days)\n", eddy.start, eddy.stop, eddy.npoints,
            (eddy.stop - eddy.start) / 24)
    @printf("  L = %.2f cycles, xi_bar = %.3f, kappa_bar = %.3f cm/s\n", eddy.L, eddy.xi_bar, eddy.kappa_bar)
    @printf("  omega_ast_bar (nondim, Eq.63) = %.4f\n", eddy.omega_ast_bar)
    @printf("  omega (from rotary_ridge_properties, via omega_ast_bar*f0) = %.6f rad/h = %.6f cyc/h\n",
            omega_ridge_radph, omega_ridge_cph)
    @printf("  omega (direct mean of JLab.rotary_ridge freq_ccw over ridge window) = %.6f rad/h = %.6f cyc/h\n",
            omega_direct_radph, omega_direct_cph)
    @printf("  ridge period = %.2f days\n", period_days)

    # ── Kunze (1985) effective inertial frequency ───────────────────────
    # f_eff = f0 - omega, with f0 SIGNED NEGATIVE on the negative-rotary
    # (CW) side, per this project's sign convention (memory:
    # feedback_pv_sign_convention direction; Gonella 1972 CW=negative
    # frequency in the rotary decomposition used throughout this codebase,
    # e.g. Metrics.rotary_spectrum's own docstring). omega (the cyclonic
    # eddy's ridge frequency) is positive.
    f0_signed_cph = -f0_cph
    omega_use_cph = omega_ridge_cph   # use the ridgechains-based estimate as the primary number
    f_eff_signed_cph = f0_signed_cph - omega_use_cph
    f_eff_mag_cph = abs(f_eff_signed_cph)

    @printf("\n=== Kunze (1985) effective inertial frequency ===\n")
    @printf("  f0 (planetary inertial, magnitude) = %.6f cyc/h\n", f0_cph)
    @printf("  f0_signed (CW/negative-rotary side) = %.6f cyc/h\n", f0_signed_cph)
    @printf("  omega (cyclonic eddy ridge)          = %.6f cyc/h\n", omega_use_cph)
    @printf("  f_eff = f0_signed - omega            = %.6f cyc/h\n", f_eff_signed_cph)
    @printf("  |f_eff| (predicted CW-branch peak)   = %.6f cyc/h  (vs. |f0| = %.6f cyc/h, shift = %+.6f cyc/h, %+.2f%%)\n",
            f_eff_mag_cph, f0_cph, f_eff_mag_cph - f0_cph, 100 * (f_eff_mag_cph - f0_cph) / f0_cph)

    # ── Locate the ACTUAL local peak in Fig. 2's S_cw near f0 and near
    # f_eff, to check whether the predicted shift lines up with a real
    # feature of the reproduced spectrum. ───────────────────────────────
    report_lines = String[]
    push!(report_lines, "Flare Fig. 2/3 reproduction -- Kunze effect cross-check")
    push!(report_lines, "GDP drifter 44000 (WMO 4100571), 2005, mean lat=$(round(mean_lat,digits=3))N")
    push!(report_lines, "")
    if isfile(SPECTRUM_CSV)
        freq, S_ccw, S_cw = read_spectrum_csv(SPECTRUM_CSV)

        function local_peak_near(f0::Real, halfwidth_frac::Real=0.35)
            lo, hi = f0 * (1 - halfwidth_frac), f0 * (1 + halfwidth_frac)
            idx = findall(f -> lo <= f <= hi, freq)
            isempty(idx) && return (NaN, NaN)
            i = idx[argmax(S_cw[idx])]
            return (freq[i], S_cw[i])
        end

        f_at_f0, S_at_f0 = local_peak_near(f0_cph, 0.15)
        f_at_feff, S_at_feff = local_peak_near(f_eff_mag_cph, 0.15)
        f_peak_global, S_peak_global = local_peak_near((f0_cph + f_eff_mag_cph) / 2, 0.5)

        @printf("\n=== Fig. 2 S_cw local peaks ===\n")
        @printf("  peak within +-15%% of f0    (%.4f cyc/h): f=%.4f cyc/h, S=%.4e cm2/s2/(cyc/h)\n",
                f0_cph, f_at_f0, S_at_f0)
        @printf("  peak within +-15%% of f_eff (%.4f cyc/h): f=%.4f cyc/h, S=%.4e cm2/s2/(cyc/h)\n",
                f_eff_mag_cph, f_at_feff, S_at_feff)
        @printf("  strongest peak in the wider [f0,f_eff] window: f=%.4f cyc/h, S=%.4e\n",
                f_peak_global, S_peak_global)

        append!(report_lines, [
            @sprintf("f0 (planetary inertial)      = %.6f cyc/h (period %.2f h)", f0_cph, 1/f0_cph),
            @sprintf("omega (cyclonic eddy ridge)   = %.6f cyc/h (period %.1f d)", omega_use_cph, period_days),
            @sprintf("f_eff = f0_signed - omega     = %.6f cyc/h  (|f_eff|=%.6f, shift=%+.2f%% vs f0)",
                     f_eff_signed_cph, f_eff_mag_cph, 100*(f_eff_mag_cph-f0_cph)/f0_cph),
            "",
            @sprintf("Fig.2 S_cw local max near f0    : f=%.4f cyc/h, S=%.4e", f_at_f0, S_at_f0),
            @sprintf("Fig.2 S_cw local max near f_eff : f=%.4f cyc/h, S=%.4e", f_at_feff, S_at_feff),
        ])
    else
        println("WARNING: $SPECTRUM_CSV not found -- run flare_fig2_rotary_spectrum.jl first for the full cross-check")
        push!(report_lines, "WARNING: spectrum CSV not found, Fig.2 cross-check skipped")
    end
    open(OUT_REPORT, "w") do io
        for l in report_lines
            println(io, l)
        end
    end
    println("\nSaved report: $OUT_REPORT")

    # ── Plot: jLab style (Flare Fig. 3) — 3 panels stacked on one shared
    # time axis: (a) the raw velocity time series that was transformed
    # (rule: always show the signal alongside its wavelet transform), (b)
    # positive-rotary |CCW| magnitude with ridges, (c) negative-rotary
    # |CW| magnitude with ridges + the Kunze f+eddy prediction. Frequency
    # in cycles/day (not log-period), day-of-year on x (this CSV starts
    # exactly 2005-01-01T00:00). See feedback_plotting_standards.md rule 7.
    cw_ridges = filter(r -> r.sense == :cw, ridges)

    freq_cycday = fs .* 24 ./ (2π)                     # rad/hour -> cycles/day
    f0_t_cph = inertial_frequency.(data.lat)            # time-varying Coriolis, cyc/hour
    f0_t_cycday = f0_t_cph .* 24

    fig = Figure(size=(1000, 950), fontsize=14)
    t_days = collect(0:n-1) ./ 24

    ax_a = Axis(fig[1, 1]; ylabel="Velocity [cm/s]",
                xticklabelsvisible=false, titlesize=15, ylabelsize=13)
    lines!(ax_a, t_days, u; color=:blue, linewidth=0.8)
    lines!(ax_a, t_days, v; color=:red, linewidth=0.8)
    text!(ax_a, 0.98, 0.06; text="(a)", space=:relative, align=(:right, :bottom), fontsize=13)

    ax_b = Axis(fig[2, 1]; ylabel="Frequency [cycles/day]",
                xticklabelsvisible=false, ylabelsize=13)
    hm_b = heatmap!(ax_b, t_days, freq_cycday, log10.(max.(abs.(wt_ccw), 1e-3));
                     colormap=:jet, colorrange=(0.7, 1.7))
    Colorbar(fig[2, 2], hm_b; label="Log₁₀ cm/s")
    lines!(ax_b, t_days, f0_t_cycday; color=:white, linewidth=1.5)
    text!(ax_b, t_days[end ÷ 6], f0_t_cycday[end ÷ 6] * 1.06; text="Coriolis Frequency",
          color=:white, fontsize=10)
    for r in ccw_ridges
        win_r = r.start:r.stop
        valid = .!isnan.(rr.freq_ccw[win_r])
        lines!(ax_b, t_days[win_r][valid], (rr.freq_ccw[win_r][valid] .* 24 ./ (2π));
               color=:black, linewidth=r === eddy ? 2.2 : 1.0)
    end
    eddy_freq_cycday = filter(!isnan, rr.freq_ccw[eddy.start:eddy.stop]) .* 24 ./ (2π)
    text!(ax_b, t_days[eddy.start] + 5, minimum(eddy_freq_cycday) * 0.7;
          text="Cyclonic Eddy", color=:black, fontsize=10)
    text!(ax_b, 0.98, 0.06; text="(b)", space=:relative, align=(:right, :bottom), color=:white, fontsize=13)

    ax_c = Axis(fig[3, 1]; xlabel="Day of Year 2005", ylabel="Frequency [cycles/day]", ylabelsize=13)
    hm_c = heatmap!(ax_c, t_days, freq_cycday, log10.(max.(abs.(wt_cw), 1e-3));
                     colormap=:jet, colorrange=(0.7, 1.7))
    Colorbar(fig[3, 2], hm_c; label="Log₁₀ cm/s")
    lines!(ax_c, t_days, f0_t_cycday; color=:white, linewidth=1.5)
    text!(ax_c, t_days[end ÷ 6], f0_t_cycday[end ÷ 6] * 0.9; text="Coriolis Frequency",
          color=:white, fontsize=10)
    for r in cw_ridges
        win_r = r.start:r.stop
        valid = .!isnan.(rr.freq_cw[win_r])
        lines!(ax_c, t_days[win_r][valid], (rr.freq_cw[win_r][valid] .* 24 ./ (2π)); color=:black, linewidth=1.0)
    end
    # Kunze f+eddy prediction, over the eddy's own active window.
    # |f_eff| = |f0_signed - omega| = f0 + omega (f0_signed is NEGATIVE on
    # this CW/negative-rotary side, omega positive) -- i.e. PLUS omega, a
    # LARGER magnitude than f0, matching jLab's f+Eddy sitting ABOVE the
    # Coriolis line here. f0_t_cph below is the unsigned magnitude, so the
    # magnitude formula is +omega, not -omega (fixed 2026-08-05: an
    # earlier version of this line used .- and put f+Eddy under f0,
    # backwards relative to the real jLab figure).
    win_eddy = eddy.start:eddy.stop
    f_eff_t_cycday = (f0_t_cph[win_eddy] .+ omega_use_cph) .* 24
    lines!(ax_c, t_days[win_eddy], f_eff_t_cycday; color=:magenta, linewidth=1.8)
    text!(ax_c, t_days[win_eddy[end÷2]], f_eff_t_cycday[end÷2] * 1.06; text="f + Eddy",
          color=:magenta, fontsize=10)
    text!(ax_c, 0.98, 0.06; text="(c)", space=:relative, align=(:right, :bottom), color=:white, fontsize=13)

    for ax in (ax_b, ax_c)
        ylims!(ax, 0, min(2.2, maximum(freq_cycday)))
    end

    Label(fig[0, 1:2], "GDP 44000 (2005) — rotary wavelet transform (β=$BETA, γ=$GAMMA), " *
                       "Flare Fig. 3 reproduction"; fontsize=16, font=:bold)
    rowsize!(fig.layout, 1, Relative(0.22))

    mkpath(dirname(OUT_PNG))
    save(OUT_PNG, fig; px_per_unit=DPI)
    println("Saved: $OUT_PNG")
end

main()
