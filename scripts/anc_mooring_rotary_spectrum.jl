# Rotary spectra at TWO real depths from the SAME mooring (ARE site, CANEK
# program): LR75DW long-ranger (~672m, mid-water column -- not surface,
# despite the file's near-instrument bin range of 2-24m, which is relative
# to the instrument's own ~670m position) and WH600DW workhorse (~1934m,
# deep). Both instruments share one mooring line, so this is a genuine
# depth comparison at the same lat/lon, not a site-to-site comparison.
#
# Usage: julia --project=envs/cpu scripts/anc_mooring_rotary_spectrum.jl
# (envs/cpu needs MAT and Multitaper added: `Pkg.add(["MAT","Multitaper"])`)

using ValTools, ValTools.JLab, MAT, Multitaper, CairoMakie
using Statistics, Printf

const DIR = "/Volumes/DATA_SSD/ADominguez/DATA/ANCLAJES/ANC_GoMW_deep"
const LR75_FILE = joinpath(DIR, "ARE-T2000-LR75DW-NS10885-Z715-INS14-REC18.mat")
const WH600_FILE = joinpath(DIR, "ARE-T2000-WH600DW-NS10732-Z1980-INS14-REC18.mat")
const DT_HOURS = 0.5   # confirmed exactly regular at load time, both instruments

# Band-limited peak search around f0 -- same pattern as
# flare_fig3_rotary_wavelet_ridge.jl's local_peak_near, generalized to
# both rotary senses. A naive global argmax finds the wrong thing here:
# the low-frequency mesoscale/sub-inertial background is much more
# energetic in absolute terms than the (real, narrower) inertial peak
# riding on top of it. halfwidth_frac=0.35 brackets the near-inertial
# band without reaching into the semidiurnal tide or the background.
function local_peak_near(freq::Vector{Float64}, S::Vector{Float64}, f0::Real; halfwidth_frac::Real=0.35)
    lo, hi = f0 * (1 - halfwidth_frac), f0 * (1 + halfwidth_frac)
    idx = findall(f -> lo <= f <= hi, freq)
    isempty(idx) && return (freq=NaN, power=NaN, n_bins=0)
    i = idx[argmax(S[idx])]
    return (freq=freq[i], power=S[i], n_bins=length(idx))
end

function load_and_spectrum(path; bin_idx=1, label="")
    r = ANCMooringLoader(path)
    p = anc_mooring_profiles(r)
    u = p.u[:, bin_idx] .* 100.0   # m/s -> cm/s
    v = p.v[:, bin_idx] .* 100.0
    finite = isfinite.(u) .& isfinite.(v)
    depth_m = p.depths[bin_idx]

    @printf("%s: n=%d (%d finite), depth~%.1fm, lat=%.4f lon=%.4f\n",
            label, length(u), count(finite), depth_m, p.lat, p.lon)
    @printf("  Record: %s to %s\n", p.time[1], p.time[end])

    rse = rotary_spectrum(u[finite], v[finite]; dt_hours=DT_HOURS, detrend="linear",
                          nw=8.0, ntapers=15, ci=false, ftest=false)
    f0_cph = inertial_frequency(p.lat)
    pk_ccw = local_peak_near(rse.freq, rse.S_ccw, f0_cph)
    pk_cw = local_peak_near(rse.freq, rse.S_cw, f0_cph)

    @printf("  f0=%.4f cyc/h. Band-limited peak (±35%%, %d bins): CCW f=%.4f (%.1f%% off), CW f=%.4f (%.1f%% off)\n",
            f0_cph, pk_cw.n_bins, pk_ccw.freq, 100 * (pk_ccw.freq - f0_cph) / f0_cph,
            pk_cw.freq, 100 * (pk_cw.freq - f0_cph) / f0_cph)
    @printf("  CW/CCW power ratio at band peak: %.1fx\n\n", pk_cw.power / pk_ccw.power)

    return (rse=rse, f0_cph=f0_cph, pk_ccw=pk_ccw, pk_cw=pk_cw, depth=depth_m, label=label)
end

mid = load_and_spectrum(LR75_FILE; bin_idx=1, label="Mid-depth (LR75DW)")
deep = load_and_spectrum(WH600_FILE; bin_idx=1, label="Deep (WH600DW)")

fig = Figure(size=(1050, 480))
for (i, r) in enumerate((mid, deep))
    ax = Axis(fig[1, i]; xlabel="Frequency [cycles/hour]", ylabel="Power [cm²/s²/(cyc/h)]",
              xscale=log10, yscale=log10,
              title="$(r.label), depth~$(round(r.depth, digits=0))m")
    lines!(ax, r.rse.freq, r.rse.S_ccw; color=:steelblue, label="CCW")
    lines!(ax, r.rse.freq, r.rse.S_cw; color=:firebrick, label="CW")
    vlines!(ax, [r.f0_cph]; color=:black, linestyle=:dash, label="f₀ (inertial)")
    scatter!(ax, [r.pk_cw.freq], [r.pk_cw.power]; color=:black, marker=:circle,
             markersize=12, strokewidth=1.5, strokecolor=:white, label="CW peak (±35% band)")
    i == 1 && axislegend(ax; position=:lb, labelsize=10)
end
Label(fig[0, 1:2], "ARE mooring (CANEK) — rotary spectra at two depths, same mooring line"; fontsize=15, font=:bold)

outpath = joinpath(@__DIR__, "..", "results", "anc_mooring_rotary_spectrum.png")
mkpath(dirname(outpath))
save(outpath, fig)
println("Saved: $outpath")

# ── WH600DW bin sweep: this instrument sits at ~1934m and is
# downward-looking, and per the user this deployment is measuring in the
# BOTTOM boundary layer -- its 14 bins (5-18m relative range, so ~1939-
# 1952m absolute) likely span from just below the instrument down toward
# the seafloor. Sweep every bin to see whether the near-inertial signal
# (and the general spectral shape/energy level) changes approaching the
# bottom -- e.g. bottom friction broadening/damping the inertial peak, or
# added turbulent energy at high frequencies.
r_wh600 = ANCMooringLoader(WH600_FILE)
n_bins_wh600 = size(r_wh600.u, 2)
println("WH600DW bin sweep: $n_bins_wh600 bins, relative range $(r_wh600.bins[1])-$(r_wh600.bins[end])m, " *
        "absolute depth $(round(minimum(r_wh600.depths),digits=1))-$(round(maximum(r_wh600.depths),digits=1))m\n")

bin_results = [load_and_spectrum(WH600_FILE; bin_idx=b, label="WH600DW bin $b") for b in 1:n_bins_wh600]

depths_wh600 = [br.depth for br in bin_results]
ratios_wh600 = [br.pk_cw.power / br.pk_ccw.power for br in bin_results]
offsets_wh600 = [100 * (br.pk_cw.freq - br.f0_cph) / br.f0_cph for br in bin_results]
totalpower_wh600 = [sum(br.rse.S_ccw) + sum(br.rse.S_cw) for br in bin_results]

fig2 = Figure(size=(1050, 420))

ax_ratio = Axis(fig2[1, 1]; xlabel="CW/CCW power ratio at band peak", ylabel="Depth [m]",
                yreversed=true, title="Near-inertial dominance vs. depth")
scatterlines!(ax_ratio, ratios_wh600, depths_wh600; color=:firebrick, markersize=10)

ax_offset = Axis(fig2[1, 2]; xlabel="CW peak offset from f₀ [%]", ylabel="Depth [m]",
                  yreversed=true, title="Peak frequency shift vs. depth")
scatterlines!(ax_offset, offsets_wh600, depths_wh600; color=:steelblue, markersize=10)
vlines!(ax_offset, [0.0]; color=:gray50, linestyle=:dash)

ax_total = Axis(fig2[1, 3]; xlabel="Total spectral power [cm²/s²]", ylabel="Depth [m]",
                 yreversed=true, title="Total energy vs. depth", xscale=log10)
scatterlines!(ax_total, totalpower_wh600, depths_wh600; color=:seagreen, markersize=10)

Label(fig2[0, 1:3], "ARE WH600DW bottom boundary layer — bin sweep (1934m instrument, bins toward seafloor)";
      fontsize=15, font=:bold)

outpath2 = joinpath(@__DIR__, "..", "results", "anc_mooring_wh600_bin_sweep.png")
save(outpath2, fig2)
println("\nSaved: $outpath2")

# ── Rotary coherence between the two instruments (LR75DW @672m vs
# WH600DW @1939m, same mooring line). The two records don't share an
# exact time base (30-min offset start, off-by-one length -- confirmed
# above: LR75 starts 11:00, WH600 starts 10:30), so align by exact
# timestamp intersection before calling rotary_coherence (which requires
# equal-length input). Coherence/phase are normalized (cross-power over
# sqrt of the two auto-powers), so unlike the raw spectra earlier, this
# quantity was never affected by the rotary_spectrum /n normalization bug
# -- it's scale-invariant by construction, not a coincidental match.
function align_by_time(t1, u1, v1, t2, u2, v2)
    common = sort(collect(intersect(Set(t1), Set(t2))))
    idx1 = Dict(t => i for (i, t) in enumerate(t1))
    idx2 = Dict(t => i for (i, t) in enumerate(t2))
    i1 = [idx1[t] for t in common]
    i2 = [idx2[t] for t in common]
    return common, u1[i1], v1[i1], u2[i2], v2[i2]
end

r_lr75 = ANCMooringLoader(LR75_FILE)
p_lr75 = anc_mooring_profiles(r_lr75)
p_wh600 = anc_mooring_profiles(r_wh600)

t_common, u1, v1, u2, v2 = align_by_time(p_lr75.time, p_lr75.u[:, 1] .* 100.0, p_lr75.v[:, 1] .* 100.0,
                                         p_wh600.time, p_wh600.u[:, 1] .* 100.0, p_wh600.v[:, 1] .* 100.0)
println("\nAligned $(length(t_common)) common timestamps (LR75 n=$(length(p_lr75.time)), " *
        "WH600 n=$(length(p_wh600.time))) for coherence: $(t_common[1]) to $(t_common[end])")

finite12 = isfinite.(u1) .& isfinite.(v1) .& isfinite.(u2) .& isfinite.(v2)
coh = rotary_coherence(u1[finite12], v1[finite12], u2[finite12], v2[finite12];
                       dt_hours=DT_HOURS, detrend="linear", nw=8.0, ntapers=15)

f0_common = inertial_frequency(r_lr75.lat)
coh_ccw_at_f0 = local_peak_near(coh.freq, coh.coh_ccw, f0_common)
coh_cw_at_f0 = local_peak_near(coh.freq, coh.coh_cw, f0_common)
@printf("\nRotary coherence LR75(672m) vs WH600(1939m): significance level (K=15, 95%%)=%.3f\n",
        coh.significance_level)
@printf("  Near f0: CCW coherence=%.3f, CW coherence=%.3f (>%.3f = significant)\n",
        coh_ccw_at_f0.power, coh_cw_at_f0.power, coh.significance_level)

fig3 = Figure(size=(1000, 420))
ax_coh = Axis(fig3[1, 1]; xlabel="Frequency [cycles/hour]", ylabel="Coherence² (0-1)",
              xscale=log10, title="Rotary coherence: LR75DW (672m) vs WH600DW (1939m)")
lines!(ax_coh, coh.freq, coh.coh_ccw; color=:steelblue, label="CCW")
lines!(ax_coh, coh.freq, coh.coh_cw; color=:firebrick, label="CW")
hlines!(ax_coh, [coh.significance_level]; color=:gray40, linestyle=:dot, label="95% significance")
vlines!(ax_coh, [f0_common]; color=:black, linestyle=:dash, label="f₀")
ylims!(ax_coh, 0, 1)
axislegend(ax_coh; position=:rt, labelsize=10)

ax_phase = Axis(fig3[1, 2]; xlabel="Frequency [cycles/hour]", ylabel="Phase [rad]",
                 xscale=log10, title="Phase (WH600 relative to LR75)")
lines!(ax_phase, coh.freq, coh.phase_ccw; color=:steelblue, label="CCW")
lines!(ax_phase, coh.freq, coh.phase_cw; color=:firebrick, label="CW")
vlines!(ax_phase, [f0_common]; color=:black, linestyle=:dash)
hlines!(ax_phase, [0.0]; color=:gray70, linestyle=:solid, linewidth=0.5)

outpath3 = joinpath(@__DIR__, "..", "results", "anc_mooring_coherence.png")
save(outpath3, fig3)
println("Saved: $outpath3")

# ── Rotary wavelet transforms + ridges, LR75DW (672m) and WH600DW (1939m),
# beta=3 gamma=3 (this project's established Flare-figure convention),
# subsampled to a MESOSCALE-focused setup:
#
# 1. Block-averaged (not just decimated) from 0.5h to 4h -- averaging
#    over each 8-sample window is a real anti-aliasing step (removes
#    tidal/high-frequency content properly before subsampling), unlike
#    picking every 8th point, which would let that same content alias
#    back down into the retained band.
# 2. Eddy-band frequency grid reused from flare_fig3_rotary_wavelet_ridge.jl
#    (fmax_ratio=2, fmin_ratio=1/64 relative to f0) -- at this f0 that
#    already reaches ~85-day periods, well into the mesoscale range, so
#    the same grid choice that worked for eddy detection on the drifter
#    works here too.
# 3. LOG-period y-axis, not linear cycles/day like the Flare figures --
#    Flare Fig. 3 used linear cycles/day because its whole point was
#    resolving fine structure NEAR f0; capturing mesoscale variability
#    (periods of days to weeks) needs log spacing instead, or the
#    mesoscale band gets squashed into an unreadable sliver near the
#    bottom of a linear axis.
# 4. rotary_ridge_properties on the same grid, mirroring the drifter
#    workflow, to find persistent mesoscale-band ridges (a mesoscale eddy
#    passing the mooring would leave exactly this kind of signature in a
#    moored record's Eulerian rotary spectrum).

function block_average(x::AbstractVector, factor::Int)
    n_out = length(x) ÷ factor
    return [mean(@view x[((i - 1) * factor + 1):(i * factor)]) for i in 1:n_out]
end

const SUBSAMPLE_FACTOR = 8   # 0.5h * 8 = 4h
const DT_SUB_HOURS = DT_HOURS * SUBSAMPLE_FACTOR

function compute_wavelet_and_ridges(u_native, v_native, time_native, lat, label, depth_m)
    u = block_average(u_native, SUBSAMPLE_FACTOR)
    v = block_average(v_native, SUBSAMPLE_FACTOR)
    n = length(u)
    println("Computing rotary wavelet + ridges for $label (n=$n at $(DT_SUB_HOURS)h after block-averaging) ...")

    f0_cph = inertial_frequency(lat)
    f0_radph = 2π * f0_cph
    P_w, _, _ = morseprops(3.0, 3.0)
    f_high = 2.0 * f0_radph
    f_low = max(f0_radph / 64, π * P_w / n)
    fs = morsespace(3.0, 3.0, n; f_high=f_high, f_low=f_low, density=16)
    period_days = 2π ./ fs ./ 24
    @printf("  eddy-band grid: %d freqs, periods %.2f-%.1f days\n", length(fs), minimum(period_days), maximum(period_days))

    wt_ccw, wt_cw, fs_out = rotary_wavetrans(u, v; dt=DT_SUB_HOURS, gamma=3.0, beta=3.0, fs=fs)
    ridges = rotary_ridge_properties(u, v; dt=DT_SUB_HOURS, gamma=3.0, beta=3.0, f_coriolis=f0_radph,
                                     fmax_ratio=2.0, fmin_ratio=1 / 64)
    rr = rotary_ridge(u, v; dt=DT_SUB_HOURS, fs=fs, gamma=3.0, beta=3.0)
    println("  found $(length(ridges)) ridge(s)")
    for r in sort(ridges; by=x -> -x.L)[1:min(5, length(ridges))]
        per_d = abs(2π / (r.omega_ast_bar * f0_radph)) / 24
        @printf("    L=%.2f cyc, sense=%s, xi_bar=%.2f, kappa_bar=%.2f cm/s, period=%.1fd, span=%.1fd\n",
                r.L, r.sense, r.xi_bar, r.kappa_bar, per_d, (r.stop - r.start) * DT_SUB_HOURS / 24)
    end

    t_days = collect(0:(n - 1)) .* (DT_SUB_HOURS / 24)
    mag_ccw = log10.(max.(abs.(wt_ccw), 1e-4))
    mag_cw = log10.(max.(abs.(wt_cw), 1e-4))
    crange = (quantile(vec(mag_ccw), 0.01), quantile(vec(mag_ccw), 0.99))

    fig = Figure(size=(1000, 750))
    ax_a = Axis(fig[1, 1]; ylabel="Velocity [cm/s]", xticklabelsvisible=false,
                title="$label, depth~$(round(depth_m, digits=0))m (4h block-averaged)")
    lines!(ax_a, t_days, u; color=:blue, linewidth=0.6)
    lines!(ax_a, t_days, v; color=:red, linewidth=0.6)

    for (row, (wt, mag, sense_label, ridge_sense, cbar_label)) in enumerate((
        (wt_ccw, mag_ccw, "CCW", :ccw, "Log₁₀ cm/s (CCW)"),
        (wt_cw, mag_cw, "CW", :cw, "Log₁₀ cm/s (CW)")))
        ax = Axis(fig[row + 1, 1]; ylabel="Period [days]", yscale=log10,
                  xticklabelsvisible=(row == 2), xlabel=(row == 2 ? "Time [days since record start]" : ""))
        hm = heatmap!(ax, t_days, period_days, mag; colormap=:viridis, colorrange=crange)
        Colorbar(fig[row + 1, 2], hm; label=cbar_label)
        hlines!(ax, [1 / f0_cph / 24]; color=:white, linestyle=:dash)
        for r in filter(x -> x.sense == ridge_sense, ridges)
            win = r.start:r.stop
            freq_field = ridge_sense == :ccw ? rr.freq_ccw : rr.freq_cw
            valid = .!isnan.(freq_field[win])
            per = abs.(2π ./ freq_field[win][valid]) ./ 24
            lines!(ax, t_days[win][valid], per; color=:black, linewidth=1.2)
        end
    end
    rowsize!(fig.layout, 1, Relative(0.2))
    return fig, ridges
end

fig_lr75_wave, ridges_lr75 = compute_wavelet_and_ridges(p_lr75.u[:, 1] .* 100.0, p_lr75.v[:, 1] .* 100.0,
                                                         p_lr75.time, r_lr75.lat, "LR75DW", p_lr75.depths[1])
save(joinpath(@__DIR__, "..", "results", "anc_mooring_wavelet_lr75.png"), fig_lr75_wave)
println("Saved: results/anc_mooring_wavelet_lr75.png\n")

fig_wh600_wave, ridges_wh600 = compute_wavelet_and_ridges(p_wh600.u[:, 1] .* 100.0, p_wh600.v[:, 1] .* 100.0,
                                                           p_wh600.time, r_wh600.lat, "WH600DW", p_wh600.depths[1])
save(joinpath(@__DIR__, "..", "results", "anc_mooring_wavelet_wh600.png"), fig_wh600_wave)
println("Saved: results/anc_mooring_wavelet_wh600.png")
