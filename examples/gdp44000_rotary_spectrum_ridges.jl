# # Case Study: Rotary Spectrum + Wavelet Ridges on a Real Drifter — Frequency-Shifting by an Eddy
#
# Every wavelet/ridge example elsewhere in this gallery uses synthetic data
# with a known ground truth. This one, like the GOMED case study, is real:
# GDP drifter 44000 (WMO 4100571), deployed off Cuba on 2004-10-13 — the
# drifter behind Figs. 1–3 of Jonathan Lilly's unfunded Flare NSF proposal
# for a unified cross-language signal-analysis toolbox. This example
# reproduces Flare Figs. 2 and 3 with `Metrics.rotary_spectrum` (multitaper,
# K=15) and `JLab.rotary_wavetrans`/`rotary_ridge_properties` (generalized
# Morse wavelet, β=3, γ=3) — no new library code, just the existing jLab
# port applied to a real record.
#
# **The physical story:** a drifter's velocity spectrum splits into
# counter-clockwise (CCW) and clockwise (CW) rotary components (Gonella,
# 1972). In the Northern Hemisphere, free near-inertial oscillations are
# CW and sit at the local Coriolis frequency `f₀`. But this drifter spends
# ~5 months (2005-03-15 to 2005-08-08) trapped in a cyclonic eddy, and
# Kunze (1985) predicts that trapping *shifts* the effective inertial
# frequency to `f_eff = f₀ − ω` (ω = the eddy's own rotation frequency,
# f₀ signed negative on the CW side) — a real, physically motivated
# departure from the textbook single-peak spectrum.
#
# **Why a bundled CSV instead of fetching live:** the drifter's full hourly
# 2005 record (8761 rows, complete, no gaps — ideal for spectral work) is
# small enough to ship directly; it's public NOAA AOML Global Drifter
# Program data via ERDDAP, not licensed/restricted like GOMED.
#
# **This example is also how two real bugs got found.** Building the
# original reproduction and crosschecking it against real jLab MATLAB
# output (`mspec.m`, `wavetrans.m`) — not just jLab's source, per this
# project's own working agreement — turned up two normalization bugs, both
# now fixed: `Metrics.rotary_spectrum` had an extra `/n` that made every
# power value `n`≈8761× too small (it double-counted the DPSS tapers' own
# unit-energy normalization), and `JLab.rotary_wavetrans`'s complex-input
# path was missing jLab's documented `1/√2` prefactor. The crosscheck
# scripts (`scripts/jlab_crosscheck_flare_fig23.{jl,m}`) are re-runnable if
# you want to reproduce that verification yourself.
#
# **Reference:** Gonella (1972), Deep Sea Res. 19; Kunze (1985), J. Phys.
# Oceanogr. 15; Elipot, Lumpkin & Prieto (2010), JGR 115, C09010.

# ## Load the drifter record

using ValTools, ValTools.JLab, Multitaper, CairoMakie
using Statistics, Printf

function read_drifter_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    col = Dict(name => i for (i, name) in enumerate(header))
    n = length(lines) - 2
    lat = Vector{Float64}(undef, n)
    ve = Vector{Float64}(undef, n)
    vn = Vector{Float64}(undef, n)
    for (k, line) in enumerate(@view lines[3:end])
        f = split(line, ",")
        lat[k] = parse(Float64, f[col["latitude"]])
        ve[k] = parse(Float64, f[col["ve"]])
        vn[k] = parse(Float64, f[col["vn"]])
    end
    return lat, ve, vn
end

lat, ve, vn = read_drifter_csv(joinpath(@__DIR__, "gdp44000_2005.csv"))
n = length(lat)
mean_lat = mean(lat)
u, v = ve .* 100.0, vn .* 100.0   # m/s -> cm/s

println("Loaded $n hourly rows, mean latitude $(round(mean_lat, digits=2))°N")

# ## Rotary multitaper spectrum (Flare Fig. 2)

f0_cph = inertial_frequency(mean_lat)
rse = rotary_spectrum(u, v; dt_hours=1.0, detrend="linear", nw=8.0, ntapers=15, ci=false, ftest=false)
peak_ccw = rse.freq[argmax(rse.S_ccw)]
peak_cw = rse.freq[argmax(rse.S_cw)]

@printf("Local inertial frequency f0 = %.4f cyc/h (period %.1f h)\n", f0_cph, 1 / f0_cph)
@printf("Spectral peaks: CCW at %.4f cyc/h, CW at %.4f cyc/h\n", peak_ccw, peak_cw)

# ## Rotary wavelet transform + ridges (Flare Fig. 3)

GAMMA, BETA = 3.0, 3.0
f0_radph = 2π * f0_cph
fmax_ratio, fmin_ratio = 2.0, 1 / 64
ridges = rotary_ridge_properties(u, v; dt=1.0, gamma=GAMMA, beta=BETA, f_coriolis=f0_radph,
                                  fmax_ratio=fmax_ratio, fmin_ratio=fmin_ratio)
wt_ccw, wt_cw, fs = rotary_wavetrans(u, v; dt=1.0, gamma=GAMMA, beta=BETA, fs=nothing, nv=8)
rr = rotary_ridge(u, v; dt=1.0, fs=fs, gamma=GAMMA, beta=BETA)   # per-timestep trace for the plot below

ccw_ridges = filter(r -> r.sense == :ccw, ridges)
eddy = argmax(r -> r.L, ccw_ridges)
omega_ridge_radph = eddy.omega_ast_bar * f0_radph
period_days = abs(2π / omega_ridge_radph) / 24

@printf("\nDominant CCW (cyclonic) ridge: %d points (%.1f days), xi_bar=%.3f, period=%.1f d\n",
        eddy.npoints, eddy.npoints / 24, eddy.xi_bar, period_days)

# Kunze (1985): f_eff = f0_signed - omega, f0 signed negative on the CW side
f_eff_mag_cph = abs(-f0_cph - omega_ridge_radph / (2π))
@printf("Kunze effective frequency: |f_eff| = %.4f cyc/h (vs |f0| = %.4f cyc/h, %+.1f%% shift)\n",
        f_eff_mag_cph, f0_cph, 100 * (f_eff_mag_cph - f0_cph) / f0_cph)

# ## Visualization

fig = Figure(size=(1100, 480))

ax1 = Axis(fig[1, 1]; xlabel="Frequency [cycles/hour]", ylabel="Power [cm²/s²/(cyc/h)]",
           xscale=log10, yscale=log10, title="Rotary multitaper spectrum (K=15)")
lines!(ax1, rse.freq, rse.S_ccw; color=:steelblue, label="CCW")
lines!(ax1, rse.freq, rse.S_cw; color=:firebrick, label="CW")
vlines!(ax1, [f0_cph]; color=:black, linestyle=:dash, label="f₀ (inertial)")
axislegend(ax1; position=:lb, labelsize=10)

ax2 = Axis(fig[1, 2]; xlabel="Time [days since 2005-01-01]", ylabel="Frequency [cycles/day]",
           title="Rotary wavelet |CCW| + ridges (β=3, γ=3)")
t_days = collect(0:(n - 1)) ./ 24
freq_cycday = fs .* 24 ./ (2π)
hm = heatmap!(ax2, t_days, freq_cycday, log10.(max.(abs.(wt_ccw), 1e-3)); colormap=:viridis)
Colorbar(fig[1, 3], hm; label="Log₁₀ cm/s")
for r in ccw_ridges
    win = r.start:r.stop
    valid = .!isnan.(rr.freq_ccw[win])
    lines!(ax2, t_days[win][valid], (rr.freq_ccw[win][valid] .* 24 ./ (2π));
           color=:black, linewidth=(r === eddy ? 2.2 : 1.0))
end
ylims!(ax2, 0, min(2.2, maximum(freq_cycday)))

save(joinpath(@__DIR__, "gdp44000_rotary_spectrum_ridges.png"), fig)
println("\nSaved gdp44000_rotary_spectrum_ridges.png")

# ## Verification
#
# Ground-truth checks: the record matches the documented "complete, no
# gaps" 2005 GDP 44000 dataset, the inertial frequency falls in the
# physically expected Gulf-of-Mexico range, a genuinely long-lived
# cyclonic ridge is detected (not a noise artifact), and the Kunze
# `f_eff` prediction lands ABOVE `f0` in magnitude (`f_eff = f0 + ω` once
# properly signed) rather than below it — the sign relationship that a
# from-scratch derivation of the closed-form Kunze formula gives, and
# that matches the real Flare Fig. 3's own `f+Eddy` line sitting above
# the Coriolis line, not under it.

@assert n == 8761 "expected the complete 2005 hourly record (8761 rows)"
@assert 0.025 < f0_cph < 0.055 "f0=$f0_cph cyc/h outside the physically expected GoM range"
@assert eddy.npoints > 1000 "expected a long-lived (>1000h) dominant cyclonic ridge, got $(eddy.npoints)h"
@assert eddy.xi_bar > 0.5 "expected a strongly cyclonic (xi_bar>0.5) dominant ridge, got $(eddy.xi_bar)"
@assert f_eff_mag_cph > f0_cph "Kunze f_eff should exceed f0 in magnitude (f_eff=f0+omega), got f_eff=$f_eff_mag_cph < f0=$f0_cph"
println("✓ Verification passed: $n rows, f0=$(round(f0_cph,digits=4)) cyc/h, " *
        "dominant ridge $(eddy.npoints)h (xi_bar=$(round(eddy.xi_bar,digits=2))), " *
        "f_eff=$(round(f_eff_mag_cph,digits=4)) > f0")
