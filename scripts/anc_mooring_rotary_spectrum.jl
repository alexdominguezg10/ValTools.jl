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
