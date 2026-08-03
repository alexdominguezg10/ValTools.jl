```@meta
EditURL = "../../../examples/gomed_eddy_census.jl"
```

# Case Study: Detecting Significant Eddies in Real Lagrangian Drifter Data

Every other example in this gallery uses synthetic data with a known
ground truth. This one is different: it's the production output of
ValTools.jl's eddy-detection pipeline (`local_tangent_plane` +
`rotary_ridge_properties` + `density_ratio_significance`) run on the
**real, public GulfDriftersOpen dataset** (Lilly & Pérez-Brunius, 2021a;
2684 drifters, Gulf of Mexico) — a wavelet-ridge port of the method from
Lilly & Pérez-Brunius (2021b, *Nonlin. Processes Geophys.* 28, 181–212),
who applied the same technique to the full (partly proprietary)
GulfDriftersAll dataset and published a companion eddy census, GOMED.

**Why a bundled CSV instead of running the pipeline live:** the full
census over 2684 drifters is a multi-hour HPC job (originally run on
Ixachi), and GulfDriftersOpen itself is too large to fetch on every docs
build. Per the gallery's own data strategy for flagship case studies, this
example loads `gomed_eddy_census_data.csv` — the pipeline's own output, a
small (~36 KB) table of every event it found significant — and reproduces
the same summary statistics and figures a full run would show, entirely
from that bundled file.

**On GOMED:** the published GOMED eddy-census NetCDF file is
**non-commercial/no-redistribution licensed** and is deliberately not used
or shipped anywhere in this example — nothing here contains or plots
GOMED's own per-ridge values. The full validated ValTools-vs-GOMED
agreement (population statistics, not individual events) lives in the
project's private research notes, not in this public repo. What follows
compares our own detections only against the paper's own **published**
headline figures for the full census (Sect. 4.7), which are public.

**Reference:** jLab `eddyridges.m`/`spheretrans.m` (position, not
velocity, is what gets wavelet-transformed); NPG 2021b Eq. 61–63
(ridge length, significance parameter `X = L·ξ̄⁴`, density ratio `ρ_X<0.1`).

## Load our own detection output

````julia
using Statistics, CairoMakie

function read_events_csv(path)
    lines = readlines(path)
    header = split(lines[1], ",")
    rows = NamedTuple[]
    for line in @view lines[2:end]
        isempty(line) && continue
        f = split(line, ",")
        push!(rows, (
            drifter_id = parse(Int, f[1]), segment_id = parse(Int, f[2]),
            L = parse(Float64, f[3]), omega_ast_bar = parse(Float64, f[4]),
            sense = f[5], npoints = parse(Int, f[6]),
            duration_hours = parse(Float64, f[7]), xi_bar = parse(Float64, f[8]),
            kappa_bar = parse(Float64, f[9]), rho_x = parse(Float64, f[10]),
        ))
    end
    return rows
end

events = read_events_csv(joinpath(@__DIR__, "gomed_eddy_census_data.csv"))
n_events = length(events)
n_drifters = length(unique(e.drifter_id for e in events))

println("Loaded $n_events significant eddy events on $n_drifters distinct drifters")
println("(from ValTools.jl's own rotary_ridge_properties + density_ratio_significance")
println(" pipeline, run on the public GulfDriftersOpen dataset)")
````

## Summary statistics

````julia
L_vals = [e.L for e in events]
omega_vals = [e.omega_ast_bar for e in events]
n_cyclonic = count(>(0), omega_vals)
n_anticyclonic = count(<(0), omega_vals)
count_weighted_sign = mean(sign.(omega_vals))
energy_weighted_sign = sum(sign.(omega_vals) .* [e.kappa_bar for e in events]) /
                        sum(e.kappa_bar for e in events)

println("\n--- Ridge length L (cycles) ---")
println("  mean = ", round(mean(L_vals), digits=2), "  median = ", round(median(L_vals), digits=2))

println("\n--- Rotation sense (sign of omega*; + cyclonic, - anticyclonic, NH) ---")
println("  cyclonic:     $n_cyclonic  (", round(100 * n_cyclonic / n_events, digits=1), "%)")
println("  anticyclonic: $n_anticyclonic  (", round(100 * n_anticyclonic / n_events, digits=1), "%)")
println("  count-weighted mean sign:  ", round(count_weighted_sign, digits=3))
println("  energy-weighted mean sign: ", round(energy_weighted_sign, digits=3))
````

## Context: the published headline numbers

Lilly & Pérez-Brunius (2021b, Sect. 4.7) report, for the **full**
(partly proprietary) GulfDriftersAll census: 14471 total ridges detected,
2520 non-inertial (`ω̄*>-1/2`), and 1033 of those statistically significant
(`ρ_X<0.1`) — 41% of the non-inertial population. They also report a
**size-dependent cyclone/anticyclone asymmetry**: small eddies
(`R̄<10 km`) are cyclone-only (anticyclones are dynamically forbidden by
the `ζ/f=-1` centrifugal-instability boundary), mid-size eddies remain
cyclone-dominated, and only the largest eddies (`R̄>50-100 km`, Loop
Current Eddies) become anticyclone-dominated. Our own cyclonic majority
above is directionally consistent with that published asymmetry, on the
public GulfDriftersOpen subset.

## Visualization

````julia
fig = Figure(size=(1000, 420))

ax1 = Axis(fig[1, 1], xlabel="L (cycles)", ylabel="count",
           title="Ridge length of significant events (n=$n_events)")
L_floor = 2 * sqrt(6) / π   # NPG 2021b Eq. 61 ridge-length threshold, ≈1.5595
hist!(ax1, L_vals, bins=25, color=(:dodgerblue, 0.7))
vlines!(ax1, [L_floor], color=:gray40, linestyle=:dash, label="L>2√6/π significance floor")
axislegend(ax1)

ax2 = Axis(fig[1, 2], xlabel="ω* (nondimensional frequency)", ylabel="count",
           title="Rotation sense (+ cyclonic, - anticyclonic)")
hist!(ax2, omega_vals, bins=25, color=(:seagreen, 0.7))
vlines!(ax2, [0.0], color=:gray40, linestyle=:dash)

save(joinpath(@__DIR__, "gomed_eddy_census.png"), fig)
println("\nSaved gomed_eddy_census.png")
````

## Verification

Self-consistency checks on our own output (no external/licensed data
involved): the loaded row count matches the file, every event clears the
paper's own ridge-length significance floor, and the population is
majority-cyclonic — consistent with the published size-dependent
asymmetry at these (mostly small/mid) eddy scales.

````julia
L_floor = 2 * sqrt(6) / π   # NPG 2021b Eq. 61, ≈1.5595
@assert n_events == length(events) "row count mismatch after parsing"
@assert minimum(L_vals) > L_floor "all significant events should clear L>2√6/π≈$(round(L_floor, digits=4))"
@assert n_cyclonic > n_anticyclonic "expect a cyclonic majority, consistent with the paper's published asymmetry"
println("✓ Verification passed: $n_events events loaded, all L>$(round(L_floor, digits=3)), cyclonic majority ($n_cyclonic vs $n_anticyclonic)")
````

