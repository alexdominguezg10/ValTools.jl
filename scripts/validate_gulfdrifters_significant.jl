# GulfDrifters baseline validation, significance-test variant — FULL CENSUS.
#
# scripts/validate_gulfdrifters.jl uses eddy_census() directly, which by its
# own docstring is "a fixed-threshold heuristic, not a statistical
# significance test" -- it does NOT implement the density-ratio significance
# criterion of Lilly & Perez-Brunius (2021, NPG).
#
# METHOD (rewritten 2026-08-01, "ValTools 7"): this script now runs the
# paper's OWN significance machinery -- rotary_ridge_properties (isotropic
# rotary wavetrans, jLab's real eddy-band fmax_ratio=2/fmin_ratio=1/64) +
# rotary_noise_surrogate (Sect. 4.3 isotropic noise model) +
# density_ratio_significance (Sect. 4.6, rho_X = L*|xi_bar|^4 criterion,
# rho<0.1 & omega_ast_bar>-0.5 per compare_gomed.jl's reproduction of the
# paper's published 1033 count) -- identical to case_study_gomed.jl's
# default METHOD=:rhox path, which validated at 26/28 (92.9%) case matches
# against real GOMED ridges. This REPLACES the previous ad hoc
# eddy_census_significant/wavelet_significance AR(1)-red-noise univariate
# test, which was calibrated against velocity input and, when the main loop
# below was switched from velocity to position (see next paragraph), turned
# out to have silently collapsed: 0/28 case-study detections (0%) at
# METHOD=:ar1 on position input, vs. the same cases' 26/28 under :rhox --
# confirmed locally 2026-08-01 before ever trusting a full-census run
# again. Position series are much redder than velocity (integration boosts
# low-frequency power), so an AR(1) red-noise null fit to position sets a
# far harsher bar than the same null fit to velocity -- it wasn't a sync or
# deployment bug (checksums between local and Ixachi matched exactly), it
# was the wrong significance test for the new input signal. See
# project_gomed_validation_results memory for the full trail.
#
# POSITION, NOT VELOCITY (fixed 2026-08-01, "ValTools 7"): jLab's actual
# eddyridges.m/spheretrans.m wavelet-transforms POSITION, not velocity --
# the main loop below passes `local_tangent_plane(data.lat[i], data.lon[i])`
# (a simple local tangent-plane displacement projection,
# src/JLab/validation.jl) instead of `data.u[i], data.v[i]`. See
# project_gomed_validation_results memory for the empirical confirmation
# (mean Jaccard vs. real jLab output 0.575->0.957 on the case-study harness
# after this exact swap).
#
# TWO-PASS DESIGN: density_ratio_significance is a GLOBAL statistic (a
# survival-function ratio over the pooled data/noise ridge population, not
# something evaluable one drifter at a time) -- see its docstring
# (src/JLab/wavelets.jl). Pass 1 (threaded over drifters) extracts each
# drifter's own ridges plus N_NOISE_PER_DRIFTER noise-surrogate ridges into
# per-drifter buckets (race-free, same pattern as before: one bucket per
# absolute drifter index). Pass 2 (single-threaded, cheap -- histogram
# accumulation over ~1e5 ridge points against a fixed 1000x2000 bin grid,
# not a per-ridge O(n^2) operation) pools everything and calls
# density_ratio_significance ONCE. Pass 3 slices the resulting rho values
# back out per drifter and keeps only ridges passing rho<RHO_SIG &
# omega_ast_bar>OMEGA_AST_MIN.
#
# UNITS NOTE: rotary_ridge_properties expects f_coriolis in radians per
# sample (see its docstring); inertial_frequency() (src/JLab/validation.jl)
# returns cycles/hour, converted via 2*pi below (DT_HOURS=1.0, so
# rad/sample == rad/hour here).

using ValTools, ValTools.JLab
using ValTools.Metrics: rotary_noise_surrogate
using Multitaper   # loads the rotary_noise_surrogate extension (weakdep)
using Statistics, Printf, Dates, Random

const DT_HOURS = 1.0
const MIN_SAMPLES = 64
const N_NOISE_PER_DRIFTER = 5   # unchanged from the validated case-study run
const RHO_SIG = 0.1             # paper's accepted-event cutoff (rho_X = L*xi_bar^4)
const OMEGA_AST_MIN = -0.5      # exclude inertial oscillations
const ALPHA = 4                 # X = L*|xi_bar|^alpha exponent (paper's choice)
const OUTPUT_PATH = joinpath(@__DIR__, "..", "results", "gulfdrifters_significant_full.csv")

println("Loading GulfDriftersOpen.nc (cached, no download)...")
data = load_gulfdrifters(; download=false)
n = data.n_drifters
println("n_drifters = $n")

valid_idx = [i for i in 1:n if length(data.u[i]) >= MIN_SAMPLES &&
                                all(isfinite, data.u[i]) && all(isfinite, data.v[i]) &&
                                all(isfinite, data.lat[i]) && all(isfinite, data.lon[i])]
println("valid_idx = $(length(valid_idx)) / $n (full census, no subsample)")
println("Threads.nthreads() = $(Threads.nthreads())")

# One bucket per drifter (indexed by absolute drifter index), so each thread
# only ever writes to its own slot -- avoids shared-state races under
# Threads.@threads. Unused slots (invalid drifters) stay empty.
per_drifter_data = Vector{Vector{NamedTuple}}(undef, n)
per_drifter_noise = Vector{Vector{NamedTuple}}(undef, n)
per_drifter_npts = zeros(Int, n)
for i in 1:n
    per_drifter_data[i] = NamedTuple[]
    per_drifter_noise[i] = NamedTuple[]
end

progress = Threads.Atomic{Int}(0)
t0 = time()

println("\nPass 1: extracting ridges + $(N_NOISE_PER_DRIFTER) noise surrogates per drifter...")
Threads.@threads for i in valid_idx
    mean_lat = mean(data.lat[i])
    f0 = 2π * inertial_frequency(mean_lat)   # rad/hour, see UNITS NOTE above

    x_km, y_km = local_tangent_plane(data.lat[i], data.lon[i])
    rr = rotary_ridge_properties(x_km, y_km; f_coriolis=f0, fmax_ratio=2, fmin_ratio=1/64)
    per_drifter_data[i] = rr
    per_drifter_npts[i] = length(x_km)

    noise_rr = NamedTuple[]
    for s in 1:N_NOISE_PER_DRIFTER
        ex, ey = rotary_noise_surrogate(x_km, y_km; dt_hours=DT_HOURS, nw=4.0,
                                        rng=MersenneTwister(9_000_000 + 100i + s))
        append!(noise_rr, rotary_ridge_properties(ex, ey; f_coriolis=f0, fmax_ratio=2, fmin_ratio=1/64))
    end
    per_drifter_noise[i] = noise_rr

    p = Threads.atomic_add!(progress, 1)
    (p % 200 == 0) && println("  ...$(p) / $(length(valid_idx)) drifters done ($(round(time() - t0, digits=1))s)")
end
pass1_elapsed = time() - t0

data_pool = reduce(vcat, per_drifter_data[valid_idx])
noise_pool = reduce(vcat, per_drifter_noise[valid_idx])
n_data_pts_total = sum(per_drifter_npts[valid_idx])
n_noise_pts_total = N_NOISE_PER_DRIFTER * n_data_pts_total
println("Pass 1 done: $(length(data_pool)) data ridges, $(length(noise_pool)) noise ridges ($(round(pass1_elapsed, digits=1))s)")

println("\nPass 2: density-ratio significance over the pooled population...")
t1 = time()
rho_all = density_ratio_significance(data_pool, noise_pool, n_data_pts_total, n_noise_pts_total; alpha=ALPHA)
pass2_elapsed = time() - t1
println("Pass 2 done ($(round(pass2_elapsed, digits=1))s)")

println("\nPass 3: slicing significance back out per drifter...")
# Wrapped in a function (not a bare top-level loop) so `off` is an ordinary
# local -- at top level `for` opens a soft scope and a running counter
# silently breaks (same pitfall documented in case_study_gomed.jl's own
# _rhox_pass).
function _slice_significant(valid_idx, per_drifter_data, rho_all, data, DT_HOURS)
    rows = NamedTuple[]
    off = 0
    for i in valid_idx
        rr = per_drifter_data[i]
        rhos = rho_all[off+1:off+length(rr)]
        off += length(rr)
        for (r, rho) in zip(rr, rhos)
            (rho < RHO_SIG && r.omega_ast_bar > OMEGA_AST_MIN) || continue
            push!(rows, (drifter_id=data.drifter_id[i], segment_id=data.id[i],
                          L=r.L, omega_ast_bar=r.omega_ast_bar, sense=String(r.sense),
                          npoints=r.npoints, duration_hours=(r.stop - r.start + 1) * DT_HOURS,
                          xi_bar=r.xi_bar, kappa_bar=r.kappa_bar, rho_x=rho))
        end
    end
    return rows
end
all_rows = _slice_significant(valid_idx, per_drifter_data, rho_all, data, DT_HOURS)
elapsed = pass1_elapsed + pass2_elapsed + (time() - t1 - pass2_elapsed)
total_events = length(all_rows)

mkpath(dirname(OUTPUT_PATH))
open(OUTPUT_PATH, "w") do io
    println(io, "drifter_id,segment_id,L,omega_ast_bar,sense,npoints,duration_hours,xi_bar,kappa_bar,rho_x")
    for r in all_rows
        @printf(io, "%d,%d,%.6f,%.6f,%s,%d,%.4f,%.6f,%.6f,%.6f\n",
                r.drifter_id, r.segment_id, r.L, r.omega_ast_bar, r.sense,
                r.npoints, r.duration_hours, r.xi_bar, r.kappa_bar, r.rho_x)
    end
end

n_used = length(valid_idx)
n_cyclonic = count(r -> r.omega_ast_bar > 0, all_rows)
n_anticyclonic = count(r -> r.omega_ast_bar < 0, all_rows)

@printf("\n=== GulfDrifters density-ratio significance FULL CENSUS (%s) ===\n", Dates.format(Dates.now(), "yyyy-mm-dd"))
@printf("Drifters used:            %d / %d valid / %d total\n", n_used, length(valid_idx), n)
@printf("Data ridges (pre-filter):  %d\n", length(data_pool))
@printf("Significant eddy events:  %d (%.3f events/drifter)\n", total_events, total_events / n_used)
@printf("Cyclonic (omega_ast>0):   %d (%.1f%%)\n", n_cyclonic, 100 * n_cyclonic / max(total_events, 1))
@printf("Anticyclonic (omega_ast<0): %d (%.1f%%)\n", n_anticyclonic, 100 * n_anticyclonic / max(total_events, 1))
@printf("Params:                   N_NOISE_PER_DRIFTER=%d, rho_sig=%.2f, omega_ast_min=%.1f, alpha=%d, threads=%d\n",
        N_NOISE_PER_DRIFTER, RHO_SIG, OMEGA_AST_MIN, ALPHA, Threads.nthreads())
@printf("Elapsed:                  pass1=%.1fs pass2=%.1fs total=%.1fs\n", pass1_elapsed, pass2_elapsed, pass1_elapsed + pass2_elapsed)
@printf("Output written to:        %s\n", OUTPUT_PATH)
@printf("\nNOTE: R_bar/V_bar (GOMED's ellipse radius/KE-velocity) are still NOT\n")
@printf("      computed here -- rotary_ridge_properties tracks one-sided ridge\n")
@printf("      amplitude/frequency (xi_bar/kappa_bar), not full ellipse semi-axes.\n")
@printf("      See scripts/compare_gomed.jl header.\n")
