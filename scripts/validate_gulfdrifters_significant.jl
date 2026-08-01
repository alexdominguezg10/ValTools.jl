# GulfDrifters baseline validation, significance-test variant — FULL CENSUS.
#
# scripts/validate_gulfdrifters.jl uses eddy_census() directly, which by its
# own docstring is "a fixed-threshold heuristic, not a statistical
# significance test" -- it does NOT implement the density-ratio significance
# criterion of Lilly & Perez-Brunius (2021, NPG). That paper's own published
# count (1033 statistically significant ridges out of 2520 non-inertial, on
# their full 3770-drifter GulfDriftersAll dataset) is nowhere close to
# reproducible with the plain heuristic, and isn't meant to be -- confirmed
# directly against the paper (Sect. 4.7) after the first validation run's
# numbers (2804 events) turned out to be ~10x the remembered ad hoc baseline.
#
# This script instead combines eddy_census's own ridge-chaining logic with
# wavelet_significance()/ridge_significant() -- the Monte Carlo noise-surrogate
# test the eddy_census docstring itself recommends as the real alternative --
# giving a genuine (if approximate) statistical test on our own public
# GulfDriftersOpen subset (2731 drifters), rather than the arbitrary
# amp_thresh heuristic. It is still not the paper's exact method (their
# density-ratio criterion normalizes jointly across frequency using a
# bivariate/rotary noise model; wavelet_significance here tests only the
# univariate u-component signal against isotropic white/red-noise
# surrogates) and it runs on a smaller, non-proprietary dataset -- so it
# will not reproduce 1033 either. It is a closer, defensible approximation
# now that the real published GOMED dataset (gomed_1.1.0.nc) is available
# for direct comparison via scripts/compare_gomed.jl.
#
# COST: wavelet_significance() runs n_surrogates independent wavelet
# transforms per drifter (Monte Carlo). Calibrated at ~9.6s/drifter for
# n_surrogates=50 on a ~1500-sample record -- serial over the full ~2684
# valid drifters would take several hours. This version runs the FULL
# census (no subsample) but threads the per-drifter loop with
# Threads.@threads -- intended to run on an Ixachi CPU node with many
# cores (see run_gulfdrifters_significant.slurm), not on a laptop.
#
# background=:red is the default (not :white): a same-subsample comparison
# run found white noise gives 5.25 events/drifter vs. red noise's 3.11 --
# consistent with the paper's own Sect. 4.3 observation that "the
# generation of ridges due to noise is more efficient for white noise and
# less efficient for more strongly sloped [red] processes." Real drifter
# background turbulence is reddish (more low-frequency power, exactly
# where eddies live), so a white-noise surrogate sets too permissive a
# threshold at low frequencies -- red is the more defensible null model
# here, even though it's still not the paper's actual bivariate/rotary
# density-ratio criterion.
#
# UNITS NOTE (easy to get wrong, worth stating explicitly): wavetrans's
# returned `fs` (and therefore ridgemap's `ridge_freq`) is in *radians* per
# sample (see wavetrans docstring), i.e. rad/hour given DT_HOURS=1.0 --
# NOT cycles/hour despite eddy_census's own docstring claiming that unit.
# inertial_frequency() (src/JLab/validation.jl) returns cycles/hour. Both
# quantities below are therefore put in rad/hour before dividing, to avoid
# a silent factor-of-2pi error in omega_ast_est.
#
# SENSE NOTE (fixed 2026-07-31, after GOMED comparison showed a real bias):
# an earlier version of this function used the same heuristic as
# eddy_census (src/JLab/validation.jl) -- phase progression of the
# u-component's own wavelet coefficient along the ridge. On the real
# GOMED comparison, that heuristic came out 93.8% cyclonic on the 461
# overlapping drifters vs. GOMED's real 79.0% -- a substantial, consistent
# skew, not noise. Now fixed to the paper's actual method (Lilly &
# Perez-Brunius 2021, Eq. 62): transform u and v SEPARATELY on the same
# frequency grid, form the rotary components w+ = (wx+i*wy)/sqrt(2)
# (CCW/cyclonic in the NH) and w- = (wx-i*wy)/sqrt(2) (CW/anticyclonic),
# and call sense by whichever branch carries more power over the ridge.
# NOTE for future edits: the second wavetrans call below passes the first
# call's `fs` output back in via the `fs=` kwarg, NOT `scales=`. eddy_census
# passes it via `scales=`, which -- per wavetrans's own dispatch --
# re-derives fs as `2*pi*scales*dt`, a DIFFERENT (rescaled) frequency grid
# from the original whenever dt=1. That mismatch is currently inert in
# eddy_census (its wt_v is computed but never actually used for anything),
# but would silently misalign wx/wy if anyone later relies on it -- use
# fs= here specifically to get the true matching grid.

using ValTools, ValTools.JLab
using Multitaper
using Statistics, Printf, Dates, Random

const DT_HOURS = 1.0
const MIN_SAMPLES = 64
const N_SURROGATES = 30   # unchanged from the validated 150-drifter run
const CONFIDENCE = 0.95
const OUTPUT_PATH = joinpath(@__DIR__, "..", "results", "gulfdrifters_significant_full.csv")

# Same event-chaining logic as eddy_census (src/JLab/validation.jl), but
# gating ridge points on statistical significance (ridge_amp^2 >
# sig_level at that frequency) instead of an arbitrary amp_thresh fraction
# of the record's own peak amplitude. Also records "sense" and
# "mean_amplitude" (matching eddy_census's event dict, needed to derive
# omega_ast_est below -- the plain significance-only version this script
# started from didn't track either).
function eddy_census_significant(u::AbstractVector, v::AbstractVector, dt::Real;
                                  nv::Int=8, gamma::Real=3.0, beta::Real=8.0,
                                  min_duration::Real=3.0,
                                  confidence::Real=CONFIDENCE,
                                  n_surrogates::Int=N_SURROGATES,
                                  background::Symbol=:red,
                                  rng::Random.AbstractRNG=Random.default_rng())
    N = length(u)
    z = collect(Float64, u) .+ im .* collect(Float64, v)

    wt, scales = wavetrans(real.(z); dt=Float64(dt), nv=nv, gamma=Float64(gamma), beta=Float64(beta))
    wt_v, _ = wavetrans(imag.(z); dt=Float64(dt), fs=scales, gamma=Float64(gamma), beta=Float64(beta))
    # Rotary decomposition (Lilly & Perez-Brunius 2021, Eq. 62):
    # w+ = CCW-rotating (cyclonic, NH) component, w- = CW-rotating (anticyclonic).
    w_plus  = (wt .+ im .* wt_v) ./ sqrt(2)
    w_minus = (wt .- im .* wt_v) ./ sqrt(2)
    ridge_freq, ridge_amp, ridge_qual = ridgemap(wt, scales; quality=true)  # thresh=0.0: all local maxima

    sig_level, fs_ref = wavelet_significance(collect(Float64, u); dt=Float64(dt), fs=scales,
                                              nv=nv, gamma=Float64(gamma), beta=Float64(beta),
                                              background=background, confidence=confidence,
                                              n_surrogates=n_surrogates, rng=rng)
    significant = ridge_significant(ridge_freq, ridge_amp, sig_level, fs_ref)

    events = Dict[]
    in_event = false
    event_start = 0

    function _push_event!(seg_start, seg_stop, duration)
        seg = seg_start:seg_stop
        valid_seg = seg[.!isnan.(ridge_freq[seg]) .& significant[seg]]
        isempty(valid_seg) && return
        # ridgemap doesn't expose the scale index it matched internally, only
        # the resulting frequency -- recover it by nearest-match against the
        # (small, already-known) scales array, then read off the rotary
        # power at that exact (time, scale) ridge point.
        scale_idx = [argmin(abs.(scales .- ridge_freq[t])) for t in valid_seg]
        pow_plus  = sum(abs2(w_plus[t, j])  for (t, j) in zip(valid_seg, scale_idx))
        pow_minus = sum(abs2(w_minus[t, j]) for (t, j) in zip(valid_seg, scale_idx))
        sense = pow_plus > pow_minus ? :ccw : :cw
        push!(events, Dict(
            "start" => seg_start, "stop" => seg_stop, "duration" => duration,
            "mean_frequency" => mean(ridge_freq[valid_seg]),
            "mean_amplitude" => mean(ridge_amp[valid_seg]),
            "sense" => sense,
        ))
    end

    for i in 1:N
        if !isnan(ridge_freq[i]) && significant[i]
            if !in_event
                in_event = true
                event_start = i
            end
        else
            if in_event
                duration = i - event_start
                duration >= min_duration && _push_event!(event_start, i - 1, duration)
                in_event = false
            end
        end
    end
    if in_event
        duration = N - event_start + 1
        duration >= min_duration && _push_event!(event_start, N, duration)
    end
    return events
end

println("Loading GulfDriftersOpen.nc (cached, no download)...")
data = load_gulfdrifters(; download=false)
n = data.n_drifters
println("n_drifters = $n")

valid_idx = [i for i in 1:n if length(data.u[i]) >= MIN_SAMPLES &&
                                all(isfinite, data.u[i]) && all(isfinite, data.v[i])]
println("valid_idx = $(length(valid_idx)) / $n (full census, no subsample)")
println("Threads.nthreads() = $(Threads.nthreads())")

# One results bucket per drifter (indexed by absolute drifter index), so each
# thread only ever writes to its own slot -- avoids shared-state races under
# Threads.@threads. Unused slots (invalid drifters) stay empty.
results = [NamedTuple[] for _ in 1:n]
progress = Threads.Atomic{Int}(0)
t0 = time()

Threads.@threads for i in valid_idx
    mean_lat = mean(data.lat[i])
    f0_rad_per_hour = 2π * inertial_frequency(mean_lat)   # rad/hour, see UNITS NOTE above
    rng = MersenneTwister(1_000_000 + i)                  # distinct, reproducible per drifter

    events = eddy_census_significant(data.u[i], data.v[i], DT_HOURS; rng=rng)

    rows = NamedTuple[]
    for e in events
        L_est = e["mean_frequency"] * (e["duration"] * DT_HOURS) / (2π)
        sense_sign = e["sense"] == :ccw ? 1.0 : -1.0
        omega_ast_est = sense_sign * e["mean_frequency"] / f0_rad_per_hour
        push!(rows, (drifter_id=data.drifter_id[i], segment_id=data.id[i],
                      L_est=L_est, omega_ast_est=omega_ast_est,
                      sense=String(e["sense"]), duration_hours=e["duration"] * DT_HOURS,
                      mean_frequency_rad_per_hour=e["mean_frequency"],
                      mean_amplitude=e["mean_amplitude"]))
    end
    results[i] = rows

    p = Threads.atomic_add!(progress, 1)
    (p % 200 == 0) && println("  ...$(p) / $(length(valid_idx)) drifters done ($(round(time() - t0, digits=1))s)")
end
elapsed = time() - t0

all_rows = reduce(vcat, results)
total_events = length(all_rows)

mkpath(dirname(OUTPUT_PATH))
open(OUTPUT_PATH, "w") do io
    println(io, "drifter_id,segment_id,L_est,omega_ast_est,sense,duration_hours,mean_frequency_rad_per_hour,mean_amplitude")
    for r in all_rows
        @printf(io, "%d,%d,%.6f,%.6f,%s,%.4f,%.6f,%.6f\n",
                r.drifter_id, r.segment_id, r.L_est, r.omega_ast_est, r.sense,
                r.duration_hours, r.mean_frequency_rad_per_hour, r.mean_amplitude)
    end
end

n_used = length(valid_idx)
n_cyclonic = count(r -> r.omega_ast_est > 0, all_rows)
n_anticyclonic = count(r -> r.omega_ast_est < 0, all_rows)

@printf("\n=== GulfDrifters significance-test FULL CENSUS (%s) ===\n", Dates.format(Dates.now(), "yyyy-mm-dd"))
@printf("Drifters used:            %d / %d valid / %d total\n", n_used, length(valid_idx), n)
@printf("Significant eddy events:  %d (%.2f events/drifter)\n", total_events, total_events / n_used)
@printf("Cyclonic (omega_ast>0):   %d (%.1f%%)\n", n_cyclonic, 100 * n_cyclonic / total_events)
@printf("Anticyclonic (omega_ast<0): %d (%.1f%%)\n", n_anticyclonic, 100 * n_anticyclonic / total_events)
@printf("Params:                   n_surrogates=%d, confidence=%.2f, background=red, threads=%d\n",
        N_SURROGATES, CONFIDENCE, Threads.nthreads())
@printf("Elapsed:                  %.1f s\n", elapsed)
@printf("Output written to:        %s\n", OUTPUT_PATH)
@printf("\nNOTE: xi_bar/R_bar/V_bar (GOMED's ellipse circularity/radius/KE-velocity)\n")
@printf("      are NOT computed here -- eddy_census's ridge-chaining tracks only one\n")
@printf("      rotary branch's amplitude/frequency, not both CW/CCW branches needed\n")
@printf("      to reconstruct ellipse semi-axes. See scripts/compare_gomed.jl header.\n")
