# GulfDrifters baseline validation, significance-test variant.
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
# pending access to the actual published GOMED dataset (Zenodo DOI
# 10.5281/zenodo.3978803 -- access-restricted, requested but not yet
# granted as of this run).
#
# COST: wavelet_significance() runs n_surrogates independent wavelet
# transforms per drifter (Monte Carlo). Calibrated at ~9.6s/drifter for
# n_surrogates=50 on a ~1500-sample record -- a full 2684-drifter run would
# take several hours. This script instead runs on a fixed-seed random
# SUBSAMPLE (documented below), sized to finish in a few minutes.
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

using ValTools, ValTools.JLab
using Multitaper
using Statistics, Printf, Dates, Random

const DT_HOURS = 1.0
const MIN_SAMPLES = 64
const N_SURROGATES = 30
const SUBSAMPLE_N = 150
const CONFIDENCE = 0.95

# Same event-chaining logic as eddy_census (src/JLab/validation.jl), but
# gating ridge points on statistical significance (ridge_amp^2 >
# sig_level at that frequency) instead of an arbitrary amp_thresh fraction
# of the record's own peak amplitude.
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
    ridge_freq, ridge_amp, ridge_qual = ridgemap(wt, scales; quality=true)  # thresh=0.0: all local maxima

    sig_level, fs_ref = wavelet_significance(collect(Float64, u); dt=Float64(dt), fs=scales,
                                              nv=nv, gamma=Float64(gamma), beta=Float64(beta),
                                              background=background, confidence=confidence,
                                              n_surrogates=n_surrogates, rng=rng)
    significant = ridge_significant(ridge_freq, ridge_amp, sig_level, fs_ref)

    events = Dict[]
    in_event = false
    event_start = 0
    for i in 1:N
        if !isnan(ridge_freq[i]) && significant[i]
            if !in_event
                in_event = true
                event_start = i
            end
        else
            if in_event
                duration = i - event_start
                if duration >= min_duration
                    seg = event_start:i-1
                    valid_seg = seg[.!isnan.(ridge_freq[seg]) .& significant[seg]]
                    if !isempty(valid_seg)
                        push!(events, Dict("start" => event_start, "stop" => i - 1,
                                            "duration" => duration,
                                            "mean_frequency" => mean(ridge_freq[valid_seg])))
                    end
                end
                in_event = false
            end
        end
    end
    if in_event
        duration = N - event_start + 1
        if duration >= min_duration
            seg = event_start:N
            valid_seg = seg[.!isnan.(ridge_freq[seg]) .& significant[seg]]
            if !isempty(valid_seg)
                push!(events, Dict("start" => event_start, "stop" => N,
                                    "duration" => duration,
                                    "mean_frequency" => mean(ridge_freq[valid_seg])))
            end
        end
    end
    return events
end

println("Loading GulfDriftersOpen.nc (cached, no download)...")
data = load_gulfdrifters(; download=false)
n = data.n_drifters
println("n_drifters = $n")

Random.seed!(42)
valid_idx = [i for i in 1:n if length(data.u[i]) >= MIN_SAMPLES &&
                                all(isfinite, data.u[i]) && all(isfinite, data.v[i])]
subsample = valid_idx[randperm(length(valid_idx))[1:min(SUBSAMPLE_N, length(valid_idx))]]

total_events = 0
n_used = 0
t0 = time()
for i in subsample
    global n_used += 1
    events = eddy_census_significant(data.u[i], data.v[i], DT_HOURS)
    global total_events += length(events)
end
elapsed = time() - t0

@printf("\n=== GulfDrifters significance-test baseline (%s) ===\n", Dates.format(Dates.now(), "yyyy-mm-dd"))
@printf("Subsample:             %d drifters (seed=42, of %d valid / %d total)\n", n_used, length(valid_idx), n)
@printf("Significant eddy events: %d (%.2f events/drifter)\n", total_events, total_events / n_used)
@printf("Params:                 n_surrogates=%d, confidence=%.2f, background=red\n", N_SURROGATES, CONFIDENCE)
@printf("Elapsed:                %.1f s\n", elapsed)
@printf("\nNOTE: not comparable to the paper's 1033 (different dataset size, different\n")
@printf("      noise model, subsample not full census). See file header for caveats.\n")
