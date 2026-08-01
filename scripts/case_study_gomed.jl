# Case-study comparison: N specific GOMED ridges vs. our own pipeline run
# on the exact same drifter segment.
#
# scripts/compare_gomed.jl gives aggregate/distributional agreement
# (counts, L, omega_ast distributions, sign balance). This script goes the
# other direction: pick individual, real GOMED ridges spanning the paper's
# own headline regimes (small/mid/large radius x cyclonic/anticyclonic),
# re-run our significance-test pipeline on that EXACT drifter segment, and
# check whether we detect an overlapping event in the same time window --
# and if so, whether its rotation sense agrees with GOMED's. This is a
# time-localized sanity check that the aggregate numbers in
# compare_gomed.jl aren't accidentally agreeing/disagreeing for unrelated
# reasons (e.g. right sign balance overall but on completely different
# ridges).
#
# eddy_census_significant here is DUPLICATED from
# scripts/validate_gulfdrifters_significant.jl (that script is the
# canonical source, including the Eq.62 rotary-sense fix) rather than
# shared via an include -- keeps this diagnostic script standalone. If the
# detection/sense logic is changed again there, mirror it here too.
#
# POSITION, NOT VELOCITY (fixed 2026-08-01, "ValTools 7"): both the :rhox
# (rotary_ridge_properties) and :ar1 (eddy_census_significant) pipelines
# below now analyze `local_tangent_plane(lat, lon)` -- a local
# tangent-plane displacement projection (src/JLab/validation.jl) -- not
# raw GulfDriftersOpen velocity `data.u`/`data.v`. This matches jLab's real
# eddyridges.m/spheretrans.m, which wavelet-transforms position, never
# velocity; see local_tangent_plane's docstring and
# project_gomed_validation_results memory for the full story (mean Jaccard
# vs. real jLab output on this exact 28-case harness: 0.575 -> 0.957).
# `rotary_ridge_properties` also now gets `fmax_ratio=2, fmin_ratio=1/64`
# (jLab's real Gulf-census frequency band, "ValTools 6") -- previously
# unset here, using the generic full-spectrum grid instead.

using ValTools, ValTools.JLab
using ValTools.Metrics: rotary_noise_surrogate
using Multitaper
using NCDatasets, DataFrames, Statistics, Printf, Dates, Random, CairoMakie

# Which significance method to test. :ar1 = the original per-drifter
# wavelet_significance AR(1)/red-noise null. :rhox = the paper's own
# machinery -- isotropic rotary noise surrogates (Sect. 4.3) + one-sided
# ridges + the global density-ratio criterion (Sect. 4.6). Case SELECTION
# is identical either way (same seed, same stratified bins), so the two
# runs are directly comparable case-by-case.
const METHOD = Symbol(get(ENV, "GOMED_METHOD", "rhox"))
const N_CASES = 28
const N_NOISE_PER_DRIFTER = 5   # :rhox only
const RHO_SIG = 0.1             # :rhox only, paper's accepted-event cutoff
const GOMED_PATH = joinpath(homedir(), ".valtools", "jdata", "gomed_1.1.0.nc")
const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const DT_HOURS = 1.0
const N_SURROGATES = 30
const CONFIDENCE = 0.95
const RHO_COL = 5           # X = L*xi_bar^4 (paper's chosen criterion)
const RHO_THRESHOLD = 0.1
const OMEGA_AST_MIN = -0.5  # exclude inertial oscillations

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
    w_plus  = (wt .+ im .* wt_v) ./ sqrt(2)
    w_minus = (wt .- im .* wt_v) ./ sqrt(2)
    ridge_freq, ridge_amp, ridge_qual = ridgemap(wt, scales; quality=true)

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

println("Loading GOMED ($GOMED_PATH)...")
ds = NCDatasets.Dataset(GOMED_PATH, "r")
segment_id = Int.(ds["segment_id"][:])
drifter_id = Int.(ds["drifter_id"][:])
ridge_id_arr = Int.(ds["ridge_id"][:])
Lg = Float64.(ds["L"][:])
omega_bar = Float64.(ds["omega_ast_bar"][:])
R_bar = Float64.(ds["R_bar"][:])
rho5 = Float64.(ds["rho"][:, RHO_COL])
ids_obs = Int.(ds["ids"][:])
time_obs = Float64.(ds["time"][:])
close(ds)
println("  GOMED: $(length(Lg)) total ridges")

println("Loading GulfDriftersOpen...")
data = load_gulfdrifters(; download=false)
our_ids = Set(data.id)
id_to_idx = Dict(data.id[i] => i for i in 1:data.n_drifters)

sig = (rho5 .< RHO_THRESHOLD) .& (omega_bar .> OMEGA_AST_MIN)
avail = [k for k in findall(sig) if segment_id[k] in our_ids]
@printf("  GOMED significant ridges available in our local dataset: %d / %d significant\n",
        length(avail), count(sig))

# Stratify by the paper's own size-regime boundaries (R<10km / 10-50km /
# >50km) x sign(omega_ast_bar), so the case study spans the headline
# cyclonic/anticyclonic size asymmetry rather than being a plain random
# sample dominated by whichever regime happens to be most common.
function regime(k)
    r = R_bar[k]
    size_bin = r < 10 ? :small : r < 50 ? :mid : :large
    sign_bin = omega_bar[k] > 0 ? :cyc : :acyc
    return (size_bin, sign_bin)
end

bins = Dict{Tuple{Symbol,Symbol},Vector{Int}}()
for k in avail
    push!(get!(bins, regime(k), Int[]), k)
end

Random.seed!(2026)
per_bin = cld(N_CASES, length(bins))
selected = Int[]
println("\nStratified sampling ($(length(bins)) regime bins, ~$per_bin/bin target):")
for (key, idxs) in sort(collect(bins); by=first)
    n_take = min(per_bin, length(idxs))
    append!(selected, idxs[randperm(length(idxs))[1:n_take]])
    println("  $key: $(length(idxs)) available, took $n_take")
end
selected = selected[1:min(N_CASES, length(selected))]
println("Selected $(length(selected)) cases")

# --- Run our pipeline on each case's exact segment, compare against GOMED ---
# For :rhox this is a genuine two-pass census over just these cases: ridges
# and noise-surrogate ridges are pooled across all selected drifters first,
# because the density-ratio survival functions are a GLOBAL statistic and
# cannot be evaluated one drifter at a time.
rhox_events = Dict{Int,Vector{Any}}()
# Wrapped in a function (not a bare top-level loop) so the accumulators are
# ordinary locals -- at top level `for` opens a soft scope and running totals
# silently break.
function _rhox_pass(selected, data, id_to_idx, segment_id)
    println("\nPass 1: extracting ridges + $(N_NOISE_PER_DRIFTER) noise surrogates per drifter...")
    data_pool = NamedTuple[]; noise_pool = NamedTuple[]
    n_data_pts = 0; n_noise_pts = 0
    per_case = Dict{Int,Vector{NamedTuple}}()
    for k in selected
        i = id_to_idx[segment_id[k]]
        lat, lon = data.lat[i], data.lon[i]
        # POSITION, not velocity (fixed 2026-08-01, "ValTools 7"): jLab's
        # real eddyridges.m/spheretrans.m wavelet-transforms displacement,
        # not velocity -- see local_tangent_plane's docstring and
        # project_gomed_validation_results memory for the full story and
        # the empirical confirmation (mean Jaccard vs. real jLab 0.575->0.957).
        x_km, y_km = local_tangent_plane(lat, lon)
        f0 = 2π * inertial_frequency(mean(lat))
        rr = rotary_ridge_properties(x_km, y_km; f_coriolis=f0, fmax_ratio=2, fmin_ratio=1/64)
        per_case[k] = rr
        append!(data_pool, rr); n_data_pts += length(x_km)
        for s in 1:N_NOISE_PER_DRIFTER
            ex, ey = rotary_noise_surrogate(x_km, y_km; dt_hours=DT_HOURS, nw=4.0,
                                            rng=MersenneTwister(7_000_000 + 100k + s))
            append!(noise_pool, rotary_ridge_properties(ex, ey; f_coriolis=f0, fmax_ratio=2, fmin_ratio=1/64))
            n_noise_pts += length(ex)
        end
    end
    println("Pass 2: density-ratio significance over $(length(data_pool)) data / $(length(noise_pool)) noise ridges...")
    rho_all = density_ratio_significance(data_pool, noise_pool, n_data_pts, n_noise_pts; alpha=4)

    out = Dict{Int,Vector{Any}}()
    off = 0
    for k in selected
        rr = per_case[k]
        rhos = rho_all[off+1:off+length(rr)]
        off += length(rr)
        out[k] = [(rr[j], rhos[j]) for j in eachindex(rr)
                  if rhos[j] < RHO_SIG && rr[j].omega_ast_bar > -0.5]
    end
    println("  significant: $(sum(length(v) for v in values(out); init=0)) / $(length(data_pool)) ridges")
    return out
end

if METHOD === :rhox
    global rhox_events = _rhox_pass(selected, data, id_to_idx, segment_id)
end

results = NamedTuple[]
for k in selected
    sid = segment_id[k]
    i = id_to_idx[sid]
    t, lat, lon = data.time[i], data.lat[i], data.lon[i]

    gomed_rows = findall(==(ridge_id_arr[k]), ids_obs)
    gt = time_obs[gomed_rows]
    g_t0, g_t1 = extrema(gt)

    # Normalize both methods to (start_idx, stop_idx, L, omega_ast) tuples
    # so the matching logic below is identical for either.
    cands = NamedTuple[]
    if METHOD === :rhox
        for (rr, _rho) in get(rhox_events, k, Any[])
            push!(cands, (start=rr.start, stop=rr.stop, L=rr.L, omega=rr.omega_ast_bar))
        end
        n_events_here = length(get(rhox_events, k, Any[]))
    else
        rng = MersenneTwister(2_000_000 + k)
        f0 = 2π * inertial_frequency(mean(lat))
        x_km, y_km = local_tangent_plane(lat, lon)  # position, not velocity -- see local_tangent_plane docstring
        evs = eddy_census_significant(x_km, y_km, DT_HOURS; rng=rng)
        for e in evs
            L_e = e["mean_frequency"] * (e["duration"] * DT_HOURS) / (2π)
            ss = e["sense"] == :ccw ? 1.0 : -1.0
            push!(cands, (start=e["start"], stop=e["stop"], L=L_e,
                          omega=ss * e["mean_frequency"] / f0))
        end
        n_events_here = length(evs)
    end

    best_overlap = 0.0
    best_event = nothing
    for e in cands
        e_t0, e_t1 = t[e.start], t[e.stop]
        inter = max(0.0, min(e_t1, g_t1) - max(e_t0, g_t0))
        uni = max(e_t1, g_t1) - min(e_t0, g_t0)
        jacc = uni > 0 ? inter / uni : 0.0
        if jacc > best_overlap
            best_overlap = jacc
            best_event = e
        end
    end

    if best_event !== nothing
        L_est = best_event.L
        omega_ast_est = best_event.omega
        our_t0, our_t1 = t[best_event.start], t[best_event.stop]
        sense_agree = sign(omega_ast_est) == sign(omega_bar[k])
    else
        L_est = NaN
        omega_ast_est = NaN
        our_t0 = NaN
        our_t1 = NaN
        sense_agree = missing
    end

    push!(results, (ridge_id=ridge_id_arr[k], segment_id=sid, drifter_id=drifter_id[k],
                     regime=regime(k), gomed_L=Lg[k], gomed_omega=omega_bar[k], gomed_R=R_bar[k],
                     gomed_t0=g_t0, gomed_t1=g_t1,
                     matched=best_event !== nothing, overlap=best_overlap,
                     our_L=L_est, our_omega=omega_ast_est, our_t0=our_t0, our_t1=our_t1,
                     sense_agree=sense_agree, n_our_events=n_events_here))
end

# --- Report ---
n_matched = count(r -> r.matched, results)
n_sense_agree = count(r -> r.matched && r.sense_agree === true, results)
@printf("\n=== Case study: %d GOMED ridges (METHOD=%s) ===\n", length(results), METHOD)
@printf("Our method detected an overlapping event: %d/%d (%.1f%%)\n",
        n_matched, length(results), 100 * n_matched / length(results))
@printf("Of those, rotation sense agreed with GOMED: %d/%d (%.1f%%)\n",
        n_sense_agree, n_matched, 100 * n_sense_agree / max(n_matched, 1))

# A "match" above only requires nonzero temporal Jaccard overlap -- some of
# those are tiny (overlap<0.1), i.e. a short, different (often near-inertial)
# event our method found that happens to fall inside GOMED's much longer
# ridge window, not a real detection of the same physical eddy. Report a
# stricter tier too.
const STRONG_OVERLAP = 0.4
n_strong = count(r -> r.matched && r.overlap >= STRONG_OVERLAP, results)
n_strong_agree = count(r -> r.matched && r.overlap >= STRONG_OVERLAP && r.sense_agree === true, results)
@printf("Strong matches (temporal overlap >= %.1f): %d/%d (%.1f%% of all cases)\n",
        STRONG_OVERLAP, n_strong, length(results), 100 * n_strong / length(results))
@printf("  of which sense agreed: %d/%d (%.1f%%)\n",
        n_strong_agree, n_strong, 100 * n_strong_agree / max(n_strong, 1))
println()
@printf("%-4s %-11s %-9s %8s %8s %8s | %6s %8s %8s %6s %8s\n",
        "idx", "regime", "ridge_id", "gomed_L", "gomed_w", "gomed_R", "match", "our_L", "our_w", "sense", "overlap")
for (idx, r) in enumerate(results)
    @printf("%-4d %-11s %-9d %8.2f %8.3f %8.1f | %6s %8s %8s %6s %8s\n",
            idx, join(r.regime, "_"), r.ridge_id, r.gomed_L, r.gomed_omega, r.gomed_R,
            r.matched ? "yes" : "no",
            r.matched ? @sprintf("%.2f", r.our_L) : "-",
            r.matched ? @sprintf("%+.3f", r.our_omega) : "-",
            r.matched ? (r.sense_agree ? "agree" : "DISAGREE") : "-",
            r.matched ? @sprintf("%.2f", r.overlap) : "-")
end

mkpath(RESULTS_DIR)
open(joinpath(RESULTS_DIR, "gomed_case_study.csv"), "w") do io
    println(io, "idx,regime,ridge_id,segment_id,drifter_id,gomed_L,gomed_omega,gomed_R,matched,overlap,our_L,our_omega,sense_agree,n_our_events")
    for (idx, r) in enumerate(results)
        @printf(io, "%d,%s,%d,%d,%d,%.4f,%.4f,%.2f,%s,%.4f,%s,%s,%s,%d\n",
                idx, join(r.regime, "_"), r.ridge_id, r.segment_id, r.drifter_id,
                r.gomed_L, r.gomed_omega, r.gomed_R, r.matched, r.overlap,
                r.matched ? @sprintf("%.4f", r.our_L) : "NA",
                r.matched ? @sprintf("%.4f", r.our_omega) : "NA",
                r.matched ? string(r.sense_agree) : "NA", r.n_our_events)
    end
end
println("\nSaved table: results/gomed_case_study.csv")

# --- Grid plot: real GPS trajectory loops per case, GOMED window vs. ours ---
ncols = 6
nrows = cld(length(results), ncols)
fig = Figure(size=(ncols * 260, nrows * 260 + 60))
Label(fig[0, 1:ncols],
      "gray = full local window · orange = GOMED ridge · blue dashed = our matched event",
      fontsize=13)

for (idx, r) in enumerate(results)
    i = id_to_idx[r.segment_id]
    lat, lon, t = data.lat[i], data.lon[i], data.time[i]
    pad = max(2.0, 0.3 * (r.gomed_t1 - r.gomed_t0))
    w0, w1 = r.gomed_t0 - pad, r.gomed_t1 + pad
    win = findall(x -> w0 <= x <= w1, t)

    row, col = divrem(idx - 1, ncols) .+ (1, 1)
    ax = Axis(fig[row+1, col], title="#$idx $(r.regime) $(r.matched ? (r.sense_agree ? "✓" : "✗sense") : "miss")",
              titlesize=10, xticklabelsvisible=false, yticklabelsvisible=false, aspect=DataAspect())
    isempty(win) && continue

    lines!(ax, lon[win], lat[win], color=(:gray, 0.6), linewidth=1)
    gwin = win[(t[win] .>= r.gomed_t0) .& (t[win] .<= r.gomed_t1)]
    !isempty(gwin) && lines!(ax, lon[gwin], lat[gwin], color=:orangered, linewidth=2.5)
    if r.matched
        owin = win[(t[win] .>= r.our_t0) .& (t[win] .<= r.our_t1)]
        !isempty(owin) && lines!(ax, lon[owin], lat[owin], color=:dodgerblue, linewidth=1.8, linestyle=:dash)
    end
end

save(joinpath(RESULTS_DIR, "gomed_case_study.png"), fig)
println("Saved plot: results/gomed_case_study.png")
