# GulfDrifters baseline validation (Stage 4c).
#
# No prior version of this script exists in the repo -- the numbers
# referenced in planning notes ("273 events, 127 longest, KE 87.7%, rotary
# +0.176") were computed ad hoc in an earlier, undocumented session, and the
# exact eddy_census/kinetic_energy_budget parameters and "longest" cutoff
# used to get them aren't recorded anywhere. This script establishes a
# fresh, reproducible baseline using default parameters and a documented
# "longest" cutoff (duration >= 180 days) -- treat its numbers as a NEW
# candidate baseline for review, not a reproduction of the old ones.
#
# `load_gulfdrifters()` returns lon/lat/u/v/id per drifter but not `time`
# (a docstring/return mismatch -- flagged separately, not fixed here to
# keep this script's diff self-contained). Sampling is hourly per the
# dataset's own documentation and confirmed directly against the cached
# file's `time` variable below, so dt_hours=1.0 is used throughout rather
# than reading `time` a second time from the raw NetCDF.

using ValTools, ValTools.JLab
using Multitaper  # loads ValToolsMultitaperExt (mspec/rotary_spectrum/kinetic_energy_budget backend)
using Statistics, Printf, Dates

const DT_HOURS = 1.0
const MIN_SAMPLES = 64          # eddy_census/mspec need enough points for a meaningful wavelet/multitaper estimate
const LONGEST_CUTOFF_DAYS = 180 # documented "longest" definition for this run (>= 180 days of hourly data)

println("Loading GulfDriftersOpen.nc (cached, no download)...")
data = load_gulfdrifters(; download=false)
n = data.n_drifters
println("n_drifters = $n")

total_events = 0
longest_events = 0
n_longest = 0
n_used = 0
ke_mesoscale_fracs = Float64[]
rotary_coeffs = Float64[]

t0 = time()
for i in 1:n
    u = data.u[i]; v = data.v[i]
    N = length(u)
    (N < MIN_SAMPLES || any(!isfinite, u) || any(!isfinite, v)) && continue
    global n_used += 1

    events = eddy_census(u, v, DT_HOURS)
    global total_events += length(events)

    is_longest = N * DT_HOURS / 24 >= LONGEST_CUTOFF_DAYS
    if is_longest
        global n_longest += 1
        global longest_events += length(events)
    end

    ke_total, ke_by_band, _ = kinetic_energy_budget(u, v, DT_HOURS)
    if ke_total > 0
        push!(ke_mesoscale_fracs, ke_by_band["mesoscale"] / ke_total)
    end

    rs = rotary_spectrum(u, v; dt_hours=DT_HOURS, ci=false, ftest=false)
    push!(rotary_coeffs, mean(rs.rotary_coefficient .* (rs.S_ccw .+ rs.S_cw)) / mean(rs.S_ccw .+ rs.S_cw))
end
elapsed = time() - t0

@printf("\n=== GulfDrifters baseline (%s) ===\n", Dates.format(Dates.now(), "yyyy-mm-dd"))
@printf("Drifters used:        %d / %d (>= %d samples, finite u/v)\n", n_used, n, MIN_SAMPLES)
@printf("Total eddy events:    %d\n", total_events)
@printf("Longest-subset (>=%d d): %d drifters, %d events\n", LONGEST_CUTOFF_DAYS, n_longest, longest_events)
@printf("Mesoscale KE fraction: %.1f%% (median across drifters)\n", 100 * median(ke_mesoscale_fracs))
@printf("Mean rotary coeff:     %+.3f (energy-weighted, per-drifter mean)\n", mean(rotary_coeffs))
@printf("Elapsed:               %.1f s\n", elapsed)
