# Cross-check ValTools.jl's `ellipse_polarization` (Metrics/Multitaper ext)
# against REAL jLab MATLAB output -- this function has NO existing jLab
# crosscheck (see the ValTools functionality/validation survey, 2026-08-05):
# it's tested only with synthetic rectilinear/circular/noise smoke tests,
# which is exactly the self-consistency-only gap working-agreement rule 12
# warns about, since P/alpha/beta/theta/nu are all scale-INVARIANT ratios
# and can't by themselves catch a normalization bug in d1/d2 (the
# spectral-matrix eigenvalues, in absolute power units).
#
# `ellipse_polarization` is ValTools' own combination of jLab's
# jSpectral/polparams.m + specdiag.m (specdiag.m itself just calls
# polparams.m internally, confirmed by reading the real jLab source) applied
# to a per-taper spectral matrix built the same way jLab's own mspec.m
# builds it. This script drives that EXACT real jLab pipeline --
# sleptap -> mspec -> specdiag/polparams -- on the same data ValTools
# processes, and diffs every output field.
#
# REAL data: ARE mooring, LR75DW instrument (mid-depth, ~715 m), the same
# real ANC_GoMW_deep record already loaded for
# crosscheck_variance_ellipse.jl -- bin 1 (first range cell), a N=512-sample
# contiguous window (real dt=0.5 h, confirmed from the file's own
# timestamps -- NOT assumed). Real current-meter records genuinely contain
# tidal/inertial ellipse structure, so this is a more meaningful check than
# a synthetic construction, at the cost of not having a hand-known "true"
# theta/lambda to sanity-check against -- the pass/fail criterion is
# Julia-vs-real-jLab agreement, not agreement with a designed input.
#
# N=512 (not more): jLab's sleptap.m computes EXACT tridiagonal Slepian
# tapers for M<=512 but SPLINE-INTERPOLATES from the M=512 case for longer
# series (its own documented approximation) -- using N>512 here would
# compare Julia's exact tapers against jLab's approximated ones and blame
# `ellipse_polarization` for a difference that's actually just sleptap's
# own interpolation, not a port bug.
#
# Usage (from ValTools.jl root, needs MATLAB + MAT.jl + Multitaper.jl):
#   julia --project=envs/cpu scripts/jlab_crosscheck_ellipse_polarization.jl
#   julia --project=envs/cpu scripts/jlab_crosscheck_ellipse_polarization.jl --skip-matlab

using ValTools, MAT, ValTools.JLab, ValTools.Metrics, Multitaper, DelimitedFiles, Dates
using Statistics: mean, std

const ROOT    = normpath(joinpath(@__DIR__, ".."))
const OUTDIR  = joinpath(ROOT, "results", "jlab_crosscheck_ellipse_polarization")
const DATADIR = "/Volumes/DATA_SSD/ADominguez/DATA/ANCLAJES/ANC_GoMW_deep"
const MATPATH = joinpath(DATADIR, "ARE-T2000-LR75DW-NS10885-Z715-INS14-REC18.mat")
mkpath(OUTDIR)

# ── 1. Real ARE mooring data: bin 1, a 512-sample window ────────────────────
isfile(MATPATH) || error("missing mooring file: $MATPATH (is DATA_SSD mounted?)")
r = ANCMooringLoader(MATPATH)
p = anc_mooring_profiles(r)

N = 512
idx0 = 2000                     # arbitrary but fixed/reproducible interior window, avoids deployment start-up
idx = idx0:(idx0 + N - 1)
dt_hours = Dates.value(p.time[idx0 + 1] - p.time[idx0]) / 3.6e6   # from real timestamps, NOT assumed
u = p.u[idx, 1]
v = p.v[idx, 1]
(any(!isfinite, u) || any(!isfinite, v)) && error("non-finite values in the selected window -- pick a different idx0")

nw = 4.0
K  = 7                          # 2*floor(nw)-1, passed explicitly on both sides

writedlm(joinpath(OUTDIR, "uv.csv"), hcat(u, v), ',')
open(joinpath(OUTDIR, "dt_hours.txt"), "w") do io
    println(io, dt_hours)
end
println("Real ARE/LR75DW data: N=$N samples, dt=$dt_hours h, window ",
        p.time[idx0], " to ", p.time[idx0 + N - 1])
println("u range: ", extrema(u), "  v range: ", extrema(v), " m/s")

# ── 2. Run real jLab: sleptap -> mspec -> specdiag/polparams ───────────────
mscript = joinpath(ROOT, "scripts", "jlab_crosscheck_ellipse_polarization.m")
if "--skip-matlab" in ARGS
    println("--skip-matlab: reusing existing MATLAB outputs in $OUTDIR")
else
    matlab_bin = "/Applications/MATLAB_R2023a.app/bin/matlab"
    isfile(matlab_bin) || error("matlab binary not found at $matlab_bin -- update this path")
    println("Running real jLab sleptap/mspec/specdiag/polparams via matlab -batch ...")
    run(`$matlab_bin -batch "run('$mscript')"`)
end

# ── 3. Julia side: same pipeline via the public API ────────────────────────
est = ellipse_polarization(u, v; dt_hours=dt_hours, nw=nw, ntapers=K, ci=false)

# ── 4. Compare, aligned by FFT bin index (robust to any f=0/Nyquist-row
#       convention difference between mspec's grid and Julia's strictly-
#       positive-frequency grid) ───────────────────────────────────────────
m = readdlm(joinpath(OUTDIR, "jlab_polparams.csv"), ',')
# columns: bin, freq, d1, d2, theta, nu, P, alpha, beta_re, beta_im
m_bin = round.(Int, m[:, 1])

jl_bin = round.(Int, est.freq .* N .* dt_hours)

common = intersect(m_bin, jl_bin)
isempty(common) && error("no overlapping frequency bins between jLab and Julia output -- check freq grid conventions")

mi = Dict(b => i for (i, b) in enumerate(m_bin))
ji = Dict(b => i for (i, b) in enumerate(jl_bin))

function relerr(jlv, mv)
    d = abs.(jlv .- mv)
    return maximum(d) / (maximum(abs.(mv)) + 1e-300)
end

# theta's natural period is pi (an ellipse axis, not a directed vector,
# same reasoning as ValTools' own _jackknife_ci_circular_half_angle) -- a
# naive abs(a-b) reports a near-maximal "error" whenever the true values
# straddle the +-pi/2 branch cut even though e.g. -1.5701 and 1.5697 are
# the same orientation to within 0.0004 rad. Compare via the doubled angle
# on the circle instead.
function circular_absdiff_theta(a, b)
    d = atan.(sin.(2 .* (a .- b)), cos.(2 .* (a .- b))) ./ 2
    return abs.(d)
end

bins = sort(collect(common))
jl_theta = [est.theta[ji[b]] for b in bins]
jl_nu    = [est.nu[ji[b]] for b in bins]
jl_P     = [est.P[ji[b]] for b in bins]
jl_alpha = [est.alpha[ji[b]] for b in bins]
jl_beta  = [est.beta[ji[b]] for b in bins]
jl_d1    = [est.d1[ji[b]] for b in bins]
jl_d2    = [est.d2[ji[b]] for b in bins]
jl_power = jl_d1 .+ jl_d2

m_theta = [m[mi[b], 5] for b in bins]
m_nu    = [m[mi[b], 6] for b in bins]
m_P     = [m[mi[b], 7] for b in bins]
m_alpha = [m[mi[b], 8] for b in bins]
m_beta  = ComplexF64.([m[mi[b], 9] for b in bins], [m[mi[b], 10] for b in bins])
m_d1    = [m[mi[b], 3] for b in bins]
m_d2    = [m[mi[b], 4] for b in bins]

# Low-power bins (far below the record's dominant peak) are inherently
# noise-dominated -- P/alpha/nu/theta are all poorly determined there by
# construction, regardless of implementation, and near-zero values make
# relative-difference metrics blow up on tiny absolute noise. Report both
# the raw (all-bins) and a power-filtered (>=10% of peak power) summary so
# a few noisy low-SNR bins don't masquerade as a formula bug.
strong = jl_power .>= 0.1 * maximum(jl_power)

println("\n--- Scale-invariant quantities: ALL bins (includes noise-dominated low-power bins) ---")
println("theta  max circular absdiff (rad): ", maximum(circular_absdiff_theta(jl_theta, m_theta)))
println("nu     max rel diff: ", relerr(jl_nu, m_nu))
println("P      max rel diff: ", relerr(jl_P, m_P))
println("alpha  max rel diff: ", relerr(jl_alpha, m_alpha))
println("beta   max rel diff: ", relerr(abs.(jl_beta), abs.(m_beta)))

println("\n--- Same, restricted to bins with power >= 10% of peak (", count(strong), "/", length(bins),
        " bins -- this is the meaningful check) ---")
println("theta  max circular absdiff (rad): ", maximum(circular_absdiff_theta(jl_theta[strong], m_theta[strong])))
println("nu     max rel diff: ", relerr(jl_nu[strong], m_nu[strong]))
println("P      max rel diff: ", relerr(jl_P[strong], m_P[strong]))
println("alpha  max rel diff: ", relerr(jl_alpha[strong], m_alpha[strong]))
println("beta   max rel diff: ", relerr(abs.(jl_beta[strong]), abs.(m_beta[strong])))

println("\n--- Scale-DEPENDENT quantities (this is the actual rule-12 anchor) ---")
ratio_d1 = jl_d1 ./ m_d1
ratio_d2 = jl_d2 ./ m_d2
println("d1 ratio (julia/matlab): mean=", round(mean(ratio_d1); digits=6),
        " std=", round(std(ratio_d1); digits=6),
        " -- CONSTANT ratio = normalization convention difference (benign, document it);",
        " VARYING ratio = a real formula bug")
println("d2 ratio (julia/matlab): mean=", round(mean(ratio_d2); digits=6),
        " std=", round(std(ratio_d2); digits=6))

# Report the strongest peak in total energy (d1+d2) as the record's dominant
# ellipse, for a human sanity read (e.g. does it land near the local
# inertial or M2 tidal frequency?) -- not part of the pass/fail criterion.
peak = bins[argmax(jl_d1 .+ jl_d2)]
println("\nDominant energy peak at bin $peak (f=", peak / (N * dt_hours), " cyc/hr, period=",
        round(1 / (peak / (N * dt_hours)); digits=2), " h)")
println("Julia @ peak: theta=", est.theta[ji[peak]], " P=", est.P[ji[peak]])
println("jLab  @ peak: theta=", m[mi[peak], 5], " P=", m[mi[peak], 7])
