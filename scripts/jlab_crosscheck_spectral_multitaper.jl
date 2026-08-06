# Cross-check ValTools.jl's `spectral_multitaper` (the modern, non-deprecated
# multitaper spectrum estimator, Multitaper.jl-backed) against REAL jLab
# `mspec.m` output -- another zero-crosscheck gap flagged in the ValTools
# functionality/validation survey (2026-08-05): `spectral_multitaper` is
# unit/peak-location tested only (synthetic cosine+noise, checking the peak
# lands near f0), never checked for ABSOLUTE power-level correctness against
# an external reference. Peak-location checks are scale-invariant by
# construction (working-agreement rule 12) -- a uniform normalization bug in
# the ValTools<->Multitaper.jl wrapper would sail through them undetected,
# exactly like the rotary_spectrum `/n` bug this rule is named for.
#
# Unlike the other jlab_crosscheck_* scripts, `spectral_multitaper` doesn't
# reimplement jLab's tapering/spectrum math -- it delegates entirely to the
# third-party Multitaper.jl package (`ext/ValToolsMultitaperExt/spectral_multitaper.jl`).
# So this script checks two DIFFERENT things at once, both real bugs Julia's
# own tests can't see:
#   (a) does ValTools' wrapper pass dt/nw/K through correctly (peak location,
#       shape) -- covered by existing tests too, included here as a sanity
#       check;
#   (b) does the ABSOLUTE power level (in physical units) match -- checked
#       via each side's OWN Parseval/variance-recovery identity
#       independently (a genuine external, known-in-advance reference: the
#       signal's own sample variance) -- not compared against each other,
#       but against ground truth on each side. THIS is the check existing
#       tests structurally cannot perform.
#
# jLab's mspec.m defaults to RADIAN frequency (cos(f*t)); Multitaper.jl (and
# hence ValTools' `spectral_multitaper`, confirmed by test_spectral.jl's own
# f0=0.5 direct-match peak test) uses CYCLIC frequency (cos(2*pi*f*t)) -- the
# MATLAB side below passes mspec's own 'cyclic' flag so both sides are
# genuinely in the same units; skipping that would produce a spurious 2*pi
# mismatch that has nothing to do with a real bug.
#
# REAL data: same ARE/LR75DW mooring u-component and 512-sample window as
# jlab_crosscheck_ellipse_polarization.jl (real dt=0.5 h from the file's own
# timestamps). N<=512 for the same sleptap.m spline-interpolation reason
# documented there.
#
# Usage (from ValTools.jl root, needs MATLAB + MAT.jl):
#   julia --project=envs/cpu scripts/jlab_crosscheck_spectral_multitaper.jl
#   julia --project=envs/cpu scripts/jlab_crosscheck_spectral_multitaper.jl --skip-matlab

using ValTools, MAT, ValTools.JLab, Multitaper, DelimitedFiles, Dates
using Statistics: mean, std, var

const ROOT    = normpath(joinpath(@__DIR__, ".."))
const OUTDIR  = joinpath(ROOT, "results", "jlab_crosscheck_spectral_multitaper")
const DATADIR = "/Volumes/DATA_SSD/ADominguez/DATA/ANCLAJES/ANC_GoMW_deep"
const MATPATH = joinpath(DATADIR, "ARE-T2000-LR75DW-NS10885-Z715-INS14-REC18.mat")
mkpath(OUTDIR)

# ── 1. Real ARE mooring data: bin 1, u-component, 512-sample window ────────
isfile(MATPATH) || error("missing mooring file: $MATPATH (is DATA_SSD mounted?)")
r = ANCMooringLoader(MATPATH)
p = anc_mooring_profiles(r)

N = 512
idx0 = 2000                     # same window as jlab_crosscheck_ellipse_polarization.jl
idx = idx0:(idx0 + N - 1)
dt = Dates.value(p.time[idx0 + 1] - p.time[idx0]) / 3.6e6   # hours, from real timestamps
x = p.u[idx, 1]
any(!isfinite, x) && error("non-finite values in the selected window -- pick a different idx0")

nw = 4.0
K  = 7

writedlm(joinpath(OUTDIR, "x.csv"), x, ',')
open(joinpath(OUTDIR, "dt_hours.txt"), "w") do io
    println(io, dt)
end
println("Real ARE/LR75DW u-component: N=$N samples, dt=$dt h, window ",
        p.time[idx0], " to ", p.time[idx0 + N - 1])
println("Sample variance of x (the external reference both sides must recover): ", var(x; corrected=false))

# ── 2. Run real jLab: sleptap -> mspec (cyclic frequency) ──────────────────
mscript = joinpath(ROOT, "scripts", "jlab_crosscheck_spectral_multitaper.m")
if "--skip-matlab" in ARGS
    println("--skip-matlab: reusing existing MATLAB outputs in $OUTDIR")
else
    matlab_bin = "/Applications/MATLAB_R2023a.app/bin/matlab"
    isfile(matlab_bin) || error("matlab binary not found at $matlab_bin -- update this path")
    println("Running real jLab sleptap/mspec via matlab -batch ...")
    run(`$matlab_bin -batch "run('$mscript')"`)
end

# ── 3. Julia side ────────────────────────────────────────────────────────
est = JLab.spectral_multitaper(x, dt; nw=nw, ntapers=K)
jl_freq = est.freq
jl_power = est.power

# ── 4. Each side's OWN Parseval/variance-recovery check ────────────────────
# One-sided spectrum, cyclic frequency, positive freqs only (both sides
# exclude f=0 the same way mspec's own docstring formula does): variance ≈
# 2*df*(sum(S[2:end-1]) + S[end]/2) for even N (Nyquist counted once, not
# doubled) -- mspec's own documented formula, cyclic-frequency form (drop
# the 2*(1/2/pi) radian-specific prefactor since df is already in
# cycles/time here).
function parseval_variance(freq::AbstractVector, S::AbstractVector; even_N::Bool)
    pos = freq .> 0
    f_pos = freq[pos]; S_pos = S[pos]
    df = f_pos[2] - f_pos[1]
    return even_N ? 2 * df * (sum(S_pos[1:end-1]) + S_pos[end] / 2) : 2 * df * sum(S_pos)
end

true_var = var(x; corrected=false)
jl_var_recovered = parseval_variance(jl_freq, jl_power; even_N=iseven(N))

m = readdlm(joinpath(OUTDIR, "jlab_mspec.csv"), ',')
m_freq, m_S = m[:, 1], m[:, 2]
m_var_recovered = parseval_variance(m_freq, m_S; even_N=iseven(N))

println("\n--- Parseval/variance-recovery, EACH side against the signal's own known variance ---")
println("True sample variance:        ", true_var)
println("Julia recovered (Parseval):  ", jl_var_recovered, "  (rel err ", abs(jl_var_recovered - true_var) / true_var, ")")
println("jLab  recovered (Parseval):  ", m_var_recovered, "  (rel err ", abs(m_var_recovered - true_var) / true_var, ")")
println("-- both should be within a few % of true_var (periodogram-like recovery; multitaper smoothing biases it slightly).")
println("-- if ONE side is off by a large, clean factor (2, 4, 2*pi, N, ...) while the other isn't, that side has a normalization bug.")

# ── 5. Direct bin-by-bin comparison (peak location + shape, and the
#       power ratio for diagnosing any residual constant-factor mismatch) ──
jl_bin = round.(Int, jl_freq .* N .* dt)
m_bin  = round.(Int, m_freq .* N .* dt)
common = sort(collect(intersect(jl_bin, m_bin)))
ji = Dict(b => i for (i, b) in enumerate(jl_bin))
mi = Dict(b => i for (i, b) in enumerate(m_bin))

jl_S_common = [jl_power[ji[b]] for b in common]
m_S_common  = [m_S[mi[b]] for b in common]
ratio = jl_S_common ./ m_S_common

println("\n--- Direct comparison at matched FFT bins ---")
println("power ratio (julia/matlab): mean=", round(mean(ratio); digits=6),
        " std=", round(std(ratio); digits=6),
        " -- CONSTANT ratio = normalization convention difference;",
        " VARYING ratio (esp. near the peak) = a real bug")

jl_peak_bin = common[argmax(jl_S_common)]
m_peak_bin  = common[argmax(m_S_common)]
println("Julia peak at bin ", jl_peak_bin, " (f=", jl_peak_bin / (N * dt), " cyc/hr, period=",
        round(1 / (jl_peak_bin / (N * dt)); digits=2), " h)")
println("jLab  peak at bin ", m_peak_bin, " (f=", m_peak_bin / (N * dt), " cyc/hr)")
