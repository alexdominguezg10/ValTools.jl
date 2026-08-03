# Cross-check ValTools.jl's multivariate `instmom` port against REAL jLab
# MATLAB output (working-agreement rule 11), following the established
# jlab_crosscheck_* export -> matlab -> diff pattern.
#
# Scope note: `multivariate_ridges` reuses `ridgechains_jlab` (the ridge
# chaining engine) completely unchanged from the already jLab-verified
# bivariate path (see jlab_crosscheck_wavetrans_nd.jl / the GOMED-era
# jlab_crosscheck_eddyridges.m history) -- the only genuinely new math is
# the joint amplitude/frequency feed, which is exactly `instmom`'s
# `jointmom` formula. That's what this script verifies directly against
# real jLab. Combined with the N=2 consistency test (_instfreq_joint_nd
# vs. the validated rotary _instfreq_joint, in test_multivariate.jl), this
# covers the new code without needing a separate, calling-convention-heavy
# `ridgewalk` N=3 cross-check.
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/jlab_crosscheck_multivariate.jl

using DelimitedFiles
using ValTools.JLab

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "results", "jlab_crosscheck_multivariate")
mkpath(OUTDIR)

# ── 1. Export a deterministic 3-channel complex analytic signal ────────────
# instmom takes X directly (not necessarily a wavetrans output) -- use a
# genuinely complex signal per channel, with a time-varying envelope, so
# amplitude/frequency/bandwidth/curvature all vary meaningfully and every
# term in the joint formulas gets exercised (not just the phase term).
# MATLAB's csvread can't parse complex numbers, so write interleaved
# [re1 im1 re2 im2 re3 im3] columns -- the .m script reconstructs complex X.
N = 300
t = collect(0.0:N-1)
env(a0, decay) = a0 .* exp.(-decay .* ((t .- N / 2) ./ N) .^ 2)
phase = cumsum(0.4 .+ 0.0008 .* t)
x1 = env(2.0, 3.0) .* exp.(im .* phase)
x2 = env(1.0, 1.0) .* exp.(im .* (phase .+ 0.3))
x3 = env(0.5, 2.0) .* exp.(im .* (phase .+ 1.1))
X = hcat(x1, x2, x3)

Xri = hcat(real(x1), imag(x1), real(x2), imag(x2), real(x3), imag(x3))
writedlm(joinpath(OUTDIR, "signals.csv"), Xri, ',')

# ── 2. Run real jLab ──────────────────────────────────────────────────────
mscript = joinpath(ROOT, "scripts", "jlab_crosscheck_multivariate.m")
println("Running real jLab instmom (jointmom) via matlab -batch ...")
run(`matlab -batch "run('$mscript')"`)

# ── 3. Compare ───────────────────────────────────────────────────────────────
W = reshape(X, N, 1, 3)   # joint form: (time, scale=1, channels)
ja_jl, jomega_jl, jupsilon_jl, jxi_jl = instmom(W)
ja_jl, jomega_jl, jupsilon_jl, jxi_jl = vec(ja_jl), vec(jomega_jl), vec(jupsilon_jl), vec(jxi_jl)

ja_m = vec(readdlm(joinpath(OUTDIR, "jl_ja.csv"), ','))
jomega_m = vec(readdlm(joinpath(OUTDIR, "jl_jomega.csv"), ','))
jupsilon_m = vec(readdlm(joinpath(OUTDIR, "jl_jupsilon.csv"), ','))
jxi_m = vec(readdlm(joinpath(OUTDIR, "jl_jxi.csv"), ','))

interior = 11:(N - 10)  # avoid one-sided endpoint-derivative edge effects for the tightest check

function report(name, a, b)
    rel = maximum(abs.(a[interior] .- b[interior])) / (maximum(abs.(b[interior])) + 1e-300)
    println(rpad(name, 12), " max rel diff (interior) = ", rel)
    return rel
end

r1 = report("ja", ja_jl, ja_m)
r2 = report("jomega", jomega_jl, jomega_m)
r3 = report("jupsilon", jupsilon_jl, jupsilon_m)
r4 = report("jxi", jxi_jl, jxi_m)

println()
if all(r < 1e-8 for r in (r1, r2, r3, r4))
    println("PASS: instmom joint moments match real jLab instmom/jointmom exactly")
else
    println("MISMATCH -- investigate before trusting the multivariate port")
    exit(1)
end
