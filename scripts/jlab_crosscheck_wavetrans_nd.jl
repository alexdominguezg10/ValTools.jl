# Cross-check ValTools' N-D wavetrans (trailing-dims signal batching) against
# REAL jLab MATLAB output — working-agreement rule: verify jLab ports against
# jLab's actual output on the same data, not just its source or our own tests.
#
# Flow (all in one driver, following the jlab_crosscheck_* harness pattern):
#   1. Export a deterministic (N x 3) signal matrix + a shared frequency grid.
#   2. Run scripts/jlab_crosscheck_wavetrans_nd.m via `matlab -batch`: jLab's
#      wavetrans on the full matrix (its own column batching) AND per column
#      (internal consistency), 'mirror' boundary (the variant our port
#      matches timeseries_boundary.m on exactly).
#   3. Diff our wavetrans(X; boundary=:mirror) against jLab's matrix output.
#
# Signals are pre-detrended before export so each side's internal detrend
# (if any) is a numerical no-op — removes detrend-convention ambiguity from
# the comparison.
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/jlab_crosscheck_wavetrans_nd.jl

using DelimitedFiles
using ValTools.JLab
using ValTools.JLab: _detrend_signal

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "results", "jlab_crosscheck_wavetrans_nd")
mkpath(OUTDIR)

# ── 1. Export ────────────────────────────────────────────────────────────────
N = 300
t = collect(0.0:N-1)
x1 = cos.(2π * 0.05 .* t)                                    # pure tone
x2 = cos.(2π .* (0.02 .+ 0.0002 .* t) .* t)                  # chirp
x3 = sin.(2π * 0.11 .* t) .+ 0.5 .* cos.(2π * 0.03 .* t)     # two tones
X = _detrend_signal(hcat(x1, x2, x3))

fs = morsespace(3.0, 8.0, N)   # shared grid, our defaults (gamma=3, beta=8)

writedlm(joinpath(OUTDIR, "signals.csv"), X, ',')
writedlm(joinpath(OUTDIR, "fs.csv"), fs, ',')

# ── 2. Run real jLab ─────────────────────────────────────────────────────────
mscript = joinpath(ROOT, "scripts", "jlab_crosscheck_wavetrans_nd.m")
println("Running real jLab wavetrans via matlab -batch ...")
run(`matlab -batch "run('$mscript')"`)

# ── 3. Compare ───────────────────────────────────────────────────────────────
summary = readdlm(joinpath(OUTDIR, "summary.csv"), ',', skipstart=1)
max_internal = Float64(summary[1, 3])
println("\njLab internal (matrix call vs per-column call) max |diff| = $max_internal")
max_internal < 1e-12 || error("jLab's own matrix path differs from its per-column path?!")

wt_ours, fs_ours = wavetrans(X; fs=fs, boundary=:mirror)   # (N, n_fs, 3), shared grid
@assert fs_ours ≈ fs

# Pass criteria (in order of what this harness is actually testing):
#  (b) our N-D batched path is identical to our own per-column scalar path
#      — the N-D claim itself;
#  (c) our output agrees with real jLab to the scalar port's established
#      tolerance. The scalar port matches jLab EXACTLY (~1e-16 rel) at low
#      and mid frequencies; at the very highest frequencies there is a
#      known, pre-existing near-zero filter-bank TAIL difference in
#      morsewave_freq vs jLab's morsewave (first seen in the 2026-08-02
#      msvd cross-check), which shows up here as up to ~2e-4 relative
#      deviation concentrated at the record edges (broadband edge
#      transients are what excite the differing tail). That is a scalar
#      calibration question, independent of (and unchanged by) the N-D
#      batching this harness verifies.
worst_nd_vs_scalar = 0.0
worst_vs_jlab = 0.0
worst_vs_jlab_interior = 0.0
interior = 51:(N - 50)
for k in 1:3
    re = readdlm(joinpath(OUTDIR, "jl_wt_$(k)_re.csv"), ',')
    im_ = readdlm(joinpath(OUTDIR, "jl_wt_$(k)_im.csv"), ',')
    wt_jl = complex.(re, im_)
    ours_nd = wt_ours[:, :, k]
    ours_scalar, _ = wavetrans(X[:, k]; fs=fs, boundary=:mirror)
    scale = maximum(abs.(wt_jl))

    nd_vs_scalar = maximum(abs.(ours_nd .- ours_scalar)) / scale
    vs_jlab = maximum(abs.(ours_nd .- wt_jl)) / scale
    vs_jlab_int = maximum(abs.(ours_nd[interior, :] .- wt_jl[interior, :])) / scale
    global worst_nd_vs_scalar = max(worst_nd_vs_scalar, nd_vs_scalar)
    global worst_vs_jlab = max(worst_vs_jlab, vs_jlab)
    global worst_vs_jlab_interior = max(worst_vs_jlab_interior, vs_jlab_int)
    println("col $k: N-D vs our scalar = $nd_vs_scalar, " *
            "vs jLab = $vs_jlab (interior $vs_jlab_int)")
end

println()
ok_nd = worst_nd_vs_scalar < 1e-13
ok_jlab = worst_vs_jlab < 5e-4 && worst_vs_jlab_interior < 1e-5
if ok_nd && ok_jlab
    println("PASS: (a) jLab matrix ≡ jLab per-column ($max_internal); " *
            "(b) our N-D ≡ our scalar ($worst_nd_vs_scalar); " *
            "(c) ours vs real jLab within the scalar port's established " *
            "tolerance ($worst_vs_jlab overall, $worst_vs_jlab_interior interior)")
else
    println("MISMATCH: nd_vs_scalar=$worst_nd_vs_scalar, vs_jlab=$worst_vs_jlab " *
            "(interior $worst_vs_jlab_interior) — investigate before trusting the N-D port")
    exit(1)
end
