# Independent cross-check of ValTools.jl's variance-ellipse formula
# (`current_ellipse_metrics` in src/Metrics/stats.jl and
# `mooring_variance_ellipse` in src/Observations/mooring.jl -- both
# implement byte-identical math: eigendecomposition of the u,v anomaly
# covariance matrix) against a SEPARATE MATLAB implementation of the same,
# standard "principal axis" / variance-ellipse method.
#
# NOT a jLab crosscheck. jLab's own jEllipse/ellparams.m computes the
# MODULATED ellipse of an analytic signal -- a different, time-varying
# quantity, already ported and MATLAB-verified in ValTools as `ellipsefit`
# (see test/jlab/test_ellipse.jl). jLab has no princax-equivalent function
# for the static, whole-record covariance ellipse this script checks (grep
# for "princax" and "ellipse" across the local jLab install turned up
# nothing), so the companion .m script implements the formula directly via
# MATLAB's own `eig()` -- an independent second implementation of the
# standard method (e.g. Emery & Thomson, "Data Analysis Methods in
# Physical Oceanography"), not a re-run of the same code path. If the two
# disagree, the bug is in one of the two eigenvalue derivations or the
# inclination-angle convention.
#
# Data: real ANC_GoMW_deep mooring, ARE site, two ADCP instruments --
#   ARE-T2000-LR75DW-NS10885-Z715-INS14-REC18.mat   (mid-depth,  ~715 m)
#   ARE-T2000-WH600DW-NS10732-Z1980-INS14-REC18.mat (bottom boundary layer, ~1980 m)
# from /Volumes/DATA_SSD/ADominguez/DATA/ANCLAJES/ANC_GoMW_deep/ -- the
# same format ANCMooringLoader reads (see
# src/Observations/anc_gomw_mooring.jl). Uses bin 1 (first range cell) at
# each instrument, matching mooring_variance_ellipse's own default
# depth_idx=nothing behavior (first column when unspecified).
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/crosscheck_variance_ellipse.jl
#   julia --project=envs/cpu scripts/crosscheck_variance_ellipse.jl --skip-matlab   # reuse cached MATLAB outputs

using ValTools, MAT, DelimitedFiles

const ROOT    = normpath(joinpath(@__DIR__, ".."))
const OUTDIR  = joinpath(ROOT, "results", "crosscheck_variance_ellipse")
const DATADIR = "/Volumes/DATA_SSD/ADominguez/DATA/ANCLAJES/ANC_GoMW_deep"
mkpath(OUTDIR)

const SITES = [
    ("LR75DW_mid",     joinpath(DATADIR, "ARE-T2000-LR75DW-NS10885-Z715-INS14-REC18.mat")),
    ("WH600DW_bottom", joinpath(DATADIR, "ARE-T2000-WH600DW-NS10732-Z1980-INS14-REC18.mat")),
]

# ── 1. Load real mooring data, export bin-1 u,v for MATLAB ─────────────────
println("Loading real ANC_GoMW_deep mooring data (ARE site) ...")
for (name, path) in SITES
    isfile(path) || error("missing mooring file: $path (is DATA_SSD mounted?)")
    r = ANCMooringLoader(path)
    p = anc_mooring_profiles(r)
    u1 = p.u[:, 1]
    v1 = p.v[:, 1]
    valid = isfinite.(u1) .& isfinite.(v1)
    writedlm(joinpath(OUTDIR, "$(name)_uv.csv"), hcat(u1[valid], v1[valid]), ',')
    println("  $name: site=$(p.site) lon=$(round(p.lon; digits=3)) lat=$(round(p.lat; digits=3)) ",
            "bin=$(p.bins[1]) n=$(count(valid))/$(length(u1))")
end

# ── 2. Run the independent MATLAB implementation ────────────────────────────
mscript = joinpath(ROOT, "scripts", "crosscheck_variance_ellipse.m")
if "--skip-matlab" in ARGS
    println("\n--skip-matlab: reusing existing MATLAB outputs in $OUTDIR")
else
    matlab_bin = "/Applications/MATLAB_R2023a.app/bin/matlab"
    isfile(matlab_bin) || error("matlab binary not found at $matlab_bin -- update this path")
    println("\nRunning independent MATLAB variance-ellipse computation ...")
    run(`$matlab_bin -batch "run('$mscript')"`)
end

# ── 3. Compare against ValTools' current_ellipse_metrics ───────────────────
# obs=model=the same real record isolates the exact covariance-ellipse
# formula that mooring_variance_ellipse also uses (byte-identical code,
# compare src/Metrics/stats.jl:219-233 to src/Observations/mooring.jl:56-74).
# Note: current_ellipse_metrics rounds its outputs (semi-major/minor to 4
# digits, inclination to 2, EKE to 6) -- tolerances below are set loose
# enough to absorb that rounding, not to hide a real formula bug (which
# would typically show up as an O(0.01-1) discrepancy from a wrong
# eigenvalue, radians/degrees mixup, or axis-ordering error).
function report(name, jl, m; atol)
    d = abs(jl - m)
    ok = d < atol
    println(rpad(name, 14), "  julia=", jl, "  matlab=", m, "  absdiff=", d, ok ? "  OK" : "  **FAIL**")
    return ok
end

allpass = true
for (name, _) in SITES
    uv = readdlm(joinpath(OUTDIR, "$(name)_uv.csv"), ',')
    u, v = uv[:, 1], uv[:, 2]
    cem = current_ellipse_metrics(u, v, u, v)

    mvals = vec(readdlm(joinpath(OUTDIR, "$(name)_matlab.csv"), ','))
    m_semi_major, m_semi_minor, m_inclination, m_eke = mvals

    println("\n=== $name ===")
    ok = true
    ok &= report("semi_major",  cem["obs_semi_major"],  m_semi_major;  atol=1e-3)
    ok &= report("semi_minor",  cem["obs_semi_minor"],  m_semi_minor;  atol=1e-3)
    ok &= report("inclination", cem["obs_inclination"], m_inclination; atol=2e-2)
    ok &= report("EKE",         cem["obs_EKE"],         m_eke;         atol=1e-5)
    println(ok ? "PASS" : "MISMATCH")
    global allpass &= ok
end

println()
if allpass
    println("PASS: ValTools' variance-ellipse formula matches an independent ",
            "MATLAB eig()-based implementation on real ARE mooring data")
else
    println("MISMATCH -- investigate before trusting current_ellipse_metrics / mooring_variance_ellipse")
    exit(1)
end
