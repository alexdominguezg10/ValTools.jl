# Cross-check the ACTUAL ridge list used by flare_fig3_rotary_wavelet_ridge.jl
# (both CCW and CW senses, full 2005 record N=8761, the eddy-band frequency
# grid -- fmax_ratio=2, fmin_ratio=1/64, matching jLab's own real
# jFigures/makefigs_gulfcensus.m:460 call) against real jLab ridgewalk.m.
#
# This is a DIFFERENT, more narrowly-scoped check than
# jlab_crosscheck_flare_fig23.jl's Part B, which used a 2000-hour subset,
# a generic (non-eddy-band) frequency grid, and CCW only -- useful for
# validating the wavelet-transform/ridge-chaining MATH in general, but not
# a direct check of the specific ridge table flare_fig3_ridges.csv
# actually reports. This script targets that table directly.
#
# Both known jLab-port scale bugs (Metrics.rotary_spectrum's extra /n,
# JLab.rotary_wavetrans's missing 1/sqrt(2)) were fixed 2026-08-05 before
# this script was written, so amplitudes should now match directly with
# NO rescaling -- if they don't, that is itself new information, not an
# already-known/expected factor to explain away.
#
# Usage (from ValTools.jl root, needs MATLAB installed):
#   julia --project=envs/cpu scripts/jlab_crosscheck_fig3_ridges.jl
#   julia --project=envs/cpu scripts/jlab_crosscheck_fig3_ridges.jl --skip-matlab   # reuse cached jLab output
#
# Outputs (results/jlab_crosscheck_fig3_ridges/):
#   uv_full.csv, fs_eddy.csv          -- exported inputs (Julia -> MATLAB)
#   jl_wp_re/im.csv, jl_wn_re/im.csv  -- jLab's raw wavetrans.m output
#   jl_ridge_{ccw,cw}_{ir,omega,wr_re,wr_im}.csv -- jLab's raw ridgewalk.m output
#   jlab_ridges_ccw.csv, jlab_ridges_cw.csv -- jLab ridges parsed into the
#     SAME summary format as results/flare_reproduction/flare_fig3_ridges.csv
#     (start,stop,npoints,omega_radph,kappa_cmps,sense,period_days), ready
#     to diff directly against that file.

using ValTools, ValTools.JLab
using Dates, Statistics, Printf, DelimitedFiles

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "results", "jlab_crosscheck_fig3_ridges")
const CSV_PATH = "/Volumes/DATA_SSD/gdp_44000_2005.csv"
const OURS_CSV = joinpath(ROOT, "results", "flare_reproduction", "flare_fig3_ridges.csv")
mkpath(OUTDIR)

const GAMMA = 3.0
const BETA = 3.0

function read_gdp_csv(path::String)
    lines = readlines(path)
    header = split(lines[1], ",")
    col = Dict(name => i for (i, name) in enumerate(header))
    n = length(lines) - 2
    ve = Vector{Float64}(undef, n)
    vn = Vector{Float64}(undef, n)
    lat = Vector{Float64}(undef, n)
    for (k, line) in enumerate(@view lines[3:end])
        f = split(line, ",")
        lat[k] = parse(Float64, f[col["latitude"]])
        ve[k] = parse(Float64, f[col["ve"]])
        vn[k] = parse(Float64, f[col["vn"]])
    end
    return ve, vn, lat
end

# ── Load & prepare (identical to flare_fig3_rotary_wavelet_ridge.jl) ────
ve, vn, lat = read_gdp_csv(CSV_PATH)
u = ve .* 100.0   # cm/s
v = vn .* 100.0
n = length(u)
mean_lat = mean(lat)
println("Loaded $n hourly rows (mean lat $(round(mean_lat, digits=3))N)")

f0_cph = inertial_frequency(mean_lat)
f0_radph = 2π * f0_cph

# Exact same eddy-band grid construction as flare_fig3_rotary_wavelet_ridge.jl
const_fmax_ratio, const_fmin_ratio = 2.0, 1 / 64
P_w, _, _ = morseprops(GAMMA, BETA)
f_high_eddy = const_fmax_ratio * f0_radph
f_low_eddy = max(const_fmin_ratio * f0_radph, π * P_w / n)
fs_eddy = morsespace(GAMMA, BETA, n; f_high=f_high_eddy, f_low=f_low_eddy, density=16)
println("eddy-band grid: $(length(fs_eddy)) freqs, f0=$(round(f0_radph, digits=5)) rad/h")

writedlm(joinpath(OUTDIR, "uv_full.csv"), hcat(u, v), ',')
writedlm(joinpath(OUTDIR, "fs_eddy.csv"), fs_eddy, ',')

# ── Run real jLab ─────────────────────────────────────────────────────────
mscript = joinpath(ROOT, "scripts", "jlab_crosscheck_fig3_ridges.m")
matlab_bin = "/Applications/MATLAB_R2023a.app/bin/matlab"
isfile(matlab_bin) || error("matlab binary not found at $matlab_bin -- update this path")
const SKIP_MATLAB = "--skip-matlab" in ARGS
if SKIP_MATLAB
    println("\n--skip-matlab: reusing existing jLab outputs in $OUTDIR")
else
    println("\nRunning real jLab (wavetrans + ridgewalk, both senses, full record) via matlab -batch ...")
    println("(this record is 4x longer than jlab_crosscheck_flare_fig23.jl's Part B subset -- expect a longer run)")
    run(`$matlab_bin -batch "run('$mscript')"`)
end

# ── Parse jLab's raw ridgewalk.m output (NaN-separated ir/omega/wr) into
# discrete ridge segments, in the same summary form as flare_fig3_ridges.csv
function parse_jlab_ridges(ir::Vector{Float64}, omega::Vector{Float64},
                            wr_re::Vector{Float64}, wr_im::Vector{Float64}, sense::String)
    rows = NamedTuple[]
    i = 1
    n_pts = length(ir)
    while i <= n_pts
        if isnan(ir[i])
            i += 1
            continue
        end
        j = i
        while j <= n_pts && !isnan(ir[j])
            j += 1
        end
        seg = i:(j - 1)
        start_idx = round(Int, ir[seg[1]])
        stop_idx = round(Int, ir[seg[end]])
        mean_omega = mean(omega[seg])              # rad/hour
        mean_kappa = mean(abs.(complex.(wr_re[seg], wr_im[seg])))   # cm/s
        period_days = mean_omega != 0 ? abs(2π / mean_omega) / 24 : Inf
        push!(rows, (; start=start_idx, stop=stop_idx, npoints=length(seg),
                       omega_radph=mean_omega, kappa_cmps=mean_kappa,
                       sense=sense, period_days=period_days))
        i = j
    end
    return rows
end

function write_ridge_csv(path::String, rows::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, "start,stop,npoints,omega_radph,kappa_cmps,sense,period_days")
        for r in rows
            @printf(io, "%d,%d,%d,%.6f,%.4f,%s,%.3f\n",
                    r.start, r.stop, r.npoints, r.omega_radph, r.kappa_cmps, r.sense, r.period_days)
        end
    end
end

ir_ccw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_ccw_ir.csv"), ','))
omega_ccw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_ccw_omega.csv"), ','))
wr_re_ccw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_ccw_wr_re.csv"), ','))
wr_im_ccw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_ccw_wr_im.csv"), ','))
ridges_ccw = parse_jlab_ridges(ir_ccw, omega_ccw, wr_re_ccw, wr_im_ccw, "ccw")
write_ridge_csv(joinpath(OUTDIR, "jlab_ridges_ccw.csv"), ridges_ccw)

ir_cw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_cw_ir.csv"), ','))
omega_cw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_cw_omega.csv"), ','))
wr_re_cw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_cw_wr_re.csv"), ','))
wr_im_cw = vec(readdlm(joinpath(OUTDIR, "jl_ridge_cw_wr_im.csv"), ','))
ridges_cw = parse_jlab_ridges(ir_cw, omega_cw, wr_re_cw, wr_im_cw, "cw")
write_ridge_csv(joinpath(OUTDIR, "jlab_ridges_cw.csv"), ridges_cw)

println("\njLab found $(length(ridges_ccw)) CCW ridge(s), $(length(ridges_cw)) CW ridge(s)")
println("Saved: $(joinpath(OUTDIR, "jlab_ridges_ccw.csv")), $(joinpath(OUTDIR, "jlab_ridges_cw.csv"))")

# ── Quick summary diff against our own flare_fig3_ridges.csv, if present ──
if isfile(OURS_CSV)
    ours = readdlm(OURS_CSV, ','; skipstart=1)
    our_ccw_n = count(==( "ccw"), ours[:, 8])
    our_cw_n = count(==("cw"), ours[:, 8])
    println("\n=== Ridge count comparison (ours vs jLab) ===")
    println("  CCW: ours=$our_ccw_n, jLab=$(length(ridges_ccw))")
    println("  CW:  ours=$our_cw_n, jLab=$(length(ridges_cw))")
    println("(Counts alone don't establish a match -- inspect timing/frequency overlap in")
    println(" jlab_ridges_{ccw,cw}.csv vs $OURS_CSV by hand, or extend this script to pair")
    println(" ridges by overlapping [start,stop] windows the way jlab_crosscheck_flare_fig23.jl does.)")
else
    println("\nNOTE: $OURS_CSV not found -- run flare_fig3_rotary_wavelet_ridge.jl first for a count comparison")
end
