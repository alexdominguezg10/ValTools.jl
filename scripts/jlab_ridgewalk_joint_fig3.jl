# Cheap follow-up to jlab_crosscheck_fig3_ridges.jl: calls jLab's JOINT
# (bivariate) ridgewalk.m form -- ridgewalk(dt, wp, wn, fs, P, M) -- reusing
# wp/wn already saved to disk by that earlier run, so wavetrans.m (the
# expensive step) is NOT recomputed. This is the correct jLab entry point
# for ValTools._rotary_ridge_properties_core's actual algorithm (joint
# local maxima on sqrt(|w+|^2+|w-|^2), split into CCW/CW post-hoc by which
# side locally dominates) -- the earlier univariate crosscheck compared
# against the wrong jLab mode; see jRidges/ridgewalk.m's own "Joint
# ridges" docstring section.
#
# Requires jlab_crosscheck_fig3_ridges.jl to have been run first (needs its
# saved jl_wp_re/im.csv, jl_wn_re/im.csv, fs_eddy.csv in the same OUTDIR).
#
# Usage:
#   julia --project=envs/cpu scripts/jlab_ridgewalk_joint_fig3.jl
#   julia --project=envs/cpu scripts/jlab_ridgewalk_joint_fig3.jl --skip-matlab

using Statistics, Printf, DelimitedFiles

const ROOT = normpath(joinpath(@__DIR__, ".."))
const OUTDIR = joinpath(ROOT, "results", "jlab_crosscheck_fig3_ridges")
const OURS_CSV = joinpath(ROOT, "results", "flare_reproduction", "flare_fig3_ridges.csv")
const OLD_CCW_CSV = joinpath(OUTDIR, "jlab_ridges_ccw.csv")     # wrong-mode univariate run, for reference
const OLD_CW_CSV = joinpath(OUTDIR, "jlab_ridges_cw.csv")

for f in ("jl_wp_re.csv", "jl_wp_im.csv", "jl_wn_re.csv", "jl_wn_im.csv", "fs_eddy.csv")
    isfile(joinpath(OUTDIR, f)) || error("$(joinpath(OUTDIR, f)) not found -- run jlab_crosscheck_fig3_ridges.jl first")
end

mscript = joinpath(ROOT, "scripts", "jlab_ridgewalk_joint_fig3.m")
matlab_bin = "/Applications/MATLAB_R2023a.app/bin/matlab"
isfile(matlab_bin) || error("matlab binary not found at $matlab_bin -- update this path")
if "--skip-matlab" in ARGS
    println("--skip-matlab: reusing existing joint-ridgewalk outputs in $OUTDIR")
else
    println("Running real jLab ridgewalk.m in JOINT mode (reuses saved wp/wn, skips wavetrans.m) ...")
    run(`$matlab_bin -batch "run('$mscript')"`)
end

# ── Parse the joint ridge chain into contiguous same-sense sub-runs,
# matching _rotary_ridge_properties_core's post-hoc delta-masking logic
# (sense = ccw where |wp| >= |wn| locally, cw otherwise), and report the
# JOINT amplitude sqrt(|wp|^2+|wn|^2) as kappa -- matching what
# ridgechains_jlab actually chains on internally (mag_c), not the
# single-side amplitude alone.
ir = vec(readdlm(joinpath(OUTDIR, "jl_ridge_joint_ir.csv"), ','))
omega = vec(readdlm(joinpath(OUTDIR, "jl_ridge_joint_omega.csv"), ','))
wrp = complex.(vec(readdlm(joinpath(OUTDIR, "jl_ridge_joint_wrp_re.csv"), ',')),
               vec(readdlm(joinpath(OUTDIR, "jl_ridge_joint_wrp_im.csv"), ',')))
wrn = complex.(vec(readdlm(joinpath(OUTDIR, "jl_ridge_joint_wrn_re.csv"), ',')),
               vec(readdlm(joinpath(OUTDIR, "jl_ridge_joint_wrn_im.csv"), ',')))
amp_p, amp_n = abs.(wrp), abs.(wrn)
joint_amp = sqrt.(amp_p .^ 2 .+ amp_n .^ 2)

function split_joint_ridges(ir, omega, amp_p, amp_n, joint_amp)
    rows = NamedTuple[]
    n_pts = length(ir)
    i = 1
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
        k = seg[1]
        while k <= seg[end]
            sense_k = amp_p[k] >= amp_n[k] ? "ccw" : "cw"
            m = k
            while m <= seg[end] && (amp_p[m] >= amp_n[m] ? "ccw" : "cw") == sense_k
                m += 1
            end
            run_idx = k:(m - 1)
            start_idx = round(Int, ir[run_idx[1]])
            stop_idx = round(Int, ir[run_idx[end]])
            mean_omega = mean(omega[run_idx])
            mean_kappa = mean(joint_amp[run_idx])
            period_days = mean_omega != 0 ? abs(2π / mean_omega) / 24 : Inf
            push!(rows, (; start=start_idx, stop=stop_idx, npoints=length(run_idx),
                           omega_radph=mean_omega, kappa_cmps=mean_kappa,
                           sense=sense_k, period_days=period_days))
            k = m
        end
        i = j
    end
    return rows
end

rows = split_joint_ridges(ir, omega, amp_p, amp_n, joint_amp)

function write_ridge_csv(path::String, rr::Vector{<:NamedTuple})
    open(path, "w") do io
        println(io, "start,stop,npoints,omega_radph,kappa_cmps,sense,period_days")
        for r in rr
            @printf(io, "%d,%d,%d,%.6f,%.4f,%s,%.3f\n",
                    r.start, r.stop, r.npoints, r.omega_radph, r.kappa_cmps, r.sense, r.period_days)
        end
    end
end

ccw_rows = filter(r -> r.sense == "ccw", rows)
cw_rows = filter(r -> r.sense == "cw", rows)
write_ridge_csv(joinpath(OUTDIR, "jlab_ridges_joint_ccw.csv"), ccw_rows)
write_ridge_csv(joinpath(OUTDIR, "jlab_ridges_joint_cw.csv"), cw_rows)

println("\nJoint-mode jLab: CCW=$(length(ccw_rows)) ridges ($(sum(r->r.npoints, ccw_rows; init=0)) pts), " *
        "CW=$(length(cw_rows)) ridges ($(sum(r->r.npoints, cw_rows; init=0)) pts)")
println("Saved: $(joinpath(OUTDIR, "jlab_ridges_joint_ccw.csv")), $(joinpath(OUTDIR, "jlab_ridges_joint_cw.csv"))")

# ── Three-way comparison: ours vs jLab-univariate (wrong mode) vs jLab-joint (correct mode)
if isfile(OURS_CSV)
    ours = readdlm(OURS_CSV, ','; skipstart=1)
    our_ccw_n, our_ccw_pts = count(==("ccw"), ours[:, 8]), sum(ours[ours[:, 8].=="ccw", 3])
    our_cw_n, our_cw_pts = count(==("cw"), ours[:, 8]), sum(ours[ours[:, 8].=="cw", 3])
    println("\n=== Three-way ridge comparison ===")
    @printf("  %-22s %8s %10s %8s %10s\n", "", "CCW n", "CCW pts", "CW n", "CW pts")
    @printf("  %-22s %8d %10d %8d %10d\n", "ours", our_ccw_n, our_ccw_pts, our_cw_n, our_cw_pts)
    @printf("  %-22s %8d %10d %8d %10d\n", "jLab (joint, correct)", length(ccw_rows),
            sum(r -> r.npoints, ccw_rows; init=0), length(cw_rows), sum(r -> r.npoints, cw_rows; init=0))
    if isfile(OLD_CCW_CSV) && isfile(OLD_CW_CSV)
        old_ccw = readdlm(OLD_CCW_CSV, ','; skipstart=1)
        old_cw = readdlm(OLD_CW_CSV, ','; skipstart=1)
        @printf("  %-22s %8d %10d %8d %10d\n", "jLab (univariate, WRONG)",
                size(old_ccw, 1), sum(old_ccw[:, 3]), size(old_cw, 1), sum(old_cw[:, 3]))
    end
else
    println("\nNOTE: $OURS_CSV not found -- run flare_fig3_rotary_wavelet_ridge.jl first for the full comparison")
end
