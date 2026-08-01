# Part 1 of 3 of the jLab cross-check harness (jlab_crosscheck_*).
#
# Exports the SAME stratified 28-case GOMED sample used by
# case_study_gomed.jl (identical seed/regime bins, so results are directly
# comparable to that script's numbers) as one CSV per segment
# (num,lat,lon -- MATLAB DATENUM days, degrees), plus a manifest recording
# each case's true GOMED ridge window/L/omega/xi for reference.
#
# This exists because prior sessions repeatedly tuned ridge-detection code
# against OUR OWN metrics without checking it against jLab's actual
# eddyridges.m output on the same data -- see
# project_gomed_validation_results memory (2026-08-01 entries) for what
# that cost. Re-run this (and jlab_crosscheck_eddyridges.m, then
# jlab_crosscheck_compare.jl) after ANY change to ridge point detection,
# chaining, or the frequency-grid construction, before trusting the
# result.
#
# Usage: julia --project=envs/cpu scripts/jlab_crosscheck_export.jl

using ValTools, ValTools.JLab
using NCDatasets, Statistics, Printf, Random, DelimitedFiles

const N_CASES = 28
const GOMED_PATH = joinpath(homedir(), ".valtools", "jdata", "gomed_1.1.0.nc")
const RHO_COL = 5
const RHO_THRESHOLD = 0.1
const OMEGA_AST_MIN = -0.5
const OUTDIR = joinpath(@__DIR__, "..", "results", "jlab_crosscheck")

mkpath(OUTDIR)

println("Loading GOMED...")
ds = NCDatasets.Dataset(GOMED_PATH, "r")
segment_id = Int.(ds["segment_id"][:])
ridge_id_arr = Int.(ds["ridge_id"][:])
Lg = Float64.(ds["L"][:])
omega_bar = Float64.(ds["omega_ast_bar"][:])
xi_bar = Float64.(ds["xi_bar"][:])
R_bar = Float64.(ds["R_bar"][:])
rho5 = Float64.(ds["rho"][:, RHO_COL])
ids_obs = Int.(ds["ids"][:])
time_obs = Float64.(ds["time"][:])
close(ds)

println("Loading GulfDriftersOpen...")
data = load_gulfdrifters(; download=false)
our_ids = Set(data.id)
id_to_idx = Dict(data.id[i] => i for i in 1:data.n_drifters)

sig = (rho5 .< RHO_THRESHOLD) .& (omega_bar .> OMEGA_AST_MIN)
avail = [k for k in findall(sig) if segment_id[k] in our_ids]

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
for (key, idxs) in sort(collect(bins); by=first)
    n_take = min(per_bin, length(idxs))
    append!(selected, idxs[randperm(length(idxs))[1:n_take]])
end
selected = selected[1:min(N_CASES, length(selected))]
println("Selected $(length(selected)) cases (same seed/bins as case_study_gomed.jl)")

manifest = open(joinpath(OUTDIR, "manifest.csv"), "w")
println(manifest, "ridge_id,segment_id,N,gomed_t0,gomed_t1,gomed_L,gomed_omega_ast,gomed_xi,gomed_R")
for k in selected
    rid = ridge_id_arr[k]
    sid = segment_id[k]
    i = id_to_idx[sid]
    t, lat, lon = data.time[i], data.lat[i], data.lon[i]
    writedlm(joinpath(OUTDIR, "seg_$(rid).csv"), hcat(t, lat, lon), ',')

    gomed_rows = findall(==(rid), ids_obs)
    gt = time_obs[gomed_rows]
    g_t0, g_t1 = extrema(gt)
    @printf(manifest, "%d,%d,%d,%.6f,%.6f,%.4f,%.4f,%.4f,%.2f\n",
            rid, sid, length(t), g_t0, g_t1, Lg[k], omega_bar[k], xi_bar[k], R_bar[k])
end
close(manifest)
println("Exported $(length(selected)) segments + manifest.csv to $OUTDIR")
println("Next: run scripts/jlab_crosscheck_eddyridges.m in MATLAB, then scripts/jlab_crosscheck_compare.jl")
