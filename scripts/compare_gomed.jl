# Compare ValTools' full-census significance-test output against the real
# published GOMED dataset (Lilly & Perez-Brunius 2021, NPG).
#
# Run scripts/validate_gulfdrifters_significant.jl (full census, e.g. on
# Ixachi via run_gulfdrifters_significant.slurm) FIRST -- this script only
# reads its output CSV, it does no wavelet computation itself.
#
# SCOPE (see project_lilly_perezbrunius_2021_gulfdrifters_paper /
# valtools_stage4_progress memory for the full background):
#   - DISTRIBUTIONAL comparison only, not exact per-ridge 1:1 matching.
#     Our ridge-tracking and GOMED's density-ratio method segment events
#     differently in time even on the same drifter, so lining up individual
#     ridges by timestamp isn't meaningful -- we compare population-level
#     statistics (counts, L distribution, omega_ast distribution, sign
#     balance) per drifter instead.
#   - GOMED's `traj` dimension (14471) is the FULL detected ridge population
#     including inertial oscillations, not just the paper's 1033
#     "significant" ridges. Reproducing the paper's own definition (Sect.
#     4.7) exactly: significant := rho[:, 5] < 0.1 (X = L*xi_bar^4, their
#     column 5 -- NOT column 1/L-alone, which the paper itself calls out as
#     a weaker criterion that over-accepts low-frequency events; a first
#     draft of this script used column 1 and got 83.8% "significant",
#     confirming the paper's own critique) AND omega_ast_bar > -1/2 (the
#     anticyclonic inertial/centrifugal stability boundary, Sect. 4.1,
#     applied by the paper to reject inertial oscillations before counting
#     "significant"). The same omega_ast > -1/2 cutoff is applied to our own
#     output below too, for a fair comparison -- our significance test has
#     no inertial-oscillation exclusion of its own otherwise.
#   - xi_bar/R_bar/V_bar (GOMED's ellipse circularity/radius/KE-velocity)
#     have NO counterpart on our side -- eddy_census's ridge-chaining tracks
#     one rotary branch's amplitude/frequency only, not both CW/CCW branches
#     needed to reconstruct ellipse semi-axes. GOMED's V_bar is used below
#     only as the weight in GOMED's OWN energy-weighted statistic, not
#     compared against anything of ours.
#   - Both our and GOMED's rotation sense come from single-branch heuristics
#     with known limitations (see validate_gulfdrifters_significant.jl's
#     SENSE NOTE) -- treat sign-balance agreement as suggestive, not proof.

using NCDatasets, DataFrames, Statistics, Printf, CairoMakie

const GOMED_PATH = joinpath(homedir(), ".valtools", "jdata", "gomed_1.1.0.nc")
const OUR_CSV = joinpath(@__DIR__, "..", "results", "gulfdrifters_significant_full.csv")
const RESULTS_DIR = joinpath(@__DIR__, "..", "results")
const RHO_THRESHOLD = 0.1     # matches the paper's own accepted-event cutoff (Sect. 4.6)
const RHO_COL = 5             # X = L*xi_bar^4 (paper's chosen criterion, Sect. 4.6), NOT column 1 (L alone)
const OMEGA_AST_MIN = -0.5    # exclude inertial oscillations (Sect. 4.1 stability boundary)

isfile(GOMED_PATH) || error("GOMED file not found at $GOMED_PATH")
isfile(OUR_CSV) || error("Our census output not found at $OUR_CSV -- " *
                         "run scripts/validate_gulfdrifters_significant.jl " *
                         "(full census, e.g. via run_gulfdrifters_significant.slurm on Ixachi) first")

# --- Load GOMED ---------------------------------------------------------
println("Loading GOMED ($GOMED_PATH)...")
ds = NCDatasets.Dataset(GOMED_PATH, "r")
gomed = DataFrame(
    ridge_id      = Int.(ds["ridge_id"][:]),
    segment_id    = Int.(ds["segment_id"][:]),
    drifter_id    = Int.(ds["drifter_id"][:]),
    L             = Float64.(ds["L"][:]),
    omega_ast_bar = Float64.(ds["omega_ast_bar"][:]),
    xi_bar        = Float64.(ds["xi_bar"][:]),
    R_bar         = Float64.(ds["R_bar"][:]),
    V_bar         = Float64.(ds["V_bar"][:]),
    rho_Lxi4      = Float64.(ds["rho"][:, RHO_COL]),  # NCDatasets reverses NetCDF's declared
                                                       # (rho_cols, traj) order to (traj, rho_cols)
                                                       # for Julia's column-major arrays
)
close(ds)
println("  GOMED: $(nrow(gomed)) total ridges (full detected population, includes inertial)")

gomed_sig = gomed[(gomed.rho_Lxi4 .< RHO_THRESHOLD) .& (gomed.omega_ast_bar .> OMEGA_AST_MIN), :]
@printf("  GOMED significant (rho[L*xi^4] < %.2f, omega_ast_bar > %.1f): %d ridges (%.1f%%)\n",
        RHO_THRESHOLD, OMEGA_AST_MIN, nrow(gomed_sig), 100 * nrow(gomed_sig) / nrow(gomed))

# --- Load our full-census output ----------------------------------------
# Manual CSV parse (no CSV.jl dependency -- not in Project.toml, and adding
# one wasn't part of the approved plan for this task).
#
# SCHEMA (rewritten 2026-08-01, "ValTools 7", when validate_gulfdrifters_significant.jl
# switched from the old ad hoc eddy_census_significant/AR1 test to the
# paper-faithful rotary_ridge_properties+density_ratio_significance path --
# see that script's own header for why): columns are now
# drifter_id,segment_id,L,omega_ast_bar,sense,npoints,duration_hours,xi_bar,kappa_bar,rho_x
# (was drifter_id,segment_id,L_est,omega_ast_est,sense,duration_hours,
# mean_frequency_rad_per_hour,mean_amplitude). `kappa_bar` (ridge-averaged
# joint rotary amplitude, kappa=sqrt(|wx|^2+|wy|^2)) is the natural successor
# to the old `mean_amplitude` as the energy-proxy weight below -- both are
# ridge-amplitude magnitudes used only for the energy-weighted sign-balance
# comparison, never compared in absolute units against GOMED's own V_bar.
function _read_our_csv(path)
    lines = readlines(path)
    rows = NamedTuple[]
    for line in @view lines[2:end]
        isempty(line) && continue
        f = split(line, ",")
        push!(rows, (drifter_id=parse(Int, f[1]), segment_id=parse(Int, f[2]),
                      L_est=parse(Float64, f[3]), omega_ast_est=parse(Float64, f[4]),
                      sense=f[5], npoints=parse(Int, f[6]), duration_hours=parse(Float64, f[7]),
                      xi_bar=parse(Float64, f[8]), mean_amplitude=parse(Float64, f[9]),
                      rho_x=parse(Float64, f[10])))
    end
    return DataFrame(rows)
end

println("\nLoading our full-census output ($OUR_CSV)...")
ours_raw = _read_our_csv(OUR_CSV)
println("  Ours (before inertial exclusion): $(nrow(ours_raw)) significant events, " *
        "$(length(unique(ours_raw.drifter_id))) drifters")

ours = ours_raw[ours_raw.omega_ast_est .> OMEGA_AST_MIN, :]
@printf("  Ours (omega_ast_est > %.1f, non-inertial): %d events, %d drifters\n",
        OMEGA_AST_MIN, nrow(ours), length(unique(ours.drifter_id)))

# --- Join coverage -------------------------------------------------------
gomed_drifters = Set(gomed_sig.drifter_id)
our_drifters = Set(ours.drifter_id)
overlap = intersect(gomed_drifters, our_drifters)
println("\n--- Join coverage ---")
println("  Our unique drifters:              $(length(our_drifters))")
println("  GOMED-significant unique drifters: $(length(gomed_drifters))")
@printf("  Overlap:                           %d (%.1f%% of ours)\n",
        length(overlap), 100 * length(overlap) / length(our_drifters))
println("  (Full overlap not expected -- GulfDriftersOpen is a licensed subset")
println("   of the full GulfDriftersAll both this run and GOMED derive from.)")

ours_ov = ours[in.(ours.drifter_id, Ref(overlap)), :]
gomed_ov = gomed_sig[in.(gomed_sig.drifter_id, Ref(overlap)), :]

# --- Per-drifter event counts ---------------------------------------------
our_counts = combine(groupby(ours_ov, :drifter_id), nrow => :n_ours)
gomed_counts = combine(groupby(gomed_ov, :drifter_id), nrow => :n_gomed)
counts_joined = innerjoin(our_counts, gomed_counts, on=:drifter_id)
count_diff = counts_joined.n_ours .- counts_joined.n_gomed

println("\n--- Per-drifter event counts (n=$(nrow(counts_joined)) overlapping drifters) ---")
@printf("  Ours:  mean=%.2f  median=%.1f\n", mean(counts_joined.n_ours), median(counts_joined.n_ours))
@printf("  GOMED: mean=%.2f  median=%.1f\n", mean(counts_joined.n_gomed), median(counts_joined.n_gomed))
@printf("  Diff (ours - GOMED): mean=%.2f  std=%.2f\n", mean(count_diff), std(count_diff))

# --- L (ridge length) distribution ----------------------------------------
println("\n--- L (ridge length, cycles) ---")
@printf("  Ours:  mean=%.2f  median=%.2f  (n=%d)\n", mean(ours_ov.L_est), median(ours_ov.L_est), nrow(ours_ov))
@printf("  GOMED: mean=%.2f  median=%.2f  (n=%d)\n", mean(gomed_ov.L), median(gomed_ov.L), nrow(gomed_ov))

# --- Cyclonic/anticyclonic sign balance ------------------------------------
n_cyc_ours, n_acyc_ours = count(>(0), ours_ov.omega_ast_est), count(<(0), ours_ov.omega_ast_est)
n_cyc_gomed, n_acyc_gomed = count(>(0), gomed_ov.omega_ast_bar), count(<(0), gomed_ov.omega_ast_bar)

println("\n--- Cyclonic vs. anticyclonic balance (sign of omega_ast; + cyclonic, - anticyclonic, NH) ---")
@printf("  Ours:  cyclonic=%d (%.1f%%)  anticyclonic=%d (%.1f%%)\n",
        n_cyc_ours, 100 * n_cyc_ours / nrow(ours_ov), n_acyc_ours, 100 * n_acyc_ours / nrow(ours_ov))
@printf("  GOMED: cyclonic=%d (%.1f%%)  anticyclonic=%d (%.1f%%)\n",
        n_cyc_gomed, 100 * n_cyc_gomed / nrow(gomed_ov), n_acyc_gomed, 100 * n_acyc_gomed / nrow(gomed_ov))

count_weighted_ours = mean(sign.(ours_ov.omega_ast_est))
count_weighted_gomed = mean(sign.(gomed_ov.omega_ast_bar))
energy_weighted_ours = sum(sign.(ours_ov.omega_ast_est) .* ours_ov.mean_amplitude) / sum(ours_ov.mean_amplitude)
energy_weighted_gomed = sum(sign.(gomed_ov.omega_ast_bar) .* gomed_ov.V_bar .^ 2) / sum(gomed_ov.V_bar .^ 2)

println("\n--- Rotary tendency (mean sign; + = net cyclonic, - = net anticyclonic) ---")
@printf("  Ours  count-weighted:  %+.3f   energy-weighted (amplitude proxy): %+.3f\n",
        count_weighted_ours, energy_weighted_ours)
@printf("  GOMED count-weighted:  %+.3f   energy-weighted (V_bar^2, real KE): %+.3f\n",
        count_weighted_gomed, energy_weighted_gomed)
println("  (validate_gulfdrifters.jl's earlier energy-weighted result on the FULL")
println("   eddy_census heuristic population, not just significant events: -0.228)")

# --- Plots -----------------------------------------------------------------
fig = Figure(size=(1100, 800))

ax1 = Axis(fig[1, 1], xlabel="L (cycles)", ylabel="density", title="Ridge length L")
hist!(ax1, ours_ov.L_est, normalization=:pdf, color=(:dodgerblue, 0.5), label="Ours (n=$(nrow(ours_ov)))")
hist!(ax1, gomed_ov.L, normalization=:pdf, color=(:orangered, 0.5), label="GOMED sig. (n=$(nrow(gomed_ov)))")
axislegend(ax1)

ax2 = Axis(fig[1, 2], xlabel="omega* (nondimensional frequency)", ylabel="density",
           title="Nondimensional frequency (sign = rotation sense)")
hist!(ax2, ours_ov.omega_ast_est, normalization=:pdf, color=(:dodgerblue, 0.5), label="Ours")
hist!(ax2, gomed_ov.omega_ast_bar, normalization=:pdf, color=(:orangered, 0.5), label="GOMED sig.")
axislegend(ax2)

ax3 = Axis(fig[2, 1], xlabel="GOMED events/drifter", ylabel="Our events/drifter",
           title="Per-drifter event count agreement (n=$(nrow(counts_joined)))")
scatter!(ax3, counts_joined.n_gomed, counts_joined.n_ours, color=(:seagreen, 0.6), markersize=6)
maxc = max(maximum(counts_joined.n_gomed), maximum(counts_joined.n_ours))
lines!(ax3, [0, maxc], [0, maxc], color=:gray40, linestyle=:dash, label="1:1")
axislegend(ax3, position=:lt)

ax4 = Axis(fig[2, 2], title="Rotary tendency summary")
hidedecorations!(ax4)
hidespines!(ax4)
xlims!(ax4, 0, 1)
ylims!(ax4, 0, 1)
text!(ax4, 0.02, 0.85,
      text="Count-weighted mean sign:\n  Ours:  $(round(count_weighted_ours, digits=3))\n  GOMED: $(round(count_weighted_gomed, digits=3))",
      fontsize=14, align=(:left, :top))
text!(ax4, 0.02, 0.4,
      text="Energy-weighted mean sign:\n  Ours:  $(round(energy_weighted_ours, digits=3))\n  GOMED: $(round(energy_weighted_gomed, digits=3))",
      fontsize=14, align=(:left, :top))

mkpath(RESULTS_DIR)
save(joinpath(RESULTS_DIR, "gomed_comparison.png"), fig)
println("\nSaved plot: results/gomed_comparison.png")
println("\nNOTE: xi_bar/R_bar/V_bar have no counterpart in our output -- see file")
println("      header. This is a distributional/aggregate comparison over")
println("      $(length(overlap)) overlapping drifters, not an exact per-ridge match.")
