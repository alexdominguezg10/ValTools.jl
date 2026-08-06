# Plot the variance ellipses cross-checked in crosscheck_variance_ellipse.jl,
# on the same real ARE-site ANC_GoMW_deep mooring data (see
# results/crosscheck_variance_ellipse/*_uv.csv, produced by that script).
#
# Both instruments overlaid on ONE set of axes at the same scale (rather than
# side-by-side panels), so the depth-to-depth comparison -- ellipse size,
# eccentricity, orientation -- is direct. Each site's ellipse is drawn
# directly from its own covariance matrix's eigenvectors/eigenvalues, not
# reconstructed from the "inclination from north" bearing angle (avoids any
# angle-convention round-trip bug), with the mean-current vector (MKE, a
# distinct quantity from the ellipse's EKE) shown alongside it. Equal-aspect
# axes: u and v share units (m/s), so an unequal aspect would visually
# distort the ellipses' true eccentricity/orientation.
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/plot_variance_ellipse.jl

using ValTools, CairoMakie, DelimitedFiles, Statistics, LinearAlgebra

const ROOT   = normpath(joinpath(@__DIR__, ".."))
const INDIR  = joinpath(ROOT, "results", "crosscheck_variance_ellipse")
const OUTDIR = INDIR
mkpath(OUTDIR)

const SITES = [
    (name="LR75DW_mid",     label="LR75DW (mid-depth, ~715 m)",              color=:crimson),
    (name="WH600DW_bottom", label="WH600DW (bottom boundary layer, ~1980 m)", color=:royalblue),
]

"""
    ellipse_from_uv(u, v; n=200)

Fit a covariance-matrix variance ellipse to `u,v` directly (mean, covariance,
eigendecomposition), returning polygon coordinates `(ex, ey)` for plotting
plus `(a, b, u0, v0)` -- semi-major, semi-minor, and the record's mean
current (ellipse center / MKE vector tip).
"""
function ellipse_from_uv(u::AbstractVector, v::AbstractVector; n::Int=200)
    u0, v0 = mean(u), mean(v)
    ua, va = u .- u0, v .- v0
    C = [mean(ua .^ 2)        mean(ua .* va);
         mean(ua .* va)       mean(va .^ 2)]
    F = eigen(Symmetric(C))            # ascending eigenvalues
    a = sqrt(max(F.values[2], 0.0))    # major
    b = sqrt(max(F.values[1], 0.0))    # minor
    e_major = F.vectors[:, 2]
    e_minor = F.vectors[:, 1]
    t = range(0, 2π; length=n)
    ex = u0 .+ a .* cos.(t) .* e_major[1] .+ b .* sin.(t) .* e_minor[1]
    ey = v0 .+ a .* cos.(t) .* e_major[2] .+ b .* sin.(t) .* e_minor[2]
    return (ex=ex, ey=ey, a=a, b=b, u0=u0, v0=v0)
end

function plot_variance_ellipses(; sites=SITES, indir=INDIR,
                                  outfile=joinpath(OUTDIR, "variance_ellipses.png"),
                                  figsize=(1000, 900), dpi=200,
                                  scatter_markersize=2.0, scatter_alpha=0.10,
                                  ellipse_linewidth=3, title_fontsize=20,
                                  label_fontsize=16, tick_fontsize=13,
                                  legend_fontsize=13,
                                  limit_percentile=99.5, limit_pad=1.15)
    records = [(uv = readdlm(joinpath(indir, "$(site.name)_uv.csv"), ','), site...)
               for site in sites]

    # Shared, symmetric axis limits (percentile-based, not raw min/max, per
    # the workspace's plotting standard) -- one axis for both instruments,
    # so this single number IS the shared scale.
    lim = maximum(r -> quantile(vec(hypot.(r.uv[:, 1], r.uv[:, 2])), limit_percentile / 100),
                  records) * limit_pad

    fig = Figure(size=figsize, fontsize=label_fontsize)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=(-lim, lim, -lim, lim),
               xlabel="u, eastward (m/s)", ylabel="v, northward (m/s)",
               title="ARE mooring: current variance ellipses by depth",
               titlesize=title_fontsize, xlabelsize=label_fontsize,
               ylabelsize=label_fontsize, xticklabelsize=tick_fontsize,
               yticklabelsize=tick_fontsize)

    subtitle_parts = String[]
    for r in records
        u, v = r.uv[:, 1], r.uv[:, 2]
        cem = current_ellipse_metrics(u, v, u, v)
        ell = ellipse_from_uv(u, v)

        scatter!(ax, u, v; markersize=scatter_markersize,
                 color=(r.color, scatter_alpha))
        lines!(ax, ell.ex, ell.ey; color=r.color, linewidth=ellipse_linewidth,
               label=r.label)
        arrows2d!(ax, [0.0], [0.0], [ell.u0], [ell.v0]; color=r.color, shaftwidth=2.5)
        scatter!(ax, [ell.u0], [ell.v0]; color=r.color, markersize=8, marker=:xcross,
                 strokewidth=1.5, strokecolor=:black)

        push!(subtitle_parts,
              "$(r.name): a=$(cem["obs_semi_major"]) b=$(cem["obs_semi_minor"]) m/s, " *
              "incl=$(cem["obs_inclination"])°, EKE=$(round(cem["obs_EKE"]; digits=4)) m²/s²")
    end

    Label(fig[2, 1], join(subtitle_parts, "\n"); fontsize=tick_fontsize, tellwidth=false)
    axislegend(ax; position=:rb, framevisible=false, labelsize=legend_fontsize)

    save(outfile, fig; px_per_unit=dpi / 96)
    println("Saved: $outfile")
    return outfile
end

plot_variance_ellipses()
