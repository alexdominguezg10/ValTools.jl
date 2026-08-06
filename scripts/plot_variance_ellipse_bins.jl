# Plot the variance ellipse for EVERY ADCP range bin of the WH600DW
# (bottom-boundary-layer) instrument at the ARE mooring site, all on one set
# of axes, color-coded by bin number. Companion to
# crosscheck_variance_ellipse.jl / plot_variance_ellipse.jl, which only used
# bin 1 of each instrument -- this looks at the full vertical structure a
# single instrument's multi-cell profile carries (Ekman-layer-like veering
# of the mean current and variance ellipse with distance from the bottom).
#
# Loads the real .mat file directly via ANCMooringLoader (needs `using MAT`
# for the weakdep extension) rather than reusing the crosscheck's cached
# bin-1-only CSV.
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/plot_variance_ellipse_bins.jl

using ValTools, MAT, CairoMakie, Statistics, LinearAlgebra

const ROOT    = normpath(joinpath(@__DIR__, ".."))
const OUTDIR  = joinpath(ROOT, "results", "crosscheck_variance_ellipse")
const DATADIR = "/Volumes/DATA_SSD/ADominguez/DATA/ANCLAJES/ANC_GoMW_deep"
const MATPATH = joinpath(DATADIR, "ARE-T2000-WH600DW-NS10732-Z1980-INS14-REC18.mat")
mkpath(OUTDIR)

"""
    ellipse_from_uv(u, v; n=200)

Fit a covariance-matrix variance ellipse to `u,v` directly (mean, covariance,
eigendecomposition), returning polygon coordinates `(ex, ey)` plus
`(a, b, u0, v0)` -- semi-major, semi-minor, and the record's mean current
(ellipse center).
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

function plot_ellipses_by_bin(; matpath=MATPATH,
                                outfile=joinpath(OUTDIR, "wh600dw_ellipses_by_bin.png"),
                                figsize=(1100, 900), dpi=200,
                                ellipse_linewidth=2.2, center_markersize=9,
                                path_linewidth=1.3, title_fontsize=20,
                                label_fontsize=16, tick_fontsize=13,
                                colormap=:viridis, limit_percentile=99.0,
                                limit_pad=1.15)
    r = ANCMooringLoader(matpath)
    p = anc_mooring_profiles(r)
    n_bins = size(p.u, 2)
    bins = p.bins

    ellipses = [ellipse_from_uv(p.u[isfinite.(p.u[:, k]) .& isfinite.(p.v[:, k]), k],
                                 p.v[isfinite.(p.u[:, k]) .& isfinite.(p.v[:, k]), k])
                for k in 1:n_bins]

    # Zoom to the ELLIPSES' own extent (center +/- semi-major), not the raw
    # scatter's percentile spread -- with 14 near-identical bins from one
    # short ADCP profile, the ellipses are much smaller than the full
    # current-speed distribution, so scaling to the latter would make every
    # bin-to-bin difference this plot exists to show invisible.
    lim = maximum(e -> max(abs(e.u0), abs(e.v0)) + e.a, ellipses) * limit_pad

    cmap = cgrad(colormap)
    bmin, bmax = extrema(bins)
    colorof(b) = cmap[(b - bmin) / max(bmax - bmin, eps())]

    fig = Figure(size=figsize, fontsize=label_fontsize)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), limits=(-lim, lim, -lim, lim),
               xlabel="u, eastward (m/s)", ylabel="v, northward (m/s)",
               title="WH600DW (bottom boundary layer, ~1980 m)\n" *
                     "variance ellipse per ADCP range bin ($(n_bins) bins)",
               titlesize=title_fontsize, xlabelsize=label_fontsize,
               ylabelsize=label_fontsize, xticklabelsize=tick_fontsize,
               yticklabelsize=tick_fontsize)

    # Thin path through each bin's mean-current point, in bin order, to show
    # the veering of the mean flow with distance from the bottom.
    lines!(ax, [e.u0 for e in ellipses], [e.v0 for e in ellipses];
           color=(:gray20, 0.5), linewidth=path_linewidth, linestyle=:dash)

    for (k, b) in enumerate(bins)
        c = colorof(b)
        e = ellipses[k]
        lines!(ax, e.ex, e.ey; color=c, linewidth=ellipse_linewidth)
        scatter!(ax, [e.u0], [e.v0]; color=c, markersize=center_markersize,
                 strokewidth=1, strokecolor=:black)
    end

    Colorbar(fig[1, 2]; limits=(bmin, bmax), colormap=cmap,
             label="ADCP bin (range cell)", labelsize=label_fontsize,
             ticklabelsize=tick_fontsize)

    save(outfile, fig; px_per_unit=dpi / 96)
    println("Saved: $outfile")
    println("Bins: ", bins, "  nominal depths (m): ", round.(p.depths; digits=0))
    return outfile
end

plot_ellipses_by_bin()
