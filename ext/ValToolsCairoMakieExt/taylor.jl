"""
    taylor_diagram(std_ref; samples=[], normalise=true, smin=0.0, smax=1.6)

Draw a Taylor diagram (Taylor 2001) using CairoMakie.

# Arguments
- `std_ref`: standard deviation of the reference (observations)
- `samples`: vector of `(std, corr, label)` tuples or NamedTuples with fields `std`, `corr`, `label`
- `normalise`: if true, normalize std-devs by `std_ref`
- `smin`, `smax`: radial range

# Returns
`Figure`
"""
function ValTools.taylor_diagram(std_ref::Real;
                                  samples::AbstractVector=NamedTuple[],
                                  normalise::Bool=true,
                                  smin::Real=0.0,
                                  smax::Real=1.6,
                                  title::String="",
                                  figsize::Tuple{Int,Int}=(600, 550))
    fig = Figure(size=figsize)
    ax = PolarAxis(fig[1, 1];
                   thetalimits=(0, π/2),
                   rlimits=(smin, smax),
                   rgridcolor=:grey80,
                   thetagridcolor=:grey80)

    corr_ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    ax.thetaticks = (acos.(corr_ticks), string.(corr_ticks))

    ref_r = normalise ? 1.0 : Float64(std_ref)

    # Reference point
    scatter!(ax, [0.0], [ref_r]; marker=:star5, markersize=18, color=:black)

    # CRMSE contour arcs
    ts = range(0, π/2; length=300)
    for rmse_val in range(0.1, 1.5; length=8)
        cos_t = cos.(ts)
        disc = ref_r^2 .* cos_t .^ 2 .- ref_r^2 .+ rmse_val^2
        valid = disc .>= 0
        any(valid) || continue
        r_arc = ref_r .* cos_t[valid] .+ sqrt.(disc[valid])
        r_arc = clamp.(r_arc, smin, smax)
        lines!(ax, collect(ts[valid]), collect(r_arc);
               color=(:silver, 0.6), linewidth=0.8, linestyle=:dash)
    end

    # Plot samples
    colors = Makie.wong_colors()
    for (i, s) in enumerate(samples)
        std_val = s isa NamedTuple ? s.std : s[1]
        corr_val = s isa NamedTuple ? s.corr : s[2]
        lbl = s isa NamedTuple ? s.label : (length(s) >= 3 ? s[3] : "")

        theta = acos(clamp(corr_val, -1.0, 1.0))
        r = normalise ? std_val / Float64(std_ref) : Float64(std_val)
        c = colors[mod1(i, length(colors))]
        scatter!(ax, [theta], [r]; markersize=12, color=c, label=lbl)
    end

    if !isempty(samples)
        entries = [MarkerElement(; marker=:circle, color=colors[mod1(i, length(colors))],
                                  markersize=12) => s isa NamedTuple ? s.label : string(s[3])
                   for (i, s) in enumerate(samples)]
        elements = [e[1] for e in entries]
        labels = [e[2] for e in entries]
        Legend(fig[1, 2], elements, labels)
    end

    !isempty(title) && Label(fig[0, 1], title; fontsize=14)

    return fig
end
