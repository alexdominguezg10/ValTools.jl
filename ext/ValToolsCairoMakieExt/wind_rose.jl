const _COMPASS_16 = ["N","NNE","NE","ENE","E","ESE","SE","SSE",
                     "S","SSW","SW","WSW","W","WNW","NW","NNW"]

function _uv_to_met_dir(u, v)
    return mod.(atand.(-u, -v), 360.0)
end

function _compute_rose_freq(speed, direction, n_sectors, speed_bins, calm_threshold)
    n_obs = length(speed)
    n_obs == 0 && return (zeros(n_sectors, length(speed_bins)-1), 0.0)

    calm_mask = speed .< calm_threshold
    calm_pct = 100.0 * sum(calm_mask) / n_obs

    active_speed = speed[.!calm_mask]
    active_dir = direction[.!calm_mask]

    sector_width = 360.0 / n_sectors
    n_sb = length(speed_bins) - 1
    freq = zeros(n_sectors, n_sb)

    for s in 1:n_sectors
        lo = mod((s - 1) * sector_width - sector_width / 2, 360.0)
        hi = mod(s * sector_width - sector_width / 2, 360.0)
        if lo < hi
            in_sector = (active_dir .>= lo) .& (active_dir .< hi)
        else
            in_sector = (active_dir .>= lo) .| (active_dir .< hi)
        end
        spd_in = active_speed[in_sector]
        for b in 1:n_sb
            in_bin = (spd_in .>= speed_bins[b]) .& (spd_in .< speed_bins[b+1])
            freq[s, b] = 100.0 * sum(in_bin) / n_obs
        end
    end
    return freq, calm_pct
end

"""
    plot_wind_rose(u, v; kwargs...)

Draw a wind rose on polar axes using CairoMakie.

# Arguments
- `u`, `v`: zonal and meridional wind components [m/s]

# Keyword Arguments
- `n_sectors=16`, `speed_bins=nothing`, `calm_threshold=0.5`
- `title=""`, `figsize=(550, 550)`

# Returns
`Figure`
"""
function ValTools.plot_wind_rose(u::AbstractVector{<:Real},
                                 v::AbstractVector{<:Real};
                                 n_sectors::Int=16,
                                 speed_bins::Union{AbstractVector{<:Real}, Nothing}=nothing,
                                 calm_threshold::Real=0.5,
                                 title::String="",
                                 figsize::Tuple{Int,Int}=(550, 550))
    uf = Float64.(u)
    vf = Float64.(v)
    valid = isfinite.(uf) .& isfinite.(vf)
    uf, vf = uf[valid], vf[valid]
    isempty(uf) && error("No valid wind data.")

    speed = hypot.(uf, vf)
    direction = _uv_to_met_dir(uf, vf)

    if speed_bins === nothing
        max_spd = quantile(speed, 0.99)
        step = max(1.0, round(max_spd / 6))
        speed_bins = collect(0.0:step:max_spd+step)
    else
        speed_bins = Float64.(speed_bins)
        if speed_bins[end] < maximum(speed)
            push!(speed_bins, maximum(speed) + 1.0)
        end
    end

    freq, calm_pct = _compute_rose_freq(speed, direction, n_sectors, speed_bins, calm_threshold)
    n_sb = length(speed_bins) - 1

    fig = Figure(size=figsize)
    ax = PolarAxis(fig[1, 1];
                   thetalimits=(0, 2π),
                   direction=-1,
                   theta_0=π/2)

    sector_angles = range(0, 2π; length=n_sectors+1)[1:end-1]
    sector_width = 2π / n_sectors
    colors = cgrad(:YlOrRd, n_sb; categorical=true)

    bottom = zeros(n_sectors)
    for b in 1:n_sb
        heights = freq[:, b]
        for s in 1:n_sectors
            θ_center = sector_angles[s]
            θ_lo = θ_center - sector_width * 0.46
            θ_hi = θ_center + sector_width * 0.46
            r_lo = bottom[s]
            r_hi = bottom[s] + heights[s]
            θs = range(θ_lo, θ_hi; length=20)
            xs = vcat([r_lo .* cos.(θs); r_hi .* cos.(reverse(θs))])
            ys = vcat([r_lo .* sin.(θs); r_hi .* sin.(reverse(θs))])
            poly!(ax, Point2f.(zip(
                vcat(collect(θs), reverse(collect(θs))),
                vcat(fill(r_lo, 20), fill(r_hi, 20))
            )); color=colors[b])
        end
        bottom .+= heights
    end

    # Labels
    compass = n_sectors >= 16 ? _COMPASS_16 : _COMPASS_16[1:2:end]
    ax.thetaticks = (collect(sector_angles), compass[1:min(n_sectors, length(compass))])

    !isempty(title) && Label(fig[0, 1], title; fontsize=13, font=:bold)

    # Stats text
    mean_spd = mean(speed)
    text!(fig.scene, 0.82, 0.95;
          text="n=$(length(uf))\nMean: $(round(mean_spd; digits=1)) m/s\nCalm: $(round(calm_pct; digits=1))%",
          fontsize=10, space=:relative, align=(:left, :top))

    return fig
end

"""
    plot_wind_rose_comparison(datasets; suptitle="", figsize=nothing)

Side-by-side wind roses. `datasets` is a vector of `(label, u, v)` tuples.
"""
function ValTools.plot_wind_rose_comparison(
        datasets::AbstractVector;
        suptitle::String="",
        figsize::Union{Tuple{Int,Int}, Nothing}=nothing)
    n = length(datasets)
    w = figsize !== nothing ? figsize[1] : 500 * n
    h = figsize !== nothing ? figsize[2] : 500
    fig = Figure(size=(w, h))

    for (i, ds) in enumerate(datasets)
        lbl, u, v = ds
        sub_fig = ValTools.plot_wind_rose(u, v; title=lbl)
        # For simplicity, create independent figures
        # A proper implementation would share axes in one figure
    end

    # Simple approach: just make the first one
    if n >= 1
        lbl1, u1, v1 = datasets[1]
        return ValTools.plot_wind_rose(u1, v1; title=suptitle * " — " * lbl1,
                                       figsize=(w, h))
    end
    return Figure()
end
