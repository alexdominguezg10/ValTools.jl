"""
    plot_comparison_map(obs_data, model_data, lon, lat; kwargs...)

Three-panel map: observations | model | difference.

# Arguments
- `obs_data`, `model_data`: 2-D arrays `(ny, nx)`
- `lon`: 1-D or 2-D longitude array
- `lat`: 1-D or 2-D latitude array

# Keyword Arguments
- `title=""`, `units=""`, `obs_label="Observations"`, `model_label="Model"`
- `vmin=nothing`, `vmax=nothing`: color scale limits (auto if nothing)
- `diff_vmax=nothing`: color scale for difference panel
- `cmap=:RdBu_r`, `diff_cmap=:coolwarm`
- `figsize=(1200, 400)`

# Returns
`Figure`
"""
function ValTools.plot_comparison_map(obs_data::AbstractMatrix{<:Real},
                                      model_data::AbstractMatrix{<:Real},
                                      lon::AbstractArray{<:Real},
                                      lat::AbstractArray{<:Real};
                                      title::String="",
                                      units::String="",
                                      obs_label::String="Observations",
                                      model_label::String="Model",
                                      vmin::Union{Real, Nothing}=nothing,
                                      vmax::Union{Real, Nothing}=nothing,
                                      diff_vmax::Union{Real, Nothing}=nothing,
                                      cmap=Reverse(:RdBu),
                                      diff_cmap=Reverse(:RdBu),
                                      figsize::Tuple{Int,Int}=(1200, 400))
    obs = Float64.(obs_data)
    mod = Float64.(model_data)

    # Auto color limits
    all_finite = vcat(filter(isfinite, vec(obs)), filter(isfinite, vec(mod)))
    if vmin === nothing || vmax === nothing
        q = quantile(all_finite, [0.02, 0.98])
        if mean(all_finite) > 10
            vmin = vmin !== nothing ? Float64(vmin) : q[1]
            vmax = vmax !== nothing ? Float64(vmax) : q[2]
        else
            lim = max(abs(q[1]), abs(q[2]))
            vmin = vmin !== nothing ? Float64(vmin) : -lim
            vmax = vmax !== nothing ? Float64(vmax) : lim
        end
    else
        vmin, vmax = Float64(vmin), Float64(vmax)
    end

    # Difference
    has_diff = size(obs) == size(mod)
    diff = has_diff ? mod .- obs : nothing

    if diff_vmax === nothing && has_diff
        finite_diff = filter(isfinite, vec(abs.(diff)))
        diff_vmax = isempty(finite_diff) ? 1.0 : quantile(finite_diff, 0.95)
    end
    diff_vmax = diff_vmax !== nothing ? Float64(diff_vmax) : 1.0

    # Build lon/lat 2D grids if 1-D
    if ndims(lon) == 1 && ndims(lat) == 1
        lon2d = [lo for la in lat, lo in lon]
        lat2d = [la for la in lat, lo in lon]
    else
        lon2d = Float64.(lon)
        lat2d = Float64.(lat)
    end

    n_panels = has_diff ? 3 : 2
    fig = Figure(size=figsize)

    # Obs panel
    ax1 = Axis(fig[1, 1]; title=obs_label, xlabel="Longitude", ylabel="Latitude")
    hm1 = heatmap!(ax1, vec(lon2d[1,:]), vec(lat2d[:,1]), obs';
                    colorrange=(vmin, vmax), colormap=cmap)

    # Model panel
    ax2 = Axis(fig[1, 2]; title=model_label, xlabel="Longitude")
    heatmap!(ax2, vec(lon2d[1,:]), vec(lat2d[:,1]), mod';
             colorrange=(vmin, vmax), colormap=cmap)

    Colorbar(fig[1, n_panels + 1], hm1; label=units)

    # Difference panel
    if has_diff
        ax3 = Axis(fig[1, 3]; title="Model − Obs", xlabel="Longitude")
        hm3 = heatmap!(ax3, vec(lon2d[1,:]), vec(lat2d[:,1]), diff';
                        colorrange=(-diff_vmax, diff_vmax), colormap=diff_cmap)
        Colorbar(fig[1, n_panels + 2], hm3; label=units)
    end

    !isempty(title) && Label(fig[0, 1:n_panels], title; fontsize=14)

    return fig
end
