function _op_quiver_stride(n, target)
    return max(1, div(n, target))
end

# Arrow length in DATA units needs to scale with both grid spacing (so
# neighboring arrows don't overlap) and typical speed (so it's not a fixed
# multiplier that's invisible for slow flow / a solid smear for fast flow).
# scale_mult=1.0 -> arrows span ~70% of the post-stride spacing at the
# median speed; matches matplotlib's auto-scaled quiver behavior.
function _op_quiver_lengthscale(lon_s, lat_s, u, v, scale_mult)
    (length(lon_s) < 2 || length(lat_s) < 2) && return scale_mult
    dx = abs(mean(diff(lon_s)))
    dy = abs(mean(diff(lat_s)))
    speed = filter(isfinite, vec(sqrt.(u .^ 2 .+ v .^ 2)))
    typical_speed = isempty(speed) ? 1.0 : median(speed)
    return scale_mult * 0.7 * min(dx, dy) / max(typical_speed, 1e-6)
end

function _op_color_range(field, symmetric, plow, phigh)
    finite = filter(isfinite, vec(field))
    isempty(finite) && return (-1.0, 1.0)
    if symmetric
        vmax = quantile(abs.(finite), phigh / 100)
        return (-vmax, vmax)
    else
        return (quantile(finite, plow / 100), quantile(finite, phigh / 100))
    end
end

function _op_draw_panel!(fig, pos, lon, lat, field, spec, title;
                          u=nothing, v=nothing, sx=1, sy=1,
                          quiver_color=:black, quiver_scale=1.0,
                          quiver_shaftwidth=1.5, quiver_tipwidth=4.0, quiver_tiplength=6.0,
                          color_percentile_low=2.0, color_percentile_high=98.0,
                          land_color=:gray55, font_size_title=16, font_size_labels=13,
                          font_size_ticks=11)
    ax = Axis(fig[pos...]; title=title, xlabel="Longitude", ylabel="Latitude",
              titlesize=font_size_title, xlabelsize=font_size_labels,
              ylabelsize=font_size_labels, xticklabelsize=font_size_ticks,
              yticklabelsize=font_size_ticks, aspect=DataAspect())
    vmin, vmax = _op_color_range(field, spec.symmetric, color_percentile_low, color_percentile_high)
    hm = heatmap!(ax, lon, lat, field; colormap=spec.cmap, colorrange=(vmin, vmax),
                  nan_color=land_color)
    if u !== nothing
        lon_s = lon[1:sx:end]; lat_s = lat[1:sy:end]
        u_s = u[1:sx:end, 1:sy:end]; v_s = v[1:sx:end, 1:sy:end]
        lengthscale = _op_quiver_lengthscale(lon_s, lat_s, u_s, v_s, quiver_scale)
        arrows2d!(ax, lon_s, lat_s, u_s, v_s; color=quiver_color,
                  lengthscale=lengthscale, shaftwidth=quiver_shaftwidth,
                  tiplength=quiver_tiplength, tipwidth=quiver_tipwidth)
    end
    Colorbar(fig[pos[1], pos[2]+1], hm; label=spec.label, labelsize=font_size_labels,
             ticklabelsize=font_size_ticks)
    return ax
end

"""
    plot_field_panel(lon, lat, fields, specs; kwargs...) -> Figure

Multi-panel heatmap grid (default 3 columns) for 2-D gridded lon/lat scalar
fields, with an optional velocity quiver overlay on one panel. General
successor to ad hoc per-project panel plotting code -- usable for any
gridded field (vorticity, divergence, SST, ...) across projects (GOFLOW,
TAMOC, Multiphase_Plume, ...), not tied to any one field naming convention.

# Arguments
- `lon`, `lat`: 1-D coordinate vectors
- `fields`: ordered `Vector` of `name => data` pairs, `data` as `(nlon, nlat)` matrices
- `specs`: `Dict{String,NamedTuple}`, `name -> (cmap=..., label=..., symmetric=::Bool)`.
  `symmetric=true` centers the color scale on zero (diverging fields like
  vorticity); `false` uses the raw percentile range (fields like SST/BT).

# Keyword Arguments
- `u=nothing`, `v=nothing`: optional velocity components `(nlon, nlat)` for quiver overlay
- `quiver_field=nothing`: which field name (must be a key in `fields`) gets the quiver overlay
- `title=""`
- `fig_width=6.5`, `fig_height=5.2`, `dpi=150`: per-panel size in inches, converted to pixels
- `quiver_density_x=35`, `quiver_density_y=35`: target arrows per axis
- `quiver_scale=1.0`: arrow length multiplier (auto-scaled internally by grid
  spacing and median speed -- see balanced-density guidance; 1.0 is a good default)
- `quiver_color=:black`
- `quiver_shaftwidth=1.5`, `quiver_tipwidth=4.0`, `quiver_tiplength=6.0`: arrow shaft
  thickness and arrowhead size in Makie's pixel-ish units (Makie's own `Arrows2D`
  defaults are `shaftwidth=3`, `tipwidth=14`, `tiplength=8` -- these are already
  slimmer/smaller, tuned so vectors read clearly without looking like thick darts)
- `color_percentile_low=2.0`, `color_percentile_high=98.0`: color scale clipping
- `land_color=:gray55`: NaN/land fill color
- `font_size_title=16`, `font_size_labels=13`, `font_size_ticks=11`
- `ncols=2` (2x2 grid for the typical 4-field case avoids wasted whitespace vs 3 columns)

# Returns
`Figure` -- caller does `save(path, fig)`.
"""
function ValTools.plot_field_panel(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real},
                                    fields::AbstractVector, specs::AbstractDict;
                                    u=nothing, v=nothing, quiver_field=nothing,
                                    title::String="",
                                    fig_width::Real=6.5, fig_height::Real=5.2, dpi::Real=150,
                                    quiver_density_x::Int=35, quiver_density_y::Int=35,
                                    quiver_scale::Real=1.0, quiver_color=:black,
                                    quiver_shaftwidth::Real=1.5, quiver_tipwidth::Real=4.0,
                                    quiver_tiplength::Real=6.0,
                                    color_percentile_low::Real=2.0, color_percentile_high::Real=98.0,
                                    land_color=:gray55,
                                    font_size_title::Real=16, font_size_labels::Real=13,
                                    font_size_ticks::Real=11, ncols::Int=2)
    n = length(fields)
    nrows = cld(n, ncols)
    px_w = round(Int, fig_width * dpi * ncols)
    px_h = round(Int, fig_height * dpi * nrows) + 40
    fig = Figure(size=(px_w, px_h))
    if !isempty(title)
        Label(fig[0, 1:2*ncols], title; fontsize=font_size_title+4, font=:bold)
        rowsize!(fig.layout, 0, Makie.Fixed(40))
    end

    has_uv = u !== nothing && v !== nothing
    sx = sy = 1
    if has_uv
        sx = _op_quiver_stride(length(lon), quiver_density_x)
        sy = _op_quiver_stride(length(lat), quiver_density_y)
    end

    for (idx, (name, field)) in enumerate(fields)
        r = cld(idx, ncols)
        c = ((idx - 1) % ncols) * 2 + 1
        spec = specs[name]
        panel_title = (has_uv && name == quiver_field) ? "$name + velocity" : name
        if has_uv && name == quiver_field
            _op_draw_panel!(fig, (r, c), lon, lat, field, spec, panel_title;
                            u=u, v=v, sx=sx, sy=sy, quiver_color=quiver_color,
                            quiver_scale=quiver_scale, quiver_shaftwidth=quiver_shaftwidth,
                            quiver_tipwidth=quiver_tipwidth, quiver_tiplength=quiver_tiplength,
                            color_percentile_low=color_percentile_low,
                            color_percentile_high=color_percentile_high, land_color=land_color,
                            font_size_title=font_size_title, font_size_labels=font_size_labels,
                            font_size_ticks=font_size_ticks)
        else
            _op_draw_panel!(fig, (r, c), lon, lat, field, spec, panel_title;
                            color_percentile_low=color_percentile_low,
                            color_percentile_high=color_percentile_high, land_color=land_color,
                            font_size_title=font_size_title, font_size_labels=font_size_labels,
                            font_size_ticks=font_size_ticks)
        end
    end

    return fig
end

"""
    animate_field_realtime(lon, lat, read_frame, frame_indices, outpath; spec, kwargs...)

Real-timestep animation of a 2-D gridded field (+ optional velocity quiver)
across actual time indices, one real frame per animation frame (NOT
particle-advection over a static field -- see `plot_lic`/`plot_flow` for
that). Data-source agnostic: `read_frame(i)` is a caller-supplied function
returning `(field, u, v)` for frame index `i` (`u`/`v` may be `nothing`),
so this works with NCDatasets, in-memory arrays, or anything else without
ValTools needing to know the source format.

Color scale AND arrow scale are computed ONCE from ALL frames up front (via
one pass over `frame_indices`) so neither pulses visually from per-frame
autoscaling -- same rationale as the fixed color scale used elsewhere in
this package.

Output format is chosen by `outpath`'s extension exactly like any Makie
`record` call (.mp4/.mov require ffmpeg on PATH; .gif needs no extra deps).

# Arguments
- `lon`, `lat`: 1-D coordinate vectors
- `read_frame`: `Function`, `i -> (field, u, v)` for frame index `i`
- `frame_indices`: iterable of frame indices to pass to `read_frame`
- `outpath`: output animation path

# Keyword Arguments
- `spec`: `NamedTuple` `(cmap=..., label=..., symmetric=::Bool)` for the field
- `title_fn=i->""`: `Function`, frame index -> title string (e.g. include a timestamp)
- `quiver_density_x=35`, `quiver_density_y=35`, `quiver_scale=1.0`, `quiver_color=:black`
- `quiver_shaftwidth=1.5`, `quiver_tipwidth=4.0`, `quiver_tiplength=6.0`: arrow shaft
  thickness and arrowhead size (see `plot_field_panel` for the rationale/defaults)
- `color_percentile_low=2.0`, `color_percentile_high=98.0`, `land_color=:gray55`
- `fig_width=6.5`, `fig_height=5.2`, `dpi=100`
- `font_size_title=16`, `font_size_labels=13`
- `fps=4`
"""
function ValTools.animate_field_realtime(lon::AbstractVector{<:Real}, lat::AbstractVector{<:Real},
                                          read_frame::Function, frame_indices, outpath;
                                          spec, title_fn::Function=i->"",
                                          quiver_density_x::Int=35, quiver_density_y::Int=35,
                                          quiver_scale::Real=1.0, quiver_color=:black,
                                          quiver_shaftwidth::Real=1.5, quiver_tipwidth::Real=4.0,
                                          quiver_tiplength::Real=6.0,
                                          color_percentile_low::Real=2.0, color_percentile_high::Real=98.0,
                                          land_color=:gray55, fig_width::Real=6.5, fig_height::Real=5.2,
                                          dpi::Real=100, font_size_title::Real=16, font_size_labels::Real=13,
                                          fps::Int=4)
    frame_indices = collect(frame_indices)
    isempty(frame_indices) && error("animate_field_realtime: frame_indices is empty")

    first_field, first_u, first_v = read_frame(frame_indices[1])
    has_uv = first_u !== nothing && first_v !== nothing
    sx = _op_quiver_stride(length(lon), quiver_density_x)
    sy = _op_quiver_stride(length(lat), quiver_density_y)
    lon_s = lon[1:sx:end]
    lat_s = lat[1:sy:end]

    # One pass to fix color/arrow scale across the whole animation.
    all_fields = Array{Float64}(undef, length(lon), length(lat), length(frame_indices))
    all_u = has_uv ? Array{Float64}(undef, length(lon_s), length(lat_s), length(frame_indices)) : nothing
    all_v = has_uv ? Array{Float64}(undef, length(lon_s), length(lat_s), length(frame_indices)) : nothing
    for (k, i) in enumerate(frame_indices)
        f, u, v = read_frame(i)
        all_fields[:, :, k] = f
        if has_uv
            all_u[:, :, k] = u[1:sx:end, 1:sy:end]
            all_v[:, :, k] = v[1:sx:end, 1:sy:end]
        end
    end
    vmin, vmax = _op_color_range(all_fields, spec.symmetric, color_percentile_low, color_percentile_high)
    lengthscale = has_uv ? _op_quiver_lengthscale(lon_s, lat_s, all_u, all_v, quiver_scale) : quiver_scale

    fig = Figure(size=(round(Int, fig_width*dpi), round(Int, fig_height*dpi)))
    ax = Axis(fig[1, 1]; xlabel="Longitude", ylabel="Latitude", aspect=DataAspect(),
              titlesize=font_size_title, xlabelsize=font_size_labels, ylabelsize=font_size_labels)
    field_obs = Observable(all_fields[:, :, 1])
    hm = heatmap!(ax, lon, lat, field_obs; colormap=spec.cmap, colorrange=(vmin, vmax),
                  nan_color=land_color)
    Colorbar(fig[1, 2], hm; label=spec.label, labelsize=font_size_labels)

    if has_uv
        u_obs = Observable(all_u[:, :, 1])
        v_obs = Observable(all_v[:, :, 1])
        arrows2d!(ax, lon_s, lat_s, u_obs, v_obs; color=quiver_color,
                  lengthscale=lengthscale, shaftwidth=quiver_shaftwidth,
                  tiplength=quiver_tiplength, tipwidth=quiver_tipwidth)
    end
    ax.title = title_fn(frame_indices[1])

    record(fig, outpath, eachindex(frame_indices); framerate=fps) do k
        field_obs[] = all_fields[:, :, k]
        if has_uv
            u_obs[] = all_u[:, :, k]
            v_obs[] = all_v[:, :, k]
        end
        ax.title = title_fn(frame_indices[k])
    end

    return outpath
end
