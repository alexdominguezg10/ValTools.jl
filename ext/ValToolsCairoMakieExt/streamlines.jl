# ════════════════════════════════════════════════════════════════════
# Streamline tracing engine (RK4)
# ════════════════════════════════════════════════════════════════════

function _bilinear_vel(u, v, x, y, ny, nx)
    x0 = clamp(floor(Int, x), 1, nx - 1)
    y0 = clamp(floor(Int, y), 1, ny - 1)
    x1 = x0 + 1
    y1 = y0 + 1
    fx = x - x0
    fy = y - y0

    ui = u[y0, x0] * (1 - fx) * (1 - fy) + u[y0, x1] * fx * (1 - fy) +
         u[y1, x0] * (1 - fx) * fy + u[y1, x1] * fx * fy
    vi = v[y0, x0] * (1 - fx) * (1 - fy) + v[y0, x1] * fx * (1 - fy) +
         v[y1, x0] * (1 - fx) * fy + v[y1, x1] * fx * fy
    return ui, vi
end

function _trace_streamline(u, v, x0, y0, ny, nx, max_steps, ds; forward=true)
    xs = Float64[x0]
    ys = Float64[y0]
    x, y = x0, y0
    sign = forward ? 1.0 : -1.0

    for _ in 1:max_steps
        (x < 1.5 || x > nx - 0.5 || y < 1.5 || y > ny - 0.5) && break

        u1, v1 = _bilinear_vel(u, v, x, y, ny, nx)
        mag1 = hypot(u1, v1)
        mag1 < 1e-10 && break

        dx1, dy1 = sign * ds * u1 / mag1, sign * ds * v1 / mag1

        xm, ym = x + 0.5 * dx1, y + 0.5 * dy1
        (xm < 1 || xm > nx || ym < 1 || ym > ny) && break
        u2, v2 = _bilinear_vel(u, v, xm, ym, ny, nx)
        mag2 = hypot(u2, v2)
        mag2 < 1e-10 && break

        dx2, dy2 = sign * ds * u2 / mag2, sign * ds * v2 / mag2

        xm2, ym2 = x + 0.5 * dx2, y + 0.5 * dy2
        (xm2 < 1 || xm2 > nx || ym2 < 1 || ym2 > ny) && break
        u3, v3 = _bilinear_vel(u, v, xm2, ym2, ny, nx)
        mag3 = hypot(u3, v3)
        mag3 < 1e-10 && break

        dx3, dy3 = sign * ds * u3 / mag3, sign * ds * v3 / mag3

        xe, ye = x + dx3, y + dy3
        (xe < 1 || xe > nx || ye < 1 || ye > ny) && break
        u4, v4 = _bilinear_vel(u, v, xe, ye, ny, nx)
        mag4 = hypot(u4, v4)
        mag4 < 1e-10 && break

        dx4, dy4 = sign * ds * u4 / mag4, sign * ds * v4 / mag4

        x += (dx1 + 2dx2 + 2dx3 + dx4) / 6
        y += (dy1 + 2dy2 + 2dy3 + dy4) / 6

        (x < 1 || x > nx || y < 1 || y > ny) && break
        (!isfinite(x) || !isfinite(y)) && break

        push!(xs, x)
        push!(ys, y)
    end

    return xs, ys
end

function _seed_grid(ny, nx, density)
    step_x = max(1, round(Int, nx / (density * sqrt(nx / ny))))
    step_y = max(1, round(Int, ny / (density * sqrt(ny / nx))))
    seeds = Tuple{Float64, Float64}[]
    for j in step_x÷2+1:step_x:nx, i in step_y÷2+1:step_y:ny
        push!(seeds, (Float64(j), Float64(i)))
    end
    return seeds
end

function _compute_streamlines(u, v; density=1.5, max_steps=300, ds=0.5, min_length=5)
    ny, nx = size(u)
    seeds = _seed_grid(ny, nx, density)

    occupied = falses(ny, nx)
    all_lines = Vector{Tuple{Vector{Float64}, Vector{Float64}}}()

    for (sx, sy) in seeds
        ix, iy = clamp(round(Int, sx), 1, nx), clamp(round(Int, sy), 1, ny)
        occupied[iy, ix] && continue

        (!isfinite(u[iy, ix]) || !isfinite(v[iy, ix])) && continue
        hypot(u[iy, ix], v[iy, ix]) < 1e-10 && continue

        xf, yf = _trace_streamline(u, v, sx, sy, ny, nx, max_steps, ds; forward=true)
        xb, yb = _trace_streamline(u, v, sx, sy, ny, nx, max_steps, ds; forward=false)

        xs = vcat(reverse(xb), xf[2:end])
        ys = vcat(reverse(yb), yf[2:end])

        length(xs) < min_length && continue

        for k in 1:length(xs)
            ii = clamp(round(Int, ys[k]), 1, ny)
            jj = clamp(round(Int, xs[k]), 1, nx)
            occupied[ii, jj] = true
        end

        push!(all_lines, (xs, ys))
    end

    return all_lines
end


# ════════════════════════════════════════════════════════════════════
# Public API
# ════════════════════════════════════════════════════════════════════

"""
    plot_streamlines(u, v; kwargs...) → Figure

Publication-quality streamline plot for ocean current visualization.

Traces streamlines via 4th-order Runge-Kutta integration with density
control to avoid clutter. Lines are colored by local speed.

# Arguments (positional)
- `u`: eastward velocity `(ny, nx)` [m/s]
- `v`: northward velocity `(ny, nx)` [m/s]

# Keyword Arguments
- `lon=nothing`, `lat=nothing`: coordinate vectors for axis labels
- `field=nothing`: 2-D scalar background (e.g. SST, SSH) plotted as heatmap
- `field_cmap=:viridis`: colormap for the background field
- `field_label=""`: colorbar label for the background field
- `density=1.5`: streamline seeding density (higher = more lines)
- `max_steps=300`: max RK4 steps per streamline
- `ds=0.5`: integration step size [grid cells]
- `min_length=5`: discard streamlines shorter than this
- `line_cmap=:tempo`: colormap for streamline speed coloring
- `linewidth=1.5`: streamline width
- `arrow_spacing=15`: place an arrowhead every N points along each streamline
- `arrow_size=8`: arrowhead size [pt]
- `title=""`: figure title
- `figsize=(700, 600)`: figure size in pixels

# Returns
`Makie.Figure`
"""
function ValTools.plot_streamlines(u::AbstractMatrix{<:Real},
                                   v::AbstractMatrix{<:Real};
                                   lon::Union{AbstractVector{<:Real}, Nothing}=nothing,
                                   lat::Union{AbstractVector{<:Real}, Nothing}=nothing,
                                   field::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
                                   field_cmap=:viridis,
                                   field_label::String="",
                                   density::Real=1.5,
                                   max_steps::Int=300,
                                   ds::Real=0.5,
                                   min_length::Int=5,
                                   line_cmap=:tempo,
                                   linewidth::Real=1.5,
                                   arrow_spacing::Int=15,
                                   arrow_size::Real=8.0,
                                   title::String="",
                                   figsize::Tuple{Int,Int}=(700, 600))
    ny, nx = size(u)
    uf = Float64.(u)
    vf = Float64.(v)

    x_coords = lon !== nothing ? Float64.(lon) : collect(1.0:nx)
    y_coords = lat !== nothing ? Float64.(lat) : collect(1.0:ny)

    speed = hypot.(uf, vf)
    speed_finite = filter(isfinite, vec(speed))
    spd_min = isempty(speed_finite) ? 0.0 : minimum(speed_finite)
    spd_max = isempty(speed_finite) ? 1.0 : quantile(speed_finite, 0.98)
    spd_max = max(spd_max, spd_min + 1e-10)

    lines_data = _compute_streamlines(uf, vf; density=density,
                                       max_steps=max_steps, ds=ds,
                                       min_length=min_length)

    fig = Figure(size=figsize)

    xlabel = lon !== nothing ? "Longitude" : ""
    ylabel = lat !== nothing ? "Latitude" : ""
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=xlabel, ylabel=ylabel)

    if field !== nothing
        hm = heatmap!(ax, x_coords, y_coords, Float64.(field);
                       colormap=field_cmap)
        Colorbar(fig[1, 2], hm; label=field_label)
    end

    cmap_obj = Makie.to_colormap(line_cmap)
    n_colors = length(cmap_obj)

    for (xs, ys) in lines_data
        n_pts = length(xs)
        n_pts < 2 && continue

        lx = [_grid_to_coord(xs[k], x_coords) for k in 1:n_pts]
        ly = [_grid_to_coord(ys[k], y_coords) for k in 1:n_pts]

        seg_speeds = Float64[]
        for k in 1:n_pts
            ix = clamp(round(Int, xs[k]), 1, nx)
            iy = clamp(round(Int, ys[k]), 1, ny)
            push!(seg_speeds, speed[iy, ix])
        end

        for k in 1:n_pts-1
            spd_k = 0.5 * (seg_speeds[k] + seg_speeds[k+1])
            t = clamp((spd_k - spd_min) / (spd_max - spd_min), 0.0, 1.0)
            ci = clamp(round(Int, t * (n_colors - 1)) + 1, 1, n_colors)
            c = cmap_obj[ci]

            lw = linewidth * (0.6 + 0.8 * t)

            lines!(ax, [lx[k], lx[k+1]], [ly[k], ly[k+1]];
                   color=c, linewidth=lw)
        end

        if arrow_spacing > 0 && n_pts > arrow_spacing
            for k in arrow_spacing:arrow_spacing:n_pts-1
                dx = lx[min(k+1, n_pts)] - lx[max(k-1, 1)]
                dy = ly[min(k+1, n_pts)] - ly[max(k-1, 1)]
                mag = hypot(dx, dy)
                mag < 1e-10 && continue

                spd_k = seg_speeds[k]
                t = clamp((spd_k - spd_min) / (spd_max - spd_min), 0.0, 1.0)
                ci = clamp(round(Int, t * (n_colors - 1)) + 1, 1, n_colors)
                c = cmap_obj[ci]

                scatter!(ax, [lx[k]], [ly[k]];
                         marker=:rtriangle,
                         markersize=arrow_size * (0.7 + 0.6 * t),
                         color=c,
                         rotation=[atan(dy, dx)])
            end
        end
    end

    if field === nothing
        Colorbar(fig[1, 2]; colormap=line_cmap,
                 limits=(spd_min, spd_max),
                 label="Speed [m/s]")
    end

    !isempty(title) && (ax.title = title)

    return fig
end

function _grid_to_coord(gi, coords)
    n = length(coords)
    i0 = clamp(floor(Int, gi), 1, n - 1)
    i1 = i0 + 1
    f = gi - i0
    return coords[i0] + f * (coords[i1] - coords[i0])
end


"""
    plot_flow(u, v; kwargs...) → Figure

Combined LIC texture + streamline overlay for maximum visual impact.

Renders a Line Integral Convolution texture as the background, then
overlays sparse streamlines with arrowheads for direction. The LIC
shows fine-scale flow structure while the streamlines provide
directionality at a glance.

# Arguments
- `u`, `v`: velocity field `(ny, nx)`

# Keyword Arguments
- `lon`, `lat`: coordinate vectors
- `field=nothing`: scalar field for color (defaults to speed)
- `field_cmap=:deep`: colormap for the LIC-modulated background
- `field_label="Speed [m/s]"`: colorbar label
- `lic_length=25`: LIC kernel length
- `lic_alpha=0.6`: LIC texture opacity
- `stream_density=0.8`: streamline density (sparser than plot_streamlines)
- `stream_color=:white`: streamline color (or `:speed` for speed-colored)
- `arrow_spacing=20`: arrowhead spacing
- `title=""`: figure title
- `figsize=(700, 600)`

# Returns
`Makie.Figure`
"""
function ValTools.plot_flow(u::AbstractMatrix{<:Real},
                             v::AbstractMatrix{<:Real};
                             lon::Union{AbstractVector{<:Real}, Nothing}=nothing,
                             lat::Union{AbstractVector{<:Real}, Nothing}=nothing,
                             field::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
                             field_cmap=:deep,
                             field_label::String="Speed [m/s]",
                             lic_length::Int=25,
                             lic_alpha::Real=0.6,
                             stream_density::Real=0.8,
                             stream_color::Union{Symbol, Makie.Colorant}=:white,
                             arrow_spacing::Int=20,
                             title::String="",
                             figsize::Tuple{Int,Int}=(700, 600))
    ny, nx = size(u)
    uf = Float64.(u)
    vf = Float64.(v)

    x_coords = lon !== nothing ? Float64.(lon) : collect(1.0:nx)
    y_coords = lat !== nothing ? Float64.(lat) : collect(1.0:ny)

    speed = hypot.(uf, vf)
    bg_field = field !== nothing ? Float64.(field) : speed

    tex = ValTools.lic_texture(uf, vf; length=lic_length)

    modulated = bg_field .* (0.4 .+ 0.6 .* tex)
    for i in eachindex(modulated)
        if !isfinite(bg_field[i])
            modulated[i] = NaN
        end
    end

    fig = Figure(size=figsize)
    xlabel = lon !== nothing ? "Longitude" : ""
    ylabel = lat !== nothing ? "Latitude" : ""
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel=xlabel, ylabel=ylabel,
              backgroundcolor=:black)

    hm = heatmap!(ax, x_coords, y_coords, modulated; colormap=field_cmap)
    Colorbar(fig[1, 2], hm; label=field_label)

    lines_data = _compute_streamlines(uf, vf; density=stream_density,
                                       max_steps=200, ds=0.6, min_length=8)

    for (xs, ys) in lines_data
        n_pts = length(xs)
        n_pts < 2 && continue

        lx = [_grid_to_coord(xs[k], x_coords) for k in 1:n_pts]
        ly = [_grid_to_coord(ys[k], y_coords) for k in 1:n_pts]

        c = stream_color == :speed ? (:white, 0.7) : (stream_color, 0.7)
        lines!(ax, lx, ly; color=c, linewidth=0.8)

        if arrow_spacing > 0 && n_pts > arrow_spacing
            for k in arrow_spacing:arrow_spacing:n_pts-1
                dx = lx[min(k+1, n_pts)] - lx[max(k-1, 1)]
                dy = ly[min(k+1, n_pts)] - ly[max(k-1, 1)]
                mag = hypot(dx, dy)
                mag < 1e-10 && continue

                scatter!(ax, [lx[k]], [ly[k]];
                         marker=:rtriangle, markersize=6,
                         color=(stream_color, 0.9),
                         rotation=[atan(dy, dx)])
            end
        end
    end

    !isempty(title) && (ax.title = title)

    return fig
end
