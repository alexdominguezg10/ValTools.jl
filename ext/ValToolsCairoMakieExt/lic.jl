function _bilinear_sample(field, x, y)
    ny, nx = size(field)
    x0 = clamp(floor(Int, x), 1, nx)
    y0 = clamp(floor(Int, y), 1, ny)
    x1 = min(x0 + 1, nx)
    y1 = min(y0 + 1, ny)
    fx = x - x0
    fy = y - y0
    return field[y0, x0] * (1 - fx) * (1 - fy) +
           field[y0, x1] * fx * (1 - fy) +
           field[y1, x0] * (1 - fx) * fy +
           field[y1, x1] * fx * fy
end

"""
    lic_texture(u, v; length=30, seed=42, step=0.5)

Compute a Line Integral Convolution texture (Cabral & Leedom 1993).

# Arguments
- `u`, `v`: 2-D velocity field `(ny, nx)`
- `length`: streamline integration steps (forward + backward)
- `seed`: RNG seed for noise texture
- `step`: integration step size (pixels)

# Returns
2-D `Matrix{Float64}` with values in [0, 1].
"""
function ValTools.lic_texture(u::AbstractMatrix{<:Real},
                               v::AbstractMatrix{<:Real};
                               length::Int=30,
                               seed::Int=42,
                               step::Real=0.5)
    ny, nx = size(u)
    rng = Random.MersenneTwister(seed)
    noise = rand(rng, ny, nx)

    uf = Float64.(u)
    vf = Float64.(v)

    result = zeros(ny, nx)

    for j in 1:nx, i in 1:ny
        if !isfinite(uf[i, j]) || !isfinite(vf[i, j])
            result[i, j] = NaN
            continue
        end

        total = noise[i, j]
        count = 1

        # Forward
        x, y = Float64(j), Float64(i)
        for _ in 1:length
            ui = _bilinear_sample(uf, x, y)
            vi = _bilinear_sample(vf, x, y)
            mag = hypot(ui, vi)
            mag < 1e-10 && break
            x += step * ui / mag
            y += step * vi / mag
            (x < 1 || x > nx || y < 1 || y > ny) && break
            val = _bilinear_sample(noise, x, y)
            total += val
            count += 1
        end

        # Backward
        x, y = Float64(j), Float64(i)
        for _ in 1:length
            ui = _bilinear_sample(uf, x, y)
            vi = _bilinear_sample(vf, x, y)
            mag = hypot(ui, vi)
            mag < 1e-10 && break
            x -= step * ui / mag
            y -= step * vi / mag
            (x < 1 || x > nx || y < 1 || y > ny) && break
            val = _bilinear_sample(noise, x, y)
            total += val
            count += 1
        end

        result[i, j] = total / count
    end

    # Normalize to [0, 1]
    finite_vals = filter(isfinite, vec(result))
    if !isempty(finite_vals)
        lo, hi = extrema(finite_vals)
        if hi > lo
            result .= (result .- lo) ./ (hi - lo)
        end
    end

    return result
end

"""
    plot_lic(u, v; field=nothing, title="", figsize=(600, 500))

Quick LIC visualization with optional scalar field overlay.

# Arguments
- `u`, `v`: 2-D velocity field
- `field`: optional 2-D scalar field overlaid with transparency
- `title`: figure title

# Returns
`Figure`
"""
function ValTools.plot_lic(u::AbstractMatrix{<:Real},
                            v::AbstractMatrix{<:Real};
                            field::Union{AbstractMatrix{<:Real}, Nothing}=nothing,
                            title::String="",
                            length::Int=30,
                            seed::Int=42,
                            figsize::Tuple{Int,Int}=(600, 500))
    tex = ValTools.lic_texture(u, v; length=length, seed=seed)

    fig = Figure(size=figsize)
    ax = Axis(fig[1, 1]; aspect=DataAspect())

    if field !== nothing
        heatmap!(ax, Float64.(field'); colormap=:viridis)
        heatmap!(ax, tex'; colormap=[:transparent, :white], alpha=0.6)
    else
        speed = hypot.(Float64.(u), Float64.(v))
        heatmap!(ax, tex'; colormap=:grays)
    end

    !isempty(title) && (ax.title = title)
    hidedecorations!(ax)

    return fig
end
