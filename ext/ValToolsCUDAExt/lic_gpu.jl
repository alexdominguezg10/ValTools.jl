# GPU kernel for Line Integral Convolution.
#
# Each thread traces one streamline (forward + backward) from pixel (i, j).
# The noise texture lives in GPU global memory; bilinear sampling is done
# per-thread. For a 1000×1000 field with length=30 this is ~30M streamline
# steps — highly parallel.

function _bilinear_gpu(field, x, y, ny, nx)
    x0 = clamp(floor(Int32, x), Int32(1), Int32(nx))
    y0 = clamp(floor(Int32, y), Int32(1), Int32(ny))
    x1 = min(x0 + Int32(1), Int32(nx))
    y1 = min(y0 + Int32(1), Int32(ny))
    fx = x - Float64(x0)
    fy = y - Float64(y0)
    return field[y0, x0] * (1.0 - fx) * (1.0 - fy) +
           field[y0, x1] * fx * (1.0 - fy) +
           field[y1, x0] * (1.0 - fx) * fy +
           field[y1, x1] * fx * fy
end

function _lic_kernel!(result, u, v, noise, ny, nx, len, step)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = ny * nx
    idx > total && return nothing

    idx0 = idx - 1
    i = idx0 % ny + 1
    j = idx0 ÷ ny + 1

    ui = u[i, j]
    vi = v[i, j]
    if !isfinite(ui) || !isfinite(vi)
        result[i, j] = NaN
        return nothing
    end

    total_val = noise[i, j]
    count = 1

    # Forward integration
    x = Float64(j)
    y = Float64(i)
    for _ in 1:len
        ux = _bilinear_gpu(u, x, y, ny, nx)
        vx = _bilinear_gpu(v, x, y, ny, nx)
        mag = sqrt(ux * ux + vx * vx)
        mag < 1.0e-10 && break
        x += step * ux / mag
        y += step * vx / mag
        (x < 1.0 || x > Float64(nx) || y < 1.0 || y > Float64(ny)) && break
        total_val += _bilinear_gpu(noise, x, y, ny, nx)
        count += 1
    end

    # Backward integration
    x = Float64(j)
    y = Float64(i)
    for _ in 1:len
        ux = _bilinear_gpu(u, x, y, ny, nx)
        vx = _bilinear_gpu(v, x, y, ny, nx)
        mag = sqrt(ux * ux + vx * vx)
        mag < 1.0e-10 && break
        x -= step * ux / mag
        y -= step * vx / mag
        (x < 1.0 || x > Float64(nx) || y < 1.0 || y > Float64(ny)) && break
        total_val += _bilinear_gpu(noise, x, y, ny, nx)
        count += 1
    end

    result[i, j] = total_val / Float64(count)
    return nothing
end

"""
    lic_texture_gpu(u, v; length=30, seed=42, step=0.5)

GPU-accelerated Line Integral Convolution (Cabral & Leedom 1993).

Same interface as `lic_texture`. Each pixel's streamline integration runs
as an independent GPU thread. For a 1000×1000 velocity field this is
~50-100× faster than the CPU version.

Returns a CPU `Matrix{Float64}` with values normalized to [0, 1].
"""
function ValTools.lic_texture_gpu(u::AbstractMatrix{<:Real},
                                   v::AbstractMatrix{<:Real};
                                   length::Int=30,
                                   seed::Int=42,
                                   step::Real=0.5)
    ny, nx = size(u)

    rng = Random.MersenneTwister(seed)
    noise = rand(rng, Float64, ny, nx)

    d_u     = CuArray(Float64.(u))
    d_v     = CuArray(Float64.(v))
    d_noise = CuArray(noise)
    d_out   = CUDA.zeros(Float64, ny, nx)

    total = ny * nx
    threads = 256
    blocks = cld(total, threads)

    @cuda threads=threads blocks=blocks _lic_kernel!(
        d_out, d_u, d_v, d_noise, ny, nx, Int32(length), Float64(step))

    result = Array(d_out)

    # Normalize to [0, 1]
    finite_vals = filter(isfinite, vec(result))
    if !isempty(finite_vals)
        lo, hi = extrema(finite_vals)
        if hi > lo
            @. result = (result - lo) / (hi - lo)
        end
    end

    return result
end
