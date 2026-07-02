# GPU kernel for vertical interpolation from sigma levels to target z-depths.
#
# Each thread handles one (i, j, iz, it) output point: searches the sigma
# column for the bracketing levels and linearly interpolates.

function _interp_z_kernel!(out, var, z, depths, ny, nx, N, nt, nz)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = ny * nx * nz * nt
    idx > total && return nothing

    idx0 = idx - 1
    i  = idx0 % ny + 1
    idx0 = idx0 ÷ ny
    j  = idx0 % nx + 1
    idx0 = idx0 ÷ nx
    iz = idx0 % nz + 1
    it = idx0 ÷ nz + 1

    zd = depths[iz]
    z_bot = z[i, j, 1, it]
    z_sfc = z[i, j, N, it]

    if zd < z_bot || zd > z_sfc
        out[i, j, iz, it] = NaN
        return nothing
    end

    # Find bracketing levels
    k0 = 1
    for k in 1:N-1
        if z[i, j, k, it] <= zd
            k0 = k
        end
    end
    k1 = min(k0 + 1, N)

    z0 = z[i, j, k0, it]
    z1 = z[i, j, k1, it]
    v0 = var[i, j, k0, it]
    v1 = var[i, j, k1, it]

    if !isfinite(v0) || !isfinite(v1)
        out[i, j, iz, it] = NaN
        return nothing
    end

    dz = z1 - z0
    w = abs(dz) > 1.0e-6 ? clamp((zd - z0) / dz, 0.0, 1.0) : 0.5
    out[i, j, iz, it] = v0 + w * (v1 - v0)

    return nothing
end

"""
    interp_z_gpu(var, z, depths)

GPU-accelerated vertical interpolation from sigma levels to target depths.

Same interface as `interp_z`. Transfers data to GPU, runs one thread per
output point, returns CPU array.

- `var`: `(ny, nx, N, nt)` variable on sigma levels
- `z`: `(ny, nx, N, nt)` depths on sigma levels [m, negative down]
- `depths`: vector of target depths [m, negative down]

Returns `(ny, nx, nz, nt)`.
"""
function ValTools.interp_z_gpu(var::AbstractArray{<:Real, 4},
                                z::AbstractArray{<:Real, 4},
                                depths::AbstractVector{<:Real})
    ny, nx, N, nt = size(var)
    nz = length(depths)

    d_var    = CuArray(Float64.(var))
    d_z      = CuArray(Float64.(z))
    d_depths = CuArray(Float64.(depths))
    d_out    = CUDA.fill(NaN, ny, nx, nz, nt)

    total = ny * nx * nz * nt
    threads = 256
    blocks = cld(total, threads)

    @cuda threads=threads blocks=blocks _interp_z_kernel!(
        d_out, d_var, d_z, d_depths, ny, nx, N, nt, nz)

    return Array(d_out)
end

function ValTools.interp_z_gpu(var::AbstractArray{<:Real, 3},
                                z::AbstractArray{<:Real, 3},
                                depths::AbstractVector{<:Real})
    var4 = reshape(var, size(var, 1), size(var, 2), size(var, 3), 1)
    z4   = reshape(z, size(z, 1), size(z, 2), size(z, 3), 1)
    out4 = ValTools.interp_z_gpu(var4, z4, depths)
    return dropdims(out4; dims=4)
end
