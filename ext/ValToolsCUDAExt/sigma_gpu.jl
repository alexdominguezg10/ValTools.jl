# GPU kernel for sigma_to_z
#
# Each thread computes z for one (i, j, k, it) point.
# For a typical CROCO grid (ny=400, nx=500, N=32, nt=100) that's 640M points
# — perfect for GPU parallelism.

function _sigma_to_z_kernel_v2!(z, h, zeta, sc_r, Cs_r, hc, ny, nx, N, nt)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = ny * nx * N * nt
    idx > total && return nothing

    # Decompose linear index → (i, j, k, it)  (column-major)
    idx0 = idx - 1
    i  = idx0 % ny + 1
    idx0 = idx0 ÷ ny
    j  = idx0 % nx + 1
    idx0 = idx0 ÷ nx
    k  = idx0 % N + 1
    it = idx0 ÷ N + 1

    S = (hc * sc_r[k] + h[i, j] * Cs_r[k]) / (hc + h[i, j])
    z[i, j, k, it] = zeta[i, j, it] * (1.0 + S) + h[i, j] * S

    return nothing
end

function _sigma_to_z_kernel_v1!(z, h, zeta, sc_r, Cs_r, hc, ny, nx, N, nt)
    idx = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    total = ny * nx * N * nt
    idx > total && return nothing

    idx0 = idx - 1
    i  = idx0 % ny + 1
    idx0 = idx0 ÷ ny
    j  = idx0 % nx + 1
    idx0 = idx0 ÷ nx
    k  = idx0 % N + 1
    it = idx0 ÷ N + 1

    z0 = hc * sc_r[k] + (h[i, j] - hc) * Cs_r[k]
    z[i, j, k, it] = z0 + zeta[i, j, it] * (1.0 + z0 / h[i, j])

    return nothing
end

"""
    sigma_to_z_gpu(h, zeta, sc_r, Cs_r, hc; vtransform=2)

GPU-accelerated sigma-to-z coordinate transform.

Same interface as `sigma_to_z` but runs on the GPU. Input arrays are
transferred to device, computation runs in parallel, result is returned
as a CPU array.

For a 400×500×32×100 grid this is ~20-50× faster than the CPU version.
"""
function ValTools.sigma_to_z_gpu(h::AbstractMatrix, zeta::AbstractArray,
                                  sc_r::AbstractVector, Cs_r::AbstractVector,
                                  hc::Real; vtransform::Int=2)
    ny, nx = size(h)
    N = length(sc_r)

    if ndims(zeta) == 2
        nt = 1
        zeta3 = reshape(zeta, ny, nx, 1)
    else
        nt = size(zeta, 3)
        zeta3 = zeta
    end

    d_h    = CuArray(Float64.(h))
    d_zeta = CuArray(Float64.(zeta3))
    d_sc   = CuArray(Float64.(sc_r))
    d_Cs   = CuArray(Float64.(Cs_r))
    d_z    = CUDA.zeros(Float64, ny, nx, N, nt)

    total = ny * nx * N * nt
    threads = 256
    blocks = cld(total, threads)

    if vtransform == 1
        @cuda threads=threads blocks=blocks _sigma_to_z_kernel_v1!(
            d_z, d_h, d_zeta, d_sc, d_Cs, Float64(hc), ny, nx, N, nt)
    else
        @cuda threads=threads blocks=blocks _sigma_to_z_kernel_v2!(
            d_z, d_h, d_zeta, d_sc, d_Cs, Float64(hc), ny, nx, N, nt)
    end

    z = Array(d_z)
    return ndims(zeta) == 2 ? dropdims(z; dims=4) : z
end
