"""
    build_stretching(N, theta_s, theta_b; vstretching=4)

Compute sigma at rho-points `sc_r` and stretching function `Cs_r`.

# Arguments
- `N`: number of sigma layers
- `theta_s`: surface control parameter (0–10)
- `theta_b`: bottom control parameter (0–4)
- `vstretching`: 1 = Song & Haidvogel 1994; 4 = Shchepetkin 2010 (CROCO default)

# Returns
`(sc_r, Cs_r)` each `Vector{Float64}` of length `N`.
"""
function build_stretching(N::Int, theta_s::Real, theta_b::Real; vstretching::Int=4)
    sc_r = @. -1.0 + (0.5 + (0:N-1)) / N

    if vstretching == 1
        if theta_s > 0
            csrf = @. (1 - theta_b) * sinh(theta_s * sc_r) / sinh(theta_s)
            csrb = @. theta_b * (tanh(theta_s * (sc_r + 0.5)) -
                                  tanh(theta_s * 0.5)) /
                       (2.0 * tanh(theta_s * 0.5))
            Cs_r = csrf .+ csrb
        else
            Cs_r = copy(sc_r)
        end
    elseif vstretching == 4
        if theta_s > 0
            csrf = @. (1.0 - cosh(theta_s * sc_r)) / (cosh(theta_s) - 1.0)
        else
            csrf = @. -(sc_r ^ 2)
        end
        if theta_b > 0
            Cs_r = @. (exp(theta_b * csrf) - 1.0) / (1.0 - exp(-theta_b))
        else
            Cs_r = csrf
        end
    else
        Cs_r = copy(sc_r)
    end

    return sc_r, Cs_r
end


"""
    sigma_to_z(h, zeta, sc_r, Cs_r, hc; vtransform=2)

Compute physical depths from sigma coordinates.

# Arguments
- `h`: bathymetry `(ny, nx)` [m, positive]
- `zeta`: sea-surface height `(ny, nx, nt)` [m]
- `sc_r`: sigma at rho-points `(N,)`
- `Cs_r`: stretching function `(N,)`
- `hc`: critical depth [m]
- `vtransform`: 1 or 2

# Returns
`z` array `(ny, nx, N, nt)` [m, negative downward].

Note: Julia uses column-major order. The spatial dims come first for
efficient memory access when looping over grid cells.
"""
function sigma_to_z(h::AbstractMatrix, zeta::AbstractArray,
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

    z = Array{Float64}(undef, ny, nx, N, nt)

    if vtransform == 1
        @inbounds for it in 1:nt, k in 1:N, j in 1:nx, i in 1:ny
            z0 = hc * sc_r[k] + (h[i,j] - hc) * Cs_r[k]
            z[i,j,k,it] = z0 + zeta3[i,j,it] * (1.0 + z0 / h[i,j])
        end
    else
        @inbounds for it in 1:nt, k in 1:N, j in 1:nx, i in 1:ny
            S = (hc * sc_r[k] + h[i,j] * Cs_r[k]) / (hc + h[i,j])
            z[i,j,k,it] = zeta3[i,j,it] * (1.0 + S) + h[i,j] * S
        end
    end

    return ndims(zeta) == 2 ? dropdims(z; dims=4) : z
end


"""
    interp_z(var, z, depths)

Vertically interpolate a sigma-level field to target z-depths.

# Arguments
- `var`: variable on sigma levels `(ny, nx, N, nt)`
- `z`: depths on sigma levels `(ny, nx, N, nt)` [m, negative down]
- `depths`: target depths [m, negative down]

# Returns
Array `(ny, nx, nz, nt)`.
"""
function interp_z(var::AbstractArray{T,4}, z::AbstractArray{<:Real,4},
                  depths::AbstractVector{<:Real}) where T
    ny, nx, N, nt = size(var)
    nz = length(depths)
    out = fill(T(NaN), ny, nx, nz, nt)

    @inbounds for it in 1:nt, iz in 1:nz
        zd = depths[iz]
        for j in 1:nx, i in 1:ny
            z_bot = z[i,j,1,it]
            z_sfc = z[i,j,N,it]
            if zd < z_bot || zd > z_sfc
                continue
            end

            k0 = 1
            for k in 1:N-1
                if z[i,j,k,it] <= zd
                    k0 = k
                end
            end
            k1 = min(k0 + 1, N)

            z0 = z[i,j,k0,it]
            z1 = z[i,j,k1,it]
            v0 = var[i,j,k0,it]
            v1 = var[i,j,k1,it]

            if !isfinite(v0) || !isfinite(v1)
                continue
            end

            dz = z1 - z0
            w = abs(dz) > 1e-6 ? clamp((zd - z0) / dz, 0.0, 1.0) : 0.5
            out[i,j,iz,it] = v0 + w * (v1 - v0)
        end
    end

    return out
end

function interp_z(var::AbstractArray{T,3}, z::AbstractArray{<:Real,3},
                  depths::AbstractVector{<:Real}) where T
    var4 = reshape(var, size(var,1), size(var,2), size(var,3), 1)
    z4   = reshape(z,   size(z,1),   size(z,2),   size(z,3),   1)
    out4 = interp_z(var4, z4, depths)
    return dropdims(out4; dims=4)
end
