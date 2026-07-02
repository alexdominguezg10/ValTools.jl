const _BATHY_VARS = ("h", "depth", "bathy")
const _LON_RHO_VARS = ("lon_rho", "longitude", "nav_lon", "lon")
const _LAT_RHO_VARS = ("lat_rho", "latitude", "nav_lat", "lat")
const _MASK_RHO_VARS = ("mask_rho", "mask", "tmask")
const _SC_VARS = ("sc_r", "s_rho")
const _CS_VARS = ("Cs_r", "Cs_at_rho_points")
const _TIME_VARS = ("time", "ocean_time", "scrum_time")

function _find_var(ds, candidates)
    for name in candidates
        if haskey(ds, name)
            return name
        end
    end
    return nothing
end

function _read_var(ds, candidates)
    name = _find_var(ds, candidates)
    name === nothing && return nothing
    return Array(ds[name])
end

function _get_attr(ds, key, default=nothing)
    if haskey(ds.attrib, key)
        return ds.attrib[key]
    end
    return default
end

"""
    CROCOReader

Lazy reader for CROCO/ROMS sigma-coordinate NetCDF output.

# Fields
- `ds`: `NCDataset` handle (lazy, stays open)
- `grid`: grid `NCDataset` (same as `ds` if no separate grid file)
- `h`: bathymetry `(ny, nx)` [m, positive]
- `lon`: longitudes at rho-points `(ny, nx)` or `(nx,)`
- `lat`: latitudes at rho-points `(ny, nx)` or `(ny,)`
- `mask`: land-sea mask `(ny, nx)` or `nothing`
- `sc_r`: sigma at rho-points `(N,)`
- `Cs_r`: stretching function `(N,)`
- `hc`: critical depth [m]
- `vtransform`: 1 or 2
"""
mutable struct CROCOReader
    ds::NCDataset
    grid::NCDataset
    h::Matrix{Float64}
    lon::Array{Float64}
    lat::Array{Float64}
    mask::Union{Matrix{Float64}, Nothing}
    sc_r::Vector{Float64}
    Cs_r::Vector{Float64}
    hc::Float64
    vtransform::Int
end

function CROCOReader(files::AbstractString;
                     grid_file::Union{AbstractString, Nothing}=nothing,
                     vtransform::Union{Int, Nothing}=nothing,
                     decode_times::Bool=true)
    ds = _open_nc(files)
    grd = grid_file !== nothing ? NCDataset(grid_file) : ds

    vt = if vtransform !== nothing
        vtransform
    else
        vt_attr = _get_attr(ds, "Vtransform", nothing)
        vt_attr !== nothing ? Int(vt_attr) : 2
    end

    hc_val = if haskey(ds, "hc")
        Float64(Array(ds["hc"])[1])
    else
        hc_attr = _get_attr(ds, "hc", _get_attr(ds, "Tcline", 300.0))
        Float64(hc_attr)
    end

    h_arr = _read_var(grd, _BATHY_VARS)
    h_arr === nothing && error("Cannot find bathymetry (h/depth/bathy) in dataset/grid file.")
    h_mat = Float64.(ndims(h_arr) == 2 ? h_arr : dropdims(h_arr; dims=tuple(findall(size(h_arr) .== 1)...)))

    lon_arr = _read_var(grd, _LON_RHO_VARS)
    lat_arr = _read_var(grd, _LAT_RHO_VARS)
    lon_arr === nothing && error("Cannot find longitude in dataset/grid file.")
    lat_arr === nothing && error("Cannot find latitude in dataset/grid file.")

    mask_arr = _read_var(grd, _MASK_RHO_VARS)
    mask_mat = mask_arr !== nothing ? Float64.(ndims(mask_arr) <= 2 ? mask_arr :
        dropdims(mask_arr; dims=tuple(findall(size(mask_arr) .== 1)...))) : nothing

    sc_r_arr = _read_var(ds, _SC_VARS)
    cs_r_arr = _read_var(ds, _CS_VARS)

    if sc_r_arr === nothing || cs_r_arr === nothing
        N = _get_dim(ds, ("s_rho", "N"), 32)
        theta_s = Float64(_get_attr(ds, "theta_s", 7.0))
        theta_b = Float64(_get_attr(ds, "theta_b", 0.0))
        sc_r_arr, cs_r_arr = build_stretching(N, theta_s, theta_b)
    end

    return CROCOReader(ds, grd, h_mat, Float64.(lon_arr), Float64.(lat_arr),
                       mask_mat, vec(Float64.(sc_r_arr)), vec(Float64.(cs_r_arr)),
                       hc_val, vt)
end

function _open_nc(path::AbstractString)
    expanded = glob(path)
    if length(expanded) > 1
        sort!(expanded)
        return NCDataset(expanded, "r")
    elseif length(expanded) == 1
        return NCDataset(expanded[1], "r")
    else
        return NCDataset(path, "r")
    end
end

function _get_dim(ds, candidates, default)
    for c in candidates
        if haskey(ds.dim, c)
            return ds.dim[c]
        end
    end
    return default
end

function _time_values(ds)
    tn = _find_var(ds, _TIME_VARS)
    tn === nothing && error("No time coordinate found.")
    return Array(ds[tn])
end

function _apply_mask!(data::AbstractArray, mask::Union{Matrix{Float64}, Nothing})
    mask === nothing && return data
    ny, nx = size(mask)
    for idx in CartesianIndices(data)
        i = mod1(idx[1], ny)
        j = mod1(idx[2], nx)
        if mask[i, j] <= 0
            data[idx] = NaN
        end
    end
    return data
end

# ── Surface fields ─────────────────────────────────────────────────

"""
    ssh(r::CROCOReader; apply_mask=true) → (data, lon, lat, time)

Sea-surface height ζ [m]. `data` has shape `(ny, nx, nt)`.
"""
function ssh(r::CROCOReader; apply_mask::Bool=true)
    data = Float64.(Array(r.ds["zeta"]))
    apply_mask && _apply_mask!(data, r.mask)
    return (data=data, lon=r.lon, lat=r.lat, time=_time_values(r.ds))
end

"""
    sst(r::CROCOReader; apply_mask=true) → (data, lon, lat, time)

Sea-surface temperature [°C]. Extracts top sigma level.
"""
function sst(r::CROCOReader; apply_mask::Bool=true)
    data = _surface_level(r, "temp")
    apply_mask && _apply_mask!(data, r.mask)
    return (data=data, lon=r.lon, lat=r.lat, time=_time_values(r.ds))
end

"""
    sss(r::CROCOReader; apply_mask=true) → (data, lon, lat, time)

Sea-surface salinity [PSU]. Extracts top sigma level.
"""
function sss(r::CROCOReader; apply_mask::Bool=true)
    data = _surface_level(r, "salt")
    apply_mask && _apply_mask!(data, r.mask)
    return (data=data, lon=r.lon, lat=r.lat, time=_time_values(r.ds))
end

function _surface_level(r::CROCOReader, varname::String)
    var = r.ds[varname]
    dims = dimnames(var)
    s_dim_idx = findfirst(d -> d in ("s_rho", "N"), dims)
    s_dim_idx === nothing && error("No sigma dimension found for $varname")

    all_data = Float64.(Array(var))
    N_sigma = size(all_data, s_dim_idx)
    return selectdim(all_data, s_dim_idx, N_sigma) |> collect
end

# ── 3-D fields ─────────────────────────────────────────────────────

"""
    temperature(r::CROCOReader; depths=nothing, apply_mask=true)

Temperature [°C]. If `depths` given (vector, m negative downward),
interpolates from sigma levels.
"""
function temperature(r::CROCOReader; depths=nothing, apply_mask::Bool=true)
    data = Float64.(Array(r.ds["temp"]))
    if depths !== nothing
        data = _interp_sigma(r, data, Float64.(collect(depths)))
    end
    apply_mask && _apply_mask!(data, r.mask)
    return (data=data, lon=r.lon, lat=r.lat, time=_time_values(r.ds))
end

"""
    salinity(r::CROCOReader; depths=nothing, apply_mask=true)

Salinity [PSU].
"""
function salinity(r::CROCOReader; depths=nothing, apply_mask::Bool=true)
    data = Float64.(Array(r.ds["salt"]))
    if depths !== nothing
        data = _interp_sigma(r, data, Float64.(collect(depths)))
    end
    apply_mask && _apply_mask!(data, r.mask)
    return (data=data, lon=r.lon, lat=r.lat, time=_time_values(r.ds))
end

"""
    velocities(r::CROCOReader; depths=nothing, apply_mask=true)

Horizontal velocities [m/s] averaged to rho-points.
Returns `(u=..., v=..., lon=..., lat=..., time=...)`.
"""
function velocities(r::CROCOReader; depths=nothing, apply_mask::Bool=true)
    u = _u_to_rho(r)
    v = _v_to_rho(r)
    if depths !== nothing
        d = Float64.(collect(depths))
        u = _interp_sigma(r, u, d)
        v = _interp_sigma(r, v, d)
    end
    if apply_mask
        _apply_mask!(u, r.mask)
        _apply_mask!(v, r.mask)
    end
    return (u=u, v=v, lon=r.lon, lat=r.lat, time=_time_values(r.ds))
end

"""
    z_levels(r::CROCOReader; time_idx=nothing) → Array

Physical depths [m, negative downward] on sigma levels.
Returns `(ny, nx, N, nt)` or `(ny, nx, N)` if `time_idx` given.
"""
function z_levels(r::CROCOReader; time_idx::Union{Int,Nothing}=nothing)
    if time_idx !== nothing
        zeta_raw = Float64.(Array(r.ds["zeta"]))
        zeta2d = ndims(zeta_raw) == 3 ? zeta_raw[:,:,time_idx] : zeta_raw
        z = sigma_to_z(r.h, zeta2d, r.sc_r, r.Cs_r, r.hc; vtransform=r.vtransform)
        return z
    else
        zeta = Float64.(Array(r.ds["zeta"]))
        if ndims(zeta) == 2
            zeta = reshape(zeta, size(zeta,1), size(zeta,2), 1)
        end
        return sigma_to_z(r.h, zeta, r.sc_r, r.Cs_r, r.hc; vtransform=r.vtransform)
    end
end

# ── Staggered grid averaging ──────────────────────────────────────

function _u_to_rho(r::CROCOReader)
    u_raw = Float64.(Array(r.ds["u"]))
    dims = dimnames(r.ds["u"])
    xi_idx = findfirst(d -> d in ("xi_u",), dims)
    if xi_idx !== nothing
        n_xi = size(u_raw, xi_idx)
        slices_lo = ntuple(d -> d == xi_idx ? (1:n_xi-1) : Colon(), ndims(u_raw))
        slices_hi = ntuple(d -> d == xi_idx ? (2:n_xi) : Colon(), ndims(u_raw))
        interior = 0.5 .* (u_raw[slices_lo...] .+ u_raw[slices_hi...])
        edge_lo = selectdim(u_raw, xi_idx, 1:1) |> collect
        edge_hi = selectdim(u_raw, xi_idx, n_xi:n_xi) |> collect
        return cat(edge_lo, interior, edge_hi; dims=xi_idx)
    end
    return u_raw
end

function _v_to_rho(r::CROCOReader)
    v_raw = Float64.(Array(r.ds["v"]))
    dims = dimnames(r.ds["v"])
    eta_idx = findfirst(d -> d in ("eta_v",), dims)
    if eta_idx !== nothing
        n_eta = size(v_raw, eta_idx)
        slices_lo = ntuple(d -> d == eta_idx ? (1:n_eta-1) : Colon(), ndims(v_raw))
        slices_hi = ntuple(d -> d == eta_idx ? (2:n_eta) : Colon(), ndims(v_raw))
        interior = 0.5 .* (v_raw[slices_lo...] .+ v_raw[slices_hi...])
        edge_lo = selectdim(v_raw, eta_idx, 1:1) |> collect
        edge_hi = selectdim(v_raw, eta_idx, n_eta:n_eta) |> collect
        return cat(edge_lo, interior, edge_hi; dims=eta_idx)
    end
    return v_raw
end

function _interp_sigma(r::CROCOReader, var::AbstractArray, depths::Vector{Float64})
    zeta = Float64.(Array(r.ds["zeta"]))
    if ndims(zeta) == 2
        zeta = reshape(zeta, size(zeta,1), size(zeta,2), 1)
    end
    z = sigma_to_z(r.h, zeta, r.sc_r, r.Cs_r, r.hc; vtransform=r.vtransform)
    return interp_z(var, z, depths)
end

# ── Context manager / show ────────────────────────────────────────

function Base.close(r::CROCOReader)
    close(r.ds)
    r.grid !== r.ds && close(r.grid)
end

function Base.show(io::IO, r::CROCOReader)
    ny, nx = size(r.h)
    N = length(r.sc_r)
    nt = _get_dim(r.ds, ("time", "ocean_time"), 0)
    print(io, "CROCOReader(nt=$nt, N=$N, ny=$ny, nx=$nx, Vtransform=$(r.vtransform), hc=$(r.hc) m)")
end
