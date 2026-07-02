const _NEMO_SSH_VARS   = ("zos", "sossheig", "ssh", "SSH")
const _NEMO_TEMP_VARS  = ("thetao", "votemper", "temp", "TEMP")
const _NEMO_SAL_VARS   = ("so", "vosaline", "salt", "SALT")
const _NEMO_U_VARS     = ("uo", "vozocrtx", "u", "U")
const _NEMO_V_VARS     = ("vo", "vomecrty", "v", "V")
const _NEMO_DEPTH_DIMS = ("depth", "deptht", "depthw", "lev", "z")
const _NEMO_LON_VARS   = ("lon", "longitude", "nav_lon", "glamt")
const _NEMO_LAT_VARS   = ("lat", "latitude", "nav_lat", "gphit")
const _NEMO_MASK_VARS  = ("mask", "tmask", "mask_rho")
const _NEMO_TIME_VARS  = ("time_counter", "time", "TIME")

"""
    NEMOReader

Reader for NEMO model output files (z-coordinate).

# Fields
- `ds`: `NCDataset` handle
- `lon`: longitudes
- `lat`: latitudes
- `mask_name`: name of the mask variable or `nothing`
"""
mutable struct NEMOReader
    ds::NCDataset
    lon::Union{Array{Float64}, Nothing}
    lat::Union{Array{Float64}, Nothing}
    mask_name::Union{String, Nothing}
    depth_dim::Union{String, Nothing}
end

function NEMOReader(files::AbstractString; decode_times::Bool=true)
    ds = _open_nc(files)

    lon_name = _find_var(ds, _NEMO_LON_VARS)
    lat_name = _find_var(ds, _NEMO_LAT_VARS)
    lon_arr = lon_name !== nothing ? Float64.(Array(ds[lon_name])) : nothing
    lat_arr = lat_name !== nothing ? Float64.(Array(ds[lat_name])) : nothing

    mask_name = _find_var(ds, _NEMO_MASK_VARS)
    depth_dim = _find_var(ds, _NEMO_DEPTH_DIMS)

    return NEMOReader(ds, lon_arr, lat_arr, mask_name, depth_dim)
end

function _nemo_time(r::NEMOReader)
    tn = _find_var(r.ds, _NEMO_TIME_VARS)
    tn === nothing && error("No time coordinate found.")
    return Array(r.ds[tn])
end

function _nemo_depths(r::NEMOReader)
    r.depth_dim === nothing && return nothing
    d = Float64.(Array(r.ds[r.depth_dim]))
    return -abs.(d)
end

function _nemo_mask!(data::AbstractArray, r::NEMOReader)
    r.mask_name === nothing && return data
    mask = Float64.(Array(r.ds[r.mask_name]))
    for idx in CartesianIndices(data)
        mi = ntuple(d -> d <= ndims(mask) ? idx[d] : 1, ndims(mask))
        ci = CartesianIndex(mi)
        if checkbounds(Bool, mask, ci) && mask[ci] <= 0
            data[idx] = NaN
        end
    end
    return data
end

# ── Surface fields ─────────────────────────────────────────────────

function ssh(r::NEMOReader; apply_mask::Bool=true)
    vn = _find_var(r.ds, _NEMO_SSH_VARS)
    vn === nothing && error("SSH not found. Tried: $_NEMO_SSH_VARS")
    data = Float64.(Array(r.ds[vn]))
    apply_mask && _nemo_mask!(data, r)
    return (data=data, lon=r.lon, lat=r.lat, time=_nemo_time(r))
end

function sst(r::NEMOReader; apply_mask::Bool=true)
    data = _nemo_surface(r, _NEMO_TEMP_VARS, "SST")
    apply_mask && _nemo_mask!(data, r)
    return (data=data, lon=r.lon, lat=r.lat, time=_nemo_time(r))
end

function sss(r::NEMOReader; apply_mask::Bool=true)
    data = _nemo_surface(r, _NEMO_SAL_VARS, "SSS")
    apply_mask && _nemo_mask!(data, r)
    return (data=data, lon=r.lon, lat=r.lat, time=_nemo_time(r))
end

function _nemo_surface(r::NEMOReader, candidates, label)
    vn = _find_var(r.ds, candidates)
    vn === nothing && error("$label not found. Tried: $candidates")
    var = r.ds[vn]
    dims = dimnames(var)
    d_idx = findfirst(d -> d in _NEMO_DEPTH_DIMS, dims)
    all_data = Float64.(Array(var))
    if d_idx !== nothing
        return selectdim(all_data, d_idx, 1) |> collect
    end
    return all_data
end

# ── 3-D fields ─────────────────────────────────────────────────────

function temperature(r::NEMOReader; depths=nothing, apply_mask::Bool=true)
    vn = _find_var(r.ds, _NEMO_TEMP_VARS)
    vn === nothing && error("Temperature not found. Tried: $_NEMO_TEMP_VARS")
    data = Float64.(Array(r.ds[vn]))
    if depths !== nothing
        data = _nemo_interp_z(r, data, vn, Float64.(collect(depths)))
    end
    apply_mask && _nemo_mask!(data, r)
    return (data=data, lon=r.lon, lat=r.lat, time=_nemo_time(r))
end

function salinity(r::NEMOReader; depths=nothing, apply_mask::Bool=true)
    vn = _find_var(r.ds, _NEMO_SAL_VARS)
    vn === nothing && error("Salinity not found. Tried: $_NEMO_SAL_VARS")
    data = Float64.(Array(r.ds[vn]))
    if depths !== nothing
        data = _nemo_interp_z(r, data, vn, Float64.(collect(depths)))
    end
    apply_mask && _nemo_mask!(data, r)
    return (data=data, lon=r.lon, lat=r.lat, time=_nemo_time(r))
end

function velocities(r::NEMOReader; depths=nothing, apply_mask::Bool=true)
    un = _find_var(r.ds, _NEMO_U_VARS)
    vn = _find_var(r.ds, _NEMO_V_VARS)
    (un === nothing || vn === nothing) && error("Velocity components not found.")
    u = Float64.(Array(r.ds[un]))
    v = Float64.(Array(r.ds[vn]))
    if depths !== nothing
        d = Float64.(collect(depths))
        u = _nemo_interp_z(r, u, un, d)
        v = _nemo_interp_z(r, v, vn, d)
    end
    if apply_mask
        _nemo_mask!(u, r)
        _nemo_mask!(v, r)
    end
    return (u=u, v=v, lon=r.lon, lat=r.lat, time=_nemo_time(r))
end

function _nemo_interp_z(r::NEMOReader, data::AbstractArray, varname::String,
                        target_depths::Vector{Float64})
    dims = dimnames(r.ds[varname])
    d_idx = findfirst(d -> d in _NEMO_DEPTH_DIMS, dims)
    d_idx === nothing && return data

    z_vals = _nemo_depths(r)
    z_vals === nothing && return data

    nz_target = length(target_depths)
    sz = collect(size(data))
    sz[d_idx] = nz_target
    out = fill(NaN, sz...)

    nz_src = length(z_vals)
    for iz in 1:nz_target
        zd = target_depths[iz]
        k0 = 1
        for k in 1:nz_src-1
            if z_vals[k] >= zd
                k0 = k
            end
        end
        k1 = min(k0 + 1, nz_src)

        z0, z1 = z_vals[k0], z_vals[k1]
        dz = z1 - z0
        w = abs(dz) > 1e-6 ? clamp((zd - z0) / dz, 0.0, 1.0) : 0.5

        slice0 = ntuple(d -> d == d_idx ? k0 : Colon(), ndims(data))
        slice1 = ntuple(d -> d == d_idx ? k1 : Colon(), ndims(data))
        slice_out = ntuple(d -> d == d_idx ? iz : Colon(), ndims(data))

        v0 = data[slice0...]
        v1 = data[slice1...]
        out[slice_out...] .= v0 .+ w .* (v1 .- v0)
    end

    return out
end

# ── Context manager / show ────────────────────────────────────────

function Base.close(r::NEMOReader)
    close(r.ds)
end

function Base.show(io::IO, r::NEMOReader)
    nt = _get_dim(r.ds, _NEMO_TIME_VARS, 0)
    print(io, "NEMOReader(nt=$nt)")
end
