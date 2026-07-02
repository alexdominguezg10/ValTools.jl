"""
    GOFLOWLoader

Loader for GOFLOW SST-derived surface currents (Lenain et al. 2026).
"""
const _GOFLOW_REGIONS = Dict(
    "yucatan"  => (-88.5, 19.5, -84.5, 23.5),
    "campeche" => (-97.5, 16.5, -89.0, 22.5),
    "gom"      => (-98.0, 16.0, -80.0, 31.0),
)

mutable struct GOFLOWLoader
    ds::NCDataset
    lon::Vector{Float64}
    lat::Vector{Float64}
end

function GOFLOWLoader(files::AbstractString;
                      bbox::Union{NTuple{4,Real}, Nothing}=nothing,
                      region::Union{String, Nothing}=nothing)
    ds = _obs_open_nc(files)
    lon = _obs_read_var_f64(ds, ("lon", "longitude"))
    lat = _obs_read_var_f64(ds, ("lat", "latitude"))
    return GOFLOWLoader(ds, vec(lon), vec(lat))
end

function goflow_u(r::GOFLOWLoader)
    data = _obs_read_var_f64(r.ds, ("u", "U", "uo"))
    time = _obs_read_time(r.ds, ("time",))
    return (data=data, lon=r.lon, lat=r.lat, time=time)
end

function goflow_v(r::GOFLOWLoader)
    data = _obs_read_var_f64(r.ds, ("v", "V", "vo"))
    time = _obs_read_time(r.ds, ("time",))
    return (data=data, lon=r.lon, lat=r.lat, time=time)
end

function goflow_speed(r::GOFLOWLoader)
    ur = goflow_u(r)
    vr = goflow_v(r)
    (ur.data === nothing || vr.data === nothing) && return nothing
    spd = hypot.(ur.data, vr.data)
    return (data=spd, lon=r.lon, lat=r.lat, time=ur.time)
end

function goflow_vorticity(r::GOFLOWLoader)
    ur = goflow_u(r)
    vr = goflow_v(r)
    (ur.data === nothing || vr.data === nothing) && return nothing

    u = ur.data
    v = vr.data

    if ndims(u) < 2
        return nothing
    end

    ny, nx = size(u, 1), size(u, 2)
    dx_m = length(r.lon) >= 2 ? abs(r.lon[2] - r.lon[1]) * cosd(mean(r.lat)) * 111e3 : 1.0
    dy_m = length(r.lat) >= 2 ? abs(r.lat[2] - r.lat[1]) * 111e3 : 1.0

    trailing_dims = size(u)[3:end]
    vort = fill(NaN, size(u)...)

    for idx in CartesianIndices(trailing_dims)
        u2d = u[:, :, idx]
        v2d = v[:, :, idx]
        for j in 2:nx-1, i in 2:ny-1
            dvdx = (v2d[i, j+1] - v2d[i, j-1]) / (2 * dx_m)
            dudy = (u2d[i+1, j] - u2d[i-1, j]) / (2 * dy_m)
            vort[i, j, idx] = dvdx - dudy
        end
    end

    return (data=vort, lon=r.lon, lat=r.lat, time=ur.time)
end

Base.close(r::GOFLOWLoader) = close(r.ds)
