"""
    DUACSLoader

Loader for CMEMS DUACS L4 altimetry (ADT, SLA, geostrophic velocities).
"""
mutable struct DUACSLoader
    ds::NCDataset
    lon::Vector{Float64}
    lat::Vector{Float64}
end

function DUACSLoader(files::AbstractString;
                     bbox::Union{NTuple{4,Real}, Nothing}=nothing)
    ds = _obs_open_nc(files)
    lon = _obs_read_var_f64(ds, ("longitude", "lon"))
    lat = _obs_read_var_f64(ds, ("latitude", "lat"))
    return DUACSLoader(ds, vec(lon), vec(lat))
end

function duacs_ssh(r::DUACSLoader)
    data = _obs_read_var_f64(r.ds, ("adt", "ssh", "zos", "sla"))
    time = _obs_read_time(r.ds, ("time",))
    return (data=data, lon=r.lon, lat=r.lat, time=time)
end

function duacs_sla(r::DUACSLoader)
    data = _obs_read_var_f64(r.ds, ("sla", "sea_level_anomaly"))
    time = _obs_read_time(r.ds, ("time",))
    return (data=data, lon=r.lon, lat=r.lat, time=time)
end

function duacs_geostrophic_velocity(r::DUACSLoader)
    u = _obs_read_var_f64(r.ds, ("ugos", "ugosa", "u"))
    v = _obs_read_var_f64(r.ds, ("vgos", "vgosa", "v"))
    time = _obs_read_time(r.ds, ("time",))
    return (u=u, v=v, lon=r.lon, lat=r.lat, time=time)
end

Base.close(r::DUACSLoader) = close(r.ds)
