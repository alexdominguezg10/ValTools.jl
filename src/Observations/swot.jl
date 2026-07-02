"""
    SWOTLoader

Loader for SWOT L3 SSH data (along-track or swath).
"""
mutable struct SWOTLoader
    ds::NCDataset
end

function SWOTLoader(files::AbstractString)
    ds = _obs_open_nc(files)
    return SWOTLoader(ds)
end

function swot_ssh(r::SWOTLoader)
    data = _obs_read_var_f64(r.ds, ("ssh_karin_2", "ssh_karin", "ssha_karin",
                                     "ssh", "adt", "ssha"))
    lon = _obs_read_var_f64(r.ds, ("longitude", "lon"))
    lat = _obs_read_var_f64(r.ds, ("latitude", "lat"))
    time = _obs_read_time(r.ds, ("time",))
    return (data=data, lon=lon, lat=lat, time=time)
end

function swot_sla(r::SWOTLoader)
    data = _obs_read_var_f64(r.ds, ("ssha_karin", "sla", "ssha"))
    lon = _obs_read_var_f64(r.ds, ("longitude", "lon"))
    lat = _obs_read_var_f64(r.ds, ("latitude", "lat"))
    time = _obs_read_time(r.ds, ("time",))
    return (data=data, lon=lon, lat=lat, time=time)
end

function swot_to_points(r::SWOTLoader)
    result = swot_ssh(r)
    result.data === nothing && return DataFrame()

    lon_flat = vec(Float64.(result.lon))
    lat_flat = vec(Float64.(result.lat))
    data_flat = vec(Float64.(result.data))

    valid = isfinite.(data_flat) .& isfinite.(lon_flat) .& isfinite.(lat_flat)
    return DataFrame(
        lon = lon_flat[valid],
        lat = lat_flat[valid],
        ssh = data_flat[valid],
    )
end

Base.close(r::SWOTLoader) = close(r.ds)
