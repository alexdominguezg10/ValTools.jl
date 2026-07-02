"""
    GHRSSTLoader

Loader for GHRSST MUR L4 SST data (daily, 1 km).
"""
mutable struct GHRSSTLoader
    ds::NCDataset
    lon::Vector{Float64}
    lat::Vector{Float64}
end

function GHRSSTLoader(files::AbstractString;
                      bbox::Union{NTuple{4,Real}, Nothing}=nothing)
    ds = _obs_open_nc(files)
    lon = _obs_read_var_f64(ds, ("lon", "longitude"))
    lat = _obs_read_var_f64(ds, ("lat", "latitude"))
    return GHRSSTLoader(ds, vec(lon), vec(lat))
end

function ghrsst_sst(r::GHRSSTLoader; celsius::Bool=true)
    data = _obs_read_var_f64(r.ds, ("analysed_sst", "sst", "SST"))
    data === nothing && return nothing
    time = _obs_read_time(r.ds, ("time",))

    if celsius && any(x -> isfinite(x) && x > 100, data)
        data = data .- 273.15
    end

    return (data=data, lon=r.lon, lat=r.lat, time=time)
end

Base.close(r::GHRSSTLoader) = close(r.ds)
