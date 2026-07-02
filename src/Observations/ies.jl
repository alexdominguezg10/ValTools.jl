"""
    IESLoader

Loader for Inverted Echo Sounder (IES/CPIES) data.
"""
mutable struct IESLoader
    ds::NCDataset
    site::String
    lon::Union{Float64, Nothing}
    lat::Union{Float64, Nothing}
    depth::Union{Float64, Nothing}
    lowpass_days::Float64
end

function IESLoader(files::AbstractString;
                   site::String="IES",
                   lon::Union{Real, Nothing}=nothing,
                   lat::Union{Real, Nothing}=nothing,
                   depth::Union{Real, Nothing}=nothing,
                   lowpass_days::Real=30.0)
    ds = _obs_open_nc(files)
    lon_val = lon !== nothing ? Float64(lon) : _obs_try_scalar(ds, ("longitude", "lon"))
    lat_val = lat !== nothing ? Float64(lat) : _obs_try_scalar(ds, ("latitude", "lat"))
    depth_val = depth !== nothing ? Float64(depth) : _obs_try_scalar(ds, ("depth", "DEPTH"))
    return IESLoader(ds, site, lon_val, lat_val, depth_val, Float64(lowpass_days))
end

function ies_travel_time(r::IESLoader)
    tau = _obs_read_var_f64(r.ds, ("tau", "travel_time", "TRAVEL_TIME", "TAU"))
    time = _obs_read_time(r.ds, ("time", "TIME"))
    return (data=tau, time=time, site=r.site)
end

function ies_ssh_anomaly(r::IESLoader; method::String="linear")
    tau_result = ies_travel_time(r)
    tau = tau_result.data
    tau === nothing && return nothing

    tau_f = Float64.(vec(tau))
    valid = isfinite.(tau_f)

    tau_anom = copy(tau_f)
    if method == "linear"
        idx = findall(valid)
        n = length(idx)
        if n >= 2
            x = Float64.(idx)
            A = hcat(x, ones(n))
            coeffs = A \ tau_f[idx]
            trend = A * coeffs
            tau_anom[idx] .= tau_f[idx] .- trend
        end
    else
        tau_anom[valid] .= tau_f[valid] .- mean(tau_f[valid])
    end

    c_bar = 1500.0
    Z = r.depth !== nothing ? r.depth : 1000.0
    ssh_anom = @. -c_bar^2 * tau_anom / (2.0 * Z)

    return (data=ssh_anom, time=tau_result.time, site=r.site)
end

function ies_to_dataframe(r::IESLoader)
    tau_r = ies_travel_time(r)
    ssh_r = ies_ssh_anomaly(r)
    tau_r.data === nothing && return DataFrame()

    df = DataFrame(
        time = tau_r.time,
        tau = vec(tau_r.data),
        ssh_anomaly = ssh_r !== nothing ? vec(ssh_r.data) : fill(NaN, length(tau_r.time)),
    )
    df[!, :site] .= r.site
    r.lon !== nothing && (df[!, :lon] .= r.lon)
    r.lat !== nothing && (df[!, :lat] .= r.lat)
    return df
end

Base.close(r::IESLoader) = close(r.ds)
