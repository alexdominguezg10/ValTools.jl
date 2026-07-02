"""
    ThermistorLoader

Loader for moored thermistor chain data.
"""
const _RHO_CP = 1025.0 * 3985.0  # ρ_ref × c_p [J/(m³·K)]

mutable struct ThermistorLoader
    ds::NCDataset
    site::String
    lon::Union{Float64, Nothing}
    lat::Union{Float64, Nothing}
end

function ThermistorLoader(files::AbstractString;
                          site::String="thermistor",
                          lon::Union{Real, Nothing}=nothing,
                          lat::Union{Real, Nothing}=nothing)
    ds = _obs_open_nc(files)
    lon_val = lon !== nothing ? Float64(lon) : _obs_try_scalar(ds, ("longitude", "lon"))
    lat_val = lat !== nothing ? Float64(lat) : _obs_try_scalar(ds, ("latitude", "lat"))
    return ThermistorLoader(ds, site, lon_val, lat_val)
end

function thermistor_temperature(r::ThermistorLoader)
    data = _obs_read_var_f64(r.ds, ("temperature", "TEMP", "temp", "T"))
    time = _obs_read_time(r.ds, ("time", "TIME"))
    depths = _obs_read_var_f64(r.ds, ("depth", "DEPTH", "z"))
    return (data=data, time=time, depths=depths, site=r.site)
end

function thermistor_isotherm_depth(r::ThermistorLoader, T_iso::Real)
    result = thermistor_temperature(r)
    result.data === nothing && return nothing
    result.depths === nothing && return nothing

    T = result.data
    d = Float64.(vec(result.depths))
    nd = ndims(T)

    if nd == 1
        return nothing
    end

    nt = size(T, 1)
    nz = size(T, 2)
    iso_depth = fill(NaN, nt)

    for it in 1:nt
        for iz in 1:nz-1
            T0 = T[it, iz]
            T1 = T[it, iz+1]
            (!isfinite(T0) || !isfinite(T1)) && continue
            if (T0 - T_iso) * (T1 - T_iso) <= 0
                w = abs(T1 - T0) > 1e-10 ? (T_iso - T0) / (T1 - T0) : 0.5
                iso_depth[it] = d[iz] + w * (d[iz+1] - d[iz])
                break
            end
        end
    end

    return (data=iso_depth, time=result.time)
end

function thermistor_heat_content(r::ThermistorLoader;
                                 z1::Real=0.0, z2::Real=300.0)
    result = thermistor_temperature(r)
    result.data === nothing && return nothing
    result.depths === nothing && return nothing

    T = result.data
    d = Float64.(vec(result.depths))
    nt = size(T, 1)
    Q = fill(NaN, nt)

    mask = (d .>= z1) .& (d .<= z2)
    d_sub = d[mask]
    length(d_sub) < 2 && return nothing

    for it in 1:nt
        T_sub = T[it, mask]
        valid = isfinite.(T_sub)
        sum(valid) < 2 && continue
        dv = d_sub[valid]
        tv = T_sub[valid]
        order = sortperm(dv)
        dv, tv = dv[order], tv[order]
        integral = sum((tv[1:end-1] .+ tv[2:end]) ./ 2.0 .* diff(dv))
        Q[it] = _RHO_CP * integral
    end

    return (data=Q, time=result.time)
end

Base.close(r::ThermistorLoader) = close(r.ds)
