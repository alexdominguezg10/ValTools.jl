"""
    GliderLoader

Loader for ocean glider trajectory/profile data (EGO/OceanGliders NetCDF
convention: flat TIME/LATITUDE/LONGITUDE/PRES/TEMP/PSAL arrays along the
glider's dive-climb path, optionally with a PROFILE_NUMBER/dive-cycle index
and a platform/glider-name identifier).

Returns a tidy `(time, lon, lat, pressure, temperature, salinity, ...)`
DataFrame, one row per measurement along the glider's path -- same shape
convention as `RAFOSLoader`/`CloudDriftLoader`, so colocation functions work
on it unchanged (a glider file is effectively a single trajectory, unless
`platform_id` is used to distinguish multiple gliders loaded together).
"""
mutable struct GliderLoader
    df::DataFrame
end

const _GLIDER_TIME_VARS = ("TIME", "time", "JULD")
const _GLIDER_LON_VARS = ("LONGITUDE", "longitude", "lon")
const _GLIDER_LAT_VARS = ("LATITUDE", "latitude", "lat")
const _GLIDER_PRES_VARS = ("PRES", "PRES_ADJUSTED", "pressure", "DEPTH")
const _GLIDER_TEMP_VARS = ("TEMP", "TEMP_ADJUSTED", "temperature")
const _GLIDER_PSAL_VARS = ("PSAL", "PSAL_ADJUSTED", "salinity")
const _GLIDER_PROFILE_VARS = ("PROFILE_NUMBER", "profile_index", "dive_number")
const _GLIDER_PLATFORM_VARS = ("PLATFORM_ID", "platform_id", "glider_name", "WMO_IDENTIFIER")

function GliderLoader(path::AbstractString;
                      bbox::Union{NTuple{4,Real}, Nothing}=nothing,
                      date_range::Union{Tuple{String,String}, Nothing}=nothing,
                      pressure_range::Union{Tuple{Real,Real}, Nothing}=nothing)
    ds = _obs_open_nc(path)

    lon = _obs_read_var_f64(ds, _GLIDER_LON_VARS)
    lat = _obs_read_var_f64(ds, _GLIDER_LAT_VARS)
    time = _obs_read_time(ds, _GLIDER_TIME_VARS)
    pres = _obs_read_var_f64(ds, _GLIDER_PRES_VARS)
    temp = _obs_read_var_f64(ds, _GLIDER_TEMP_VARS)
    psal = _obs_read_var_f64(ds, _GLIDER_PSAL_VARS)

    lon === nothing && error("GliderLoader: no LONGITUDE/longitude/lon variable found")
    n = length(lon)

    df = DataFrame(lon=lon, lat=lat)
    df[!, :time] = length(time) == n ? time : (isempty(time) ? fill(DateTime(0), n) : time[1:min(end, n)])

    # lon/lat/pres/temp/psal along a glider path are typically all the same
    # length (one per timestamped measurement), but some EGO files carry a
    # lower-resolution GPS-fix time/position only -- tolerate length
    # mismatches by skipping a variable rather than erroring.
    pres !== nothing && length(pres) == n && (df[!, :pressure] = pres)
    temp !== nothing && length(temp) == n && (df[!, :temperature] = temp)
    psal !== nothing && length(psal) == n && (df[!, :salinity] = psal)

    for name in _GLIDER_PROFILE_VARS
        if haskey(ds, name)
            profile_num = Array(ds[name])
            length(profile_num) == n && (df[!, :profile] = profile_num)
            break
        end
    end

    for name in _GLIDER_PLATFORM_VARS
        if haskey(ds, name)
            val = Array(ds[name])
            platform_id = val isa AbstractArray ? join(string.(vec(val))) : string(val)
            df[!, :platform_id] .= platform_id
            break
        end
    end

    close(ds)

    if bbox !== nothing
        filter!(row -> isfinite(row.lon) && isfinite(row.lat) &&
                       row.lon >= bbox[1] && row.lon <= bbox[3] &&
                       row.lat >= bbox[2] && row.lat <= bbox[4], df)
    end
    if date_range !== nothing && !isempty(df) && eltype(df.time) <: DateTime
        t0, t1 = DateTime(date_range[1]), DateTime(date_range[2])
        filter!(row -> row.time >= t0 && row.time <= t1, df)
    end
    if pressure_range !== nothing && hasproperty(df, :pressure)
        filter!(row -> isfinite(row.pressure) &&
                       row.pressure >= pressure_range[1] &&
                       row.pressure <= pressure_range[2], df)
    end

    return GliderLoader(df)
end

function glider_profiles(r::GliderLoader)
    hasproperty(r.df, :profile) || return DataFrame()
    return combine(groupby(r.df, :profile),
                   :time => first => :start_time,
                   :lon => first => :lon,
                   :lat => first => :lat,
                   nrow => :n_obs)
end

function glider_section(r::GliderLoader; var::Symbol=:temperature)
    cols = intersect(names(r.df), ["time", "lon", "lat", "pressure", string(var)])
    return select(r.df, Symbol.(cols))
end

glider_to_dataframe(r::GliderLoader) = copy(r.df)

Base.close(r::GliderLoader) = nothing
