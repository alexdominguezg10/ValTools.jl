"""
    CloudDriftLoader

Loader for CloudDrift-format ragged-array Lagrangian drifter datasets
(GLAD, LASER, and similar CF contiguous-ragged-array NetCDF files: a flat
`obs` dimension holding all trajectories' observations back-to-back, a
`rowsize` (a.k.a. `row_size`/`count`) variable of length `traj` giving each
trajectory's observation count, and one `id` value per trajectory).

Unpacks the ragged array into the same tidy `(float_id, time, lon, lat, ...)`
DataFrame shape `RAFOSLoader` produces, so `colocate_model_trajectory` and
other colocation functions work on it unchanged.
"""
mutable struct CloudDriftLoader
    df::DataFrame
end

const _CD_ROWSIZE_VARS = ("rowsize", "row_size", "count")
const _CD_ID_VARS = ("id", "ID", "drifter_id", "WMO")
const _CD_LON_VARS = ("longitude", "lon", "LONGITUDE")
const _CD_LAT_VARS = ("latitude", "lat", "LATITUDE")
const _CD_TIME_VARS = ("time", "TIME")
const _CD_U_VARS = ("u", "U", "ve")
const _CD_V_VARS = ("v", "V", "vn")

function CloudDriftLoader(path::AbstractString;
                          bbox::Union{NTuple{4,Real}, Nothing}=nothing,
                          date_range::Union{Tuple{String,String}, Nothing}=nothing)
    ds = _obs_open_nc(path)

    rowsize = _obs_read_var_f64(ds, _CD_ROWSIZE_VARS)
    rowsize === nothing && error("CloudDriftLoader: no rowsize/row_size/count variable found -- " *
                                 "is this a CF contiguous ragged array file?")
    rowsize_int = Int.(round.(rowsize))

    traj_id = nothing
    for name in _CD_ID_VARS
        if haskey(ds, name)
            traj_id = Array(ds[name])
            break
        end
    end
    traj_id === nothing && (traj_id = 1:length(rowsize_int))

    lon = _obs_read_var_f64(ds, _CD_LON_VARS)
    lat = _obs_read_var_f64(ds, _CD_LAT_VARS)
    time = _obs_read_time(ds, _CD_TIME_VARS)
    u = _obs_read_var_f64(ds, _CD_U_VARS)
    v = _obs_read_var_f64(ds, _CD_V_VARS)
    close(ds)

    n_obs = length(lon)
    sum(rowsize_int) == n_obs || error("CloudDriftLoader: sum(rowsize)=$(sum(rowsize_int)) " *
                                        "does not match obs dimension length=$n_obs -- " *
                                        "malformed ragged array")

    # The actual ragged-array "unpack" step: trajectories are stored
    # back-to-back in the flat obs dimension, with boundaries given by
    # cumulative rowsize. Expand the per-trajectory id into a
    # per-observation float_id by repeating each id rowsize[i] times.
    float_id = Vector{String}(undef, n_obs)
    pos = 1
    for (i, n) in enumerate(rowsize_int)
        n == 0 && continue
        float_id[pos:pos+n-1] .= string(traj_id[i])
        pos += n
    end

    df = DataFrame(float_id=float_id, time=time, lon=lon, lat=lat)
    u !== nothing && length(u) == n_obs && (df[!, :u] = u)
    v !== nothing && length(v) == n_obs && (df[!, :v] = v)

    if bbox !== nothing
        filter!(row -> isfinite(row.lon) && isfinite(row.lat) &&
                       row.lon >= bbox[1] && row.lon <= bbox[3] &&
                       row.lat >= bbox[2] && row.lat <= bbox[4], df)
    end
    if date_range !== nothing && !isempty(df) && eltype(df.time) <: DateTime
        t0, t1 = DateTime(date_range[1]), DateTime(date_range[2])
        filter!(row -> row.time >= t0 && row.time <= t1, df)
    end

    return CloudDriftLoader(df)
end

function clouddrift_trajectory(r::CloudDriftLoader, id)
    return filter(row -> row.float_id == string(id), r.df)
end

function clouddrift_n_trajectories(r::CloudDriftLoader)
    return length(unique(r.df.float_id))
end

clouddrift_to_dataframe(r::CloudDriftLoader) = copy(r.df)

Base.close(r::CloudDriftLoader) = nothing
