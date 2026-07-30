"""
    MooringCurrentLoader

Loader for mooring current meter data (ADCP profiles or point meters).
"""
mutable struct MooringCurrentLoader
    ds::NCDataset
    site::String
    lon::Union{Float64, Nothing}
    lat::Union{Float64, Nothing}
end

function MooringCurrentLoader(files::AbstractString;
                              site::String="mooring",
                              lon::Union{Real, Nothing}=nothing,
                              lat::Union{Real, Nothing}=nothing)
    ds = _obs_open_nc(files)
    lon_val = lon !== nothing ? Float64(lon) : _obs_try_scalar(ds, ("longitude", "lon"))
    lat_val = lat !== nothing ? Float64(lat) : _obs_try_scalar(ds, ("latitude", "lat"))
    return MooringCurrentLoader(ds, site, lon_val, lat_val)
end

function mooring_current_profiles(r::MooringCurrentLoader)
    u = _obs_read_var_f64(r.ds, ("u", "U", "UCUR", "eastward_sea_water_velocity"))
    v = _obs_read_var_f64(r.ds, ("v", "V", "VCUR", "northward_sea_water_velocity"))
    time = _obs_read_time(r.ds, ("time", "TIME"))
    depths = _obs_read_var_f64(r.ds, ("depth", "DEPTH", "z"))
    return (u=u, v=v, time=time, depths=depths, site=r.site)
end

function mooring_speed(r::MooringCurrentLoader)
    p = mooring_current_profiles(r)
    p.u === nothing && return nothing
    return (data=hypot.(p.u, p.v), time=p.time, depths=p.depths)
end

function mooring_direction(r::MooringCurrentLoader)
    p = mooring_current_profiles(r)
    p.u === nothing && return nothing
    dir = @. mod(rad2deg(atan(p.u, p.v)), 360.0)
    return (data=dir, time=p.time, depths=p.depths)
end

function mooring_variance_ellipse(r::MooringCurrentLoader; depth_idx::Union{Int,Nothing}=nothing)
    p = mooring_current_profiles(r)
    p.u === nothing && return nothing

    u = depth_idx !== nothing && ndims(p.u) >= 2 ? p.u[:, depth_idx] :
        ndims(p.u) >= 2 ? p.u[:, 1] : vec(p.u)
    v = depth_idx !== nothing && ndims(p.v) >= 2 ? p.v[:, depth_idx] :
        ndims(p.v) >= 2 ? p.v[:, 1] : vec(p.v)

    valid = isfinite.(u) .& isfinite.(v)
    u, v = u[valid], v[valid]

    u_anom = u .- mean(u)
    v_anom = v .- mean(v)
    C = [mean(u_anom .^ 2)       mean(u_anom .* v_anom);
         mean(u_anom .* v_anom)  mean(v_anom .^ 2)]

    tr_val = C[1,1] + C[2,2]
    det_val = C[1,1] * C[2,2] - C[1,2]^2
    disc = sqrt(max((tr_val / 2)^2 - det_val, 0.0))
    l1 = tr_val / 2 + disc
    l2 = tr_val / 2 - disc
    theta = 0.5 * atan(2 * C[1,2], C[1,1] - C[2,2])
    incl = mod(90.0 - rad2deg(theta), 180.0)

    return Dict{String, Float64}(
        "semi_major" => sqrt(max(l1, 0)),
        "semi_minor" => sqrt(max(l2, 0)),
        "inclination" => incl,
        "eccentricity" => l2 > 0 ? l2 / l1 : 0.0,
        "EKE" => 0.5 * (var(u) + var(v)),
        "MKE" => 0.5 * (mean(u)^2 + mean(v)^2),
    )
end

function mooring_progressive_vector(r::MooringCurrentLoader; depth_idx::Union{Int,Nothing}=nothing)
    p = mooring_current_profiles(r)
    p.u === nothing && return DataFrame()

    u = depth_idx !== nothing && ndims(p.u) >= 2 ? p.u[:, depth_idx] :
        ndims(p.u) >= 2 ? p.u[:, 1] : vec(p.u)
    v = depth_idx !== nothing && ndims(p.v) >= 2 ? p.v[:, depth_idx] :
        ndims(p.v) >= 2 ? p.v[:, 1] : vec(p.v)
    time = p.time

    n = min(length(u), length(time))
    dx_km = zeros(n)
    dy_km = zeros(n)

    for i in 2:n
        dt_s = Dates.value(time[i] - time[i-1]) / 1000.0
        dx_km[i] = dx_km[i-1] + u[i] * dt_s / 1000.0
        dy_km[i] = dy_km[i-1] + v[i] * dt_s / 1000.0
    end

    return DataFrame(time=time[1:n], dx_km=dx_km, dy_km=dy_km)
end

"""
    mooring_current_ts(r::MooringCurrentLoader) -> (u=TimeSeriesMatrix, v=TimeSeriesMatrix)

Typed counterpart to [`mooring_current_profiles`](@ref): channels are
depths (or a single `"0"` channel for a point meter with no depth axis),
sharing one `TimeSeriesMatrix`-required time axis. Two separate matrices
are returned (not one struct) since `TimeSeriesMatrix` carries a single
physical quantity and u/v are two.

Errors if no velocity variable is found, if time couldn't be decoded to
`DateTime`, or if a 2-D velocity array's leading dimension doesn't match
the time axis length (the `(n_time, n_depth)` orientation this package
assumes elsewhere is asserted here, not trusted).
"""
function mooring_current_ts(r::MooringCurrentLoader)
    p = mooring_current_profiles(r)
    p.u === nothing && error("mooring_current_ts: no u/v velocity variable found at site $(r.site)")
    time = _require_datetime(p.time, "mooring_current_ts")

    u_name = _first_present_varname(r.ds, ("u", "U", "UCUR", "eastward_sea_water_velocity"))
    v_name = _first_present_varname(r.ds, ("v", "V", "VCUR", "northward_sea_water_velocity"))
    u_unit, u_raw = _var_units(r.ds, u_name, u"m/s")
    v_unit, v_raw = _var_units(r.ds, v_name, u"m/s")

    if ndims(p.u) == 1
        nt = length(p.u)
        u_mat = reshape(Float64.(p.u), nt, 1)
        v_mat = reshape(Float64.(p.v), nt, 1)
        channels = ["0"]
    else
        size(p.u, 1) == length(time) ||
            error("mooring_current_ts: expected (n_time, n_depth) orientation but got " *
                  "size(u)=$(size(p.u)) against $(length(time)) time steps -- axis order looks swapped")
        u_mat = Float64.(p.u)
        v_mat = Float64.(p.v)
        channels = p.depths === nothing ? string.(1:size(u_mat, 2)) : string.(round.(p.depths; digits=1))
    end

    meta = (source="mooring", site=r.site)
    u_ts = Types.TimeSeriesMatrix(time, u_mat .* u_unit, channels, r.site * "_u",
                                  merge(meta, (varname=u_name, raw_units=u_raw)))
    v_ts = Types.TimeSeriesMatrix(time, v_mat .* v_unit, channels, r.site * "_v",
                                  merge(meta, (varname=v_name, raw_units=v_raw)))
    return (u=u_ts, v=v_ts)
end

Base.close(r::MooringCurrentLoader) = close(r.ds)
