"""
    colocate_model_obs(field, model_lon, model_lat, model_times,
                       obs_df; kwargs...) → DataFrame

Interpolate a model field onto observation (lon, lat, time) points.

# Arguments
- `field`: model data `(ny, nx, nt)` or `(ny, nx)`
- `model_lon`: 1-D `(nx,)` or 2-D `(ny, nx)` longitudes
- `model_lat`: 1-D `(ny,)` or 2-D `(ny, nx)` latitudes
- `model_times`: vector of DateTime or `nothing`
- `obs_df`: DataFrame with lon, lat, time columns

# Keyword Arguments
- `lon_col=:lon`, `lat_col=:lat`, `time_col=:time`: column names
- `max_dt_hours=12.0`: max time difference for matching
- `max_dist_km=50.0`: max spatial distance
- `method=:bilinear`: `:bilinear` or `:nearest`
- `out_col=:model_value`: output column name

# Returns
Copy of `obs_df` with the interpolated model values appended.
"""
function colocate_model_obs(field::AbstractArray{<:Real},
                            model_lon::AbstractArray{<:Real},
                            model_lat::AbstractArray{<:Real},
                            model_times::Union{AbstractVector, Nothing},
                            obs_df::DataFrame;
                            lon_col::Symbol=:lon,
                            lat_col::Symbol=:lat,
                            time_col::Symbol=:time,
                            max_dt_hours::Real=12.0,
                            max_dist_km::Real=50.0,
                            method::Symbol=:bilinear,
                            out_col::Symbol=:model_value)
    nrow(obs_df) == 0 && error("obs_df is empty.")

    result = copy(obs_df)
    result[!, out_col] = fill(NaN, nrow(result))

    obs_lons = Float64.(obs_df[!, lon_col])
    obs_lats = Float64.(obs_df[!, lat_col])

    regular = ndims(model_lon) == 1 && ndims(model_lat) == 1

    if model_times !== nothing
        obs_times = obs_df[!, time_col]
        time_idx, time_valid = _nearest_time_idx(model_times, obs_times, max_dt_hours)

        for tidx in unique(time_idx[time_valid])
            obs_mask = (time_idx .== tidx) .& time_valid
            f2d = ndims(field) == 3 ? field[:, :, tidx] : field
            vals = _spatial_interp(f2d, model_lon, model_lat, regular,
                                   obs_lons[obs_mask], obs_lats[obs_mask],
                                   max_dist_km, method)
            result[obs_mask, out_col] = vals
        end
    else
        f2d = ndims(field) >= 3 ? field[:, :, 1] : field
        vals = _spatial_interp(f2d, model_lon, model_lat, regular,
                               obs_lons, obs_lats, max_dist_km, method)
        result[!, out_col] = vals
    end

    return result
end


"""
    colocate_model_grid(field, model_lon, model_lat,
                        target_lon, target_lat; method=:bilinear)

Regrid a 2-D model field onto a regular target grid.

Returns `(out, target_lon, target_lat)` where `out` is `(nlat, nlon)`.
"""
function colocate_model_grid(field::AbstractMatrix{<:Real},
                             model_lon::AbstractArray{<:Real},
                             model_lat::AbstractArray{<:Real},
                             target_lon::AbstractVector{<:Real},
                             target_lat::AbstractVector{<:Real};
                             method::Symbol=:bilinear)
    regular = ndims(model_lon) == 1 && ndims(model_lat) == 1

    nlat, nlon = length(target_lat), length(target_lon)
    obs_lons = Float64.(repeat(target_lon, nlat))
    obs_lats = Float64.(repeat(target_lat; inner=nlon))

    # Actually: meshgrid order
    obs_lons2 = vec([tlo for tla in target_lat, tlo in target_lon])
    obs_lats2 = vec([tla for tla in target_lat, tlo in target_lon])

    vals = _spatial_interp(field, model_lon, model_lat, regular,
                           obs_lons2, obs_lats2, Inf, method)
    out = reshape(vals, nlat, nlon)
    return (data=out, lon=Float64.(target_lon), lat=Float64.(target_lat))
end


"""
    colocate_model_trajectory(field, model_lon, model_lat, model_times,
                              traj_df; kwargs...) → DataFrame

Follow a Lagrangian trajectory through a model field.
Works for RAFOS floats, GDP drifters, Argo profiles.
"""
function colocate_model_trajectory(field::AbstractArray{<:Real},
                                   model_lon::AbstractArray{<:Real},
                                   model_lat::AbstractArray{<:Real},
                                   model_times::Union{AbstractVector, Nothing},
                                   traj_df::DataFrame;
                                   lon_col::Symbol=:lon,
                                   lat_col::Symbol=:lat,
                                   time_col::Symbol=:time,
                                   max_dt_hours::Real=12.0,
                                   max_dist_km::Real=50.0,
                                   method::Symbol=:bilinear,
                                   out_col::Symbol=:model_value)
    nrow(traj_df) == 0 && error("traj_df is empty.")

    result = copy(traj_df)
    result[!, out_col] = fill(NaN, nrow(result))

    obs_lons = Float64.(traj_df[!, lon_col])
    obs_lats = Float64.(traj_df[!, lat_col])
    regular = ndims(model_lon) == 1 && ndims(model_lat) == 1

    if model_times !== nothing
        obs_times = traj_df[!, time_col]
        time_idx, time_valid = _nearest_time_idx(model_times, obs_times, max_dt_hours)

        for tidx in unique(time_idx[time_valid])
            obs_mask = (time_idx .== tidx) .& time_valid
            f2d = ndims(field) >= 3 ? field[:, :, tidx] : field
            vals = _spatial_interp(f2d, model_lon, model_lat, regular,
                                   obs_lons[obs_mask], obs_lats[obs_mask],
                                   max_dist_km, method)
            result[obs_mask, out_col] = vals
        end
    else
        f2d = ndims(field) >= 3 ? field[:, :, 1] : field
        vals = _spatial_interp(f2d, model_lon, model_lat, regular,
                               obs_lons, obs_lats, max_dist_km, method)
        result[!, out_col] = vals
    end

    return result
end


"""
    colocate_model_mooring(field, model_lon, model_lat, model_times,
                           lon, lat; kwargs...)

Extract model time series at a fixed mooring location.

Returns `(data, times)` where `data` is `(nt,)`.
"""
function colocate_model_mooring(field::AbstractArray{<:Real},
                                model_lon::AbstractArray{<:Real},
                                model_lat::AbstractArray{<:Real},
                                model_times::Union{AbstractVector, Nothing},
                                lon::Real, lat::Real;
                                max_dist_km::Real=25.0,
                                method::Symbol=:bilinear)
    regular = ndims(model_lon) == 1 && ndims(model_lat) == 1
    olon = Float64[lon]
    olat = Float64[lat]

    if ndims(field) == 3
        nt = size(field, 3)
        out = Vector{Float64}(undef, nt)
        for it in 1:nt
            v = _spatial_interp(field[:, :, it], model_lon, model_lat, regular,
                                olon, olat, max_dist_km, method)
            out[it] = v[1]
        end
        return (data=out, time=model_times)
    else
        v = _spatial_interp(field, model_lon, model_lat, regular,
                            olon, olat, max_dist_km, method)
        return (data=v, time=model_times)
    end
end


"""
    colocate_model_section(u_field, v_field, model_lon, model_lat, model_times,
                           lon_section, lat_section; angle_deg=0.0, kwargs...)

Extract a model velocity cross-section along a transect.

Returns `(u, v, v_along, lon_section, lat_section, times)`.
"""
function colocate_model_section(u_field::AbstractArray{<:Real,3},
                                v_field::AbstractArray{<:Real,3},
                                model_lon::AbstractArray{<:Real},
                                model_lat::AbstractArray{<:Real},
                                model_times::Union{AbstractVector, Nothing},
                                lon_section::AbstractVector{<:Real},
                                lat_section::AbstractVector{<:Real};
                                angle_deg::Real=0.0,
                                max_dist_km::Real=25.0,
                                method::Symbol=:bilinear)
    n_pts = length(lon_section)
    nt = size(u_field, 3)

    u_arr = Matrix{Float64}(undef, nt, n_pts)
    v_arr = Matrix{Float64}(undef, nt, n_pts)

    for ip in 1:n_pts
        ur = colocate_model_mooring(u_field, model_lon, model_lat, model_times,
                                    lon_section[ip], lat_section[ip];
                                    max_dist_km=max_dist_km, method=method)
        vr = colocate_model_mooring(v_field, model_lon, model_lat, model_times,
                                    lon_section[ip], lat_section[ip];
                                    max_dist_km=max_dist_km, method=method)
        u_arr[:, ip] = ur.data
        v_arr[:, ip] = vr.data
    end

    theta = deg2rad(angle_deg)
    v_along = u_arr .* sin(theta) .+ v_arr .* cos(theta)

    return (u=u_arr, v=v_arr, v_along=v_along,
            lon=Float64.(lon_section), lat=Float64.(lat_section),
            time=model_times)
end


# ══════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════

function _nearest_time_idx(model_times::AbstractVector,
                           obs_times::AbstractVector,
                           max_dt_hours::Real)
    mt_ms = [Dates.value(Dates.Millisecond(t - model_times[1])) for t in model_times]
    ot_ms = [Dates.value(Dates.Millisecond(t - model_times[1])) for t in obs_times]
    n_obs = length(ot_ms)
    n_mod = length(mt_ms)

    time_idx = Vector{Int}(undef, n_obs)
    time_valid = Vector{Bool}(undef, n_obs)
    max_dt_ms = max_dt_hours * 3600 * 1000

    for i in 1:n_obs
        best_j = 1
        best_dt = abs(ot_ms[i] - mt_ms[1])
        for j in 2:n_mod
            dt = abs(ot_ms[i] - mt_ms[j])
            if dt < best_dt
                best_dt = dt
                best_j = j
            end
        end
        time_idx[i] = best_j
        time_valid[i] = best_dt <= max_dt_ms
    end

    return time_idx, time_valid
end

function _spatial_interp(field::AbstractMatrix{<:Real},
                         mlon::AbstractArray{<:Real},
                         mlat::AbstractArray{<:Real},
                         regular::Bool,
                         obs_lon::AbstractVector{<:Real},
                         obs_lat::AbstractVector{<:Real},
                         max_dist_km::Real,
                         method::Symbol)
    if regular
        return _interp_regular(field, vec(mlon), vec(mlat), obs_lon, obs_lat, method)
    else
        return _interp_curvilinear(field, mlon, mlat, obs_lon, obs_lat,
                                   max_dist_km, method)
    end
end

function _interp_regular(field::AbstractMatrix, mlon::AbstractVector,
                         mlat::AbstractVector,
                         obs_lon::AbstractVector, obs_lat::AbstractVector,
                         method::Symbol)
    ny, nx = size(field)
    lon_sorted = copy(Float64.(mlon))
    lat_sorted = copy(Float64.(mlat))
    f = Float64.(field)

    if lat_sorted[end] < lat_sorted[1]
        reverse!(lat_sorted)
        f = f[end:-1:1, :]
    end

    n = length(obs_lon)
    out = Vector{Float64}(undef, n)

    if method == :nearest
        for i in 1:n
            jx = _nearest_idx(lon_sorted, obs_lon[i])
            jy = _nearest_idx(lat_sorted, obs_lat[i])
            if jx < 1 || jx > nx || jy < 1 || jy > ny
                out[i] = NaN
            else
                out[i] = f[jy, jx]
            end
        end
    else
        itp = interpolate((lat_sorted, lon_sorted), f, Gridded(Linear()))
        eitp = extrapolate(itp, NaN)
        for i in 1:n
            out[i] = eitp(obs_lat[i], obs_lon[i])
        end
    end

    return out
end

function _nearest_idx(sorted_arr::AbstractVector, val::Real)
    n = length(sorted_arr)
    idx = searchsortedlast(sorted_arr, val)
    if idx < 1
        return 1
    elseif idx >= n
        return n
    else
        return abs(val - sorted_arr[idx]) <= abs(val - sorted_arr[idx+1]) ? idx : idx + 1
    end
end

function _interp_curvilinear(field::AbstractMatrix, mlon2d::AbstractArray,
                             mlat2d::AbstractArray,
                             obs_lon::AbstractVector, obs_lat::AbstractVector,
                             max_dist_km::Real, method::Symbol)
    ny, nx = size(mlon2d)
    cos_lat = cosd(mean(mlat2d))
    cos_lat = max(cos_lat, 0.01)

    pts_x = vec(Float64.(mlon2d)) .* cos_lat
    pts_y = vec(Float64.(mlat2d))
    tree_data = hcat(pts_x, pts_y)'
    tree = KDTree(tree_data)

    n = length(obs_lon)
    obs_x = obs_lon .* cos_lat
    obs_y = Float64.(obs_lat)

    k = method == :nearest ? 1 : 4
    out = Vector{Float64}(undef, n)
    flat_field = vec(Float64.(field))

    for i in 1:n
        query_pt = [obs_x[i], obs_y[i]]
        idxs, dists = knn(tree, query_pt, k)

        if method == :nearest
            dist_km = dists[1] * 111.0 / cos_lat
            out[i] = dist_km <= max_dist_km ? flat_field[idxs[1]] : NaN
        else
            eps = 1e-10
            w = 1.0 ./ (dists .+ eps) .^ 2
            vals = flat_field[idxs]
            nan_mask = isnan.(vals)
            w[nan_mask] .= 0.0
            w_sum = sum(w)
            if w_sum == 0.0
                out[i] = NaN
            else
                out[i] = sum(w .* vals) / w_sum
            end
            dist_km_min = dists[1] * 111.0
            if dist_km_min > max_dist_km
                out[i] = NaN
            end
        end
    end

    return out
end
