const _DEPTH_PER_DBAR = 1.019716

"""
    pressure_to_depth(pressure; latitude=nothing)

Convert sea pressure [dbar] to depth [m, positive down].

Uses UNESCO (Fofonoff & Millard, 1983) when `latitude` is given,
otherwise a constant factor 1.019716.
"""
function pressure_to_depth(pressure::Union{Real, AbstractArray};
                           latitude::Union{Real, Nothing}=nothing)
    p = Float64.(pressure)
    if latitude === nothing
        return p .* _DEPTH_PER_DBAR
    end

    lat_rad = deg2rad(Float64(latitude))
    x = sin(lat_rad)^2
    g = @. 9.780318 * (1.0 + (5.2788e-3 + 2.36e-5 * x) * x) + 1.092e-6 * p
    return @. (((-1.82e-15 * p + 2.279e-10) * p - 2.2512e-5) * p + 9.72659) * p / g
end


"""
    depth_to_pressure(depth; latitude=nothing)

Convert depth [m, positive down] to sea pressure [dbar].
Inverse of `pressure_to_depth` via Newton-Raphson iteration.
"""
function depth_to_pressure(depth::Union{Real, AbstractArray};
                           latitude::Union{Real, Nothing}=nothing)
    d = Float64.(depth)
    if latitude === nothing
        return d ./ _DEPTH_PER_DBAR
    end

    p = d ./ _DEPTH_PER_DBAR
    for _ in 1:10
        f = pressure_to_depth(p; latitude=latitude) .- d
        dp = 1e-4
        deriv = (pressure_to_depth(p .+ dp; latitude=latitude) .-
                 pressure_to_depth(p .- dp; latitude=latitude)) ./ (2 * dp)
        p = p .- f ./ deriv
    end
    return p
end


"""
    MooringKnockdownModel

Empirical mooring knockdown model: Δz(U) = L(1 - cos(arctan(k·U²))).

# Fields
- `k::Float64`: drag/buoyancy coefficient [s²/m²]
- `line_length::Union{Float64, Nothing}`: line length L [m]
"""
mutable struct MooringKnockdownModel
    k::Float64
    line_length::Union{Float64, Nothing}
end

MooringKnockdownModel(; k::Real=0.0, line_length::Union{Real,Nothing}=nothing) =
    MooringKnockdownModel(Float64(k), line_length === nothing ? nothing : Float64(line_length))

function theta(m::MooringKnockdownModel, speed::Union{Real, AbstractArray})
    return atan.(m.k .* Float64.(speed) .^ 2)
end

function delta_z(m::MooringKnockdownModel, speed::Union{Real, AbstractArray})
    s = Float64.(speed)
    if m.k == 0.0 || m.line_length === nothing
        return zero.(s)
    end
    return m.line_length .* (1.0 .- cos.(theta(m, s)))
end

function horizontal_excursion(m::MooringKnockdownModel, speed::Union{Real, AbstractArray})
    s = Float64.(speed)
    if m.k == 0.0 || m.line_length === nothing
        return zero.(s)
    end
    return m.line_length .* sin.(theta(m, s))
end


"""
    fit_knockdown(speed, depth_actual, depth_nominal; line_length=nothing)

Calibrate knockdown model parameters from observations.

Fits Δz(U) = L(1 - cos(arctan(k·U²))) by least squares.
If `line_length` is given, only `k` is fitted; otherwise both.
"""
function fit_knockdown(speed::AbstractVector{<:Real},
                       depth_actual::AbstractVector{<:Real},
                       depth_nominal::Real;
                       line_length::Union{Real, Nothing}=nothing)
    s = Float64.(speed)
    dz = Float64.(depth_actual) .- Float64(depth_nominal)
    valid = isfinite.(s) .& isfinite.(dz)
    s = s[valid]
    dz = dz[valid]

    sum(valid) >= 4 || error("Need >= 4 finite (speed, depth) pairs; got $(sum(valid)).")

    if line_length !== nothing
        L = Float64(line_length)
        k = _fit_k_only(s, dz, L)
        return MooringKnockdownModel(k, L)
    else
        L0 = max(Float64(depth_nominal), 1.0)
        k, L = _fit_k_and_L(s, dz, L0)
        return MooringKnockdownModel(k, L)
    end
end

function _knockdown_model_fn(U, k, L)
    return L .* (1.0 .- cos.(atan.(k .* U .^ 2)))
end

function _fit_k_only(speed, dz, L)
    k0 = _initial_k_guess(speed, dz, L)
    k_best = k0
    best_cost = Inf

    for trial_k in [k0, k0 * 0.1, k0 * 10.0, 1e-3, 1e-2, 1e-1]
        trial_k <= 0 && continue
        k_opt = _simple_optimize_1d(speed, dz, L, trial_k)
        pred = _knockdown_model_fn(speed, k_opt, L)
        cost = sum((pred .- dz) .^ 2)
        if cost < best_cost
            best_cost = cost
            k_best = k_opt
        end
    end
    return max(k_best, 0.0)
end

function _fit_k_and_L(speed, dz, L0)
    k0 = _initial_k_guess(speed, dz, L0)
    k_best, L_best = k0, L0
    best_cost = Inf

    for (tk, tL) in [(k0, L0), (k0*0.1, L0*0.5), (k0*10, L0*2), (1e-3, L0)]
        tk <= 0 && continue
        tL <= 0 && continue
        k_opt, L_opt = _simple_optimize_2d(speed, dz, tk, tL)
        pred = _knockdown_model_fn(speed, k_opt, L_opt)
        cost = sum((pred .- dz) .^ 2)
        if cost < best_cost
            best_cost = cost
            k_best, L_best = k_opt, L_opt
        end
    end
    return max(k_best, 0.0), max(L_best, 0.0)
end

function _initial_k_guess(speed, dz, L)
    theta_est = acos.(clamp.(1.0 .- dz ./ L, -1.0, 1.0))
    k_est = tan.(theta_est) ./ (speed .^ 2)
    speed_q = length(speed) > 0 ? quantile(speed, 0.75) : 0.0
    valid = isfinite.(k_est) .& (speed .>= speed_q) .& (speed .> 0)
    sum(valid) == 0 && return 1e-3
    return Float64(median(k_est[valid]))
end

function _simple_optimize_1d(speed, dz, L, k0; maxiter=1000, lr=1e-6)
    k = max(k0, 1e-10)
    for _ in 1:maxiter
        pred = _knockdown_model_fn(speed, k, L)
        residual = pred .- dz
        dk = 1e-8
        pred_dk = _knockdown_model_fn(speed, k + dk, L)
        grad = 2.0 * sum(residual .* (pred_dk .- pred) ./ dk)
        k = max(k - lr * grad, 1e-15)
    end
    return k
end

function _simple_optimize_2d(speed, dz, k0, L0; maxiter=1000, lr_k=1e-6, lr_L=1e-6)
    k = max(k0, 1e-10)
    L = max(L0, 1e-10)
    for _ in 1:maxiter
        pred = _knockdown_model_fn(speed, k, L)
        residual = pred .- dz
        dk = 1e-8
        dL = 1e-4
        pred_dk = _knockdown_model_fn(speed, k + dk, L)
        pred_dL = _knockdown_model_fn(speed, k, L + dL)
        grad_k = 2.0 * sum(residual .* (pred_dk .- pred) ./ dk)
        grad_L = 2.0 * sum(residual .* (pred_dL .- pred) ./ dL)
        k = max(k - lr_k * grad_k, 1e-15)
        L = max(L - lr_L * grad_L, 1e-15)
    end
    return k, L
end

function Base.show(io::IO, m::MooringKnockdownModel)
    print(io, "MooringKnockdownModel(k=$(round(m.k; sigdigits=4)), line_length=$(m.line_length))")
end


"""
    virtual_mooring(reader, lon, lat, nominal_depths;
                    knockdown=nothing, variables=(:temperature, :salinity, :u, :v),
                    depth_grid=nothing, max_dist_km=25.0)

Sample a model reader as a "virtual mooring" with optional knockdown.

Returns a NamedTuple with fields for each variable, plus `actual_depth`,
`knockdown`, `current_speed`, `time`, and `nominal_depth`.
"""
function virtual_mooring(reader, lon::Real, lat::Real,
                         nominal_depths::AbstractVector{<:Real};
                         knockdown::Union{MooringKnockdownModel, Nothing}=nothing,
                         depth_grid::Union{AbstractVector{<:Real}, Nothing}=nothing,
                         variables::Tuple=(:temperature, :salinity, :u, :v),
                         max_dist_km::Real=25.0)
    nd = Float64.(nominal_depths)
    if depth_grid === nothing
        depth_grid = collect(0.0:10.0:maximum(nd) + 100.0)
    end
    model_depths = -Float64.(depth_grid)  # negative-down convention

    model_lon = reader.lon
    model_lat = reader.lat

    profiles = Dict{Symbol, Any}()

    if :temperature in variables
        T_result = temperature(reader; depths=model_depths)
        profiles[:temperature] = T_result.data
    end
    if :salinity in variables
        S_result = salinity(reader; depths=model_depths)
        profiles[:salinity] = S_result.data
    end

    need_uv = :u in variables || :v in variables || knockdown !== nothing
    u_prof = nothing
    v_prof = nothing
    if need_uv
        vel_result = velocities(reader; depths=model_depths)
        u_prof = vel_result.u
        v_prof = vel_result.v
        if :u in variables
            profiles[:u] = u_prof
        end
        if :v in variables
            profiles[:v] = v_prof
        end
    end

    first_key = first(keys(profiles))
    first_data = profiles[first_key]
    nt = ndims(first_data) >= 4 ? size(first_data, 4) : 1

    n_z = length(nd)
    nz_grid = length(model_depths)

    out_vars = Dict{Symbol, Matrix{Float64}}(v => fill(NaN, nt, n_z) for v in variables)
    actual_depth = fill(NaN, nt, n_z)
    knockdown_arr = zeros(nt, n_z)
    speed_arr = fill(NaN, nt, n_z)

    for iz in 1:n_z
        z_nom = nd[iz]
        model_z_nom = -z_nom

        speed_nom = fill(NaN, nt)
        if u_prof !== nothing && v_prof !== nothing
            u_at_z = _extract_at_mooring_and_depth(u_prof, model_lon, model_lat,
                                                    lon, lat, model_z_nom, model_depths,
                                                    max_dist_km)
            v_at_z = _extract_at_mooring_and_depth(v_prof, model_lon, model_lat,
                                                    lon, lat, model_z_nom, model_depths,
                                                    max_dist_km)
            speed_nom = sqrt.(u_at_z .^ 2 .+ v_at_z .^ 2)
        end

        dz = if knockdown !== nothing
            dz_raw = delta_z(knockdown, speed_nom)
            replace(dz_raw, NaN => 0.0)
        else
            zeros(nt)
        end

        speed_arr[:, iz] = speed_nom
        knockdown_arr[:, iz] = dz
        z_actual = z_nom .+ dz
        actual_depth[:, iz] = z_actual
        model_z_actual = -z_actual

        for v in variables
            if v == :pressure
                out_vars[v][:, iz] = depth_to_pressure(z_actual; latitude=lat)
            else
                prof_data = profiles[v]
                for it in 1:nt
                    out_vars[v][it, iz] = _interp_profile_at_depth(
                        prof_data, model_lon, model_lat,
                        lon, lat, model_z_actual[it], model_depths,
                        max_dist_km, it)
                end
            end
        end
    end

    return (;
        (v => out_vars[v] for v in variables)...,
        actual_depth = actual_depth,
        knockdown = knockdown_arr,
        current_speed = speed_arr,
        nominal_depth = nd,
    )
end

function _extract_at_mooring_and_depth(field4d, model_lon, model_lat,
                                        lon, lat, target_z, model_depths,
                                        max_dist_km)
    nz = length(model_depths)
    nt = ndims(field4d) >= 4 ? size(field4d, 4) : 1
    out = Vector{Float64}(undef, nt)

    for it in 1:nt
        profile = Vector{Float64}(undef, nz)
        for iz in 1:nz
            f2d = ndims(field4d) >= 4 ? field4d[:, :, iz, it] : field4d[:, :, iz]
            regular = ndims(model_lon) == 1 && ndims(model_lat) == 1
            v = _spatial_interp(f2d, model_lon, model_lat, regular,
                                Float64[lon], Float64[lat], max_dist_km, :bilinear)
            profile[iz] = v[1]
        end
        out[it] = _interp_1d_depth(model_depths, profile, target_z)
    end
    return out
end

function _interp_profile_at_depth(field, model_lon, model_lat,
                                   lon, lat, target_z, model_depths,
                                   max_dist_km, it)
    nz = length(model_depths)
    profile = Vector{Float64}(undef, nz)
    for iz in 1:nz
        f2d = ndims(field) >= 4 ? field[:, :, iz, it] : field[:, :, iz]
        regular = ndims(model_lon) == 1 && ndims(model_lat) == 1
        v = _spatial_interp(f2d, model_lon, model_lat, regular,
                            Float64[lon], Float64[lat], max_dist_km, :bilinear)
        profile[iz] = v[1]
    end
    return _interp_1d_depth(model_depths, profile, target_z)
end

function _interp_1d_depth(depth_vals, profile, target_z)
    valid = isfinite.(profile) .& isfinite.(depth_vals)
    sum(valid) < 2 && return NaN
    d = Float64.(depth_vals[valid])
    v = Float64.(profile[valid])
    order = sortperm(d)
    d = d[order]
    v = v[order]

    if target_z < d[1] || target_z > d[end]
        return NaN
    end

    idx = searchsortedlast(d, target_z)
    idx = clamp(idx, 1, length(d) - 1)
    dz = d[idx+1] - d[idx]
    if abs(dz) < 1e-10
        return v[idx]
    end
    w = clamp((target_z - d[idx]) / dz, 0.0, 1.0)
    return v[idx] + w * (v[idx+1] - v[idx])
end


"""
    project_to_fixed_depths(time, depths, values, target_depths;
                            extrapolate=false)

Project time-varying-depth mooring records onto a fixed depth grid.

# Arguments
- `time`: vector of DateTime `(n_time,)`
- `depths`: matrix `(n_time, n_instruments)` [m, positive down]
- `values`: Dict of variable name => matrix `(n_time, n_instruments)`
- `target_depths`: vector [m, positive down]
- `extrapolate`: hold nearest value outside instrument range

Returns a NamedTuple with projected variables, `n_instruments`, `time`, `depth`.
"""
function project_to_fixed_depths(time::AbstractVector,
                                  depths::AbstractMatrix{<:Real},
                                  values::Dict{Symbol, <:AbstractMatrix{<:Real}},
                                  target_depths::AbstractVector{<:Real};
                                  extrapolate::Bool=false)
    td = Float64.(target_depths)
    n_time, n_instr = size(depths)
    n_z = length(td)

    for (name, arr) in values
        size(arr) == (n_time, n_instr) || error(
            "values[:$name] has shape $(size(arr)), expected ($n_time, $n_instr)")
    end

    out = Dict{Symbol, Matrix{Float64}}(name => fill(NaN, n_time, n_z) for name in keys(values))
    n_used = zeros(Int, n_time)

    for it in 1:n_time
        d_t = Float64.(depths[it, :])
        valid_d = isfinite.(d_t)
        sum(valid_d) < 2 && continue

        order = sortperm(d_t[valid_d])
        d_sorted = d_t[valid_d][order]

        for (name, arr) in values
            v_t = Float64.(arr[it, valid_d][order])
            valid_v = isfinite.(v_t)
            sum(valid_v) < 2 && continue
            d_v = d_sorted[valid_v]
            c_v = v_t[valid_v]

            for iz in 1:n_z
                zt = td[iz]
                if !extrapolate && (zt < d_v[1] || zt > d_v[end])
                    continue
                end
                zt_c = extrapolate ? clamp(zt, d_v[1], d_v[end]) : zt
                idx = searchsortedlast(d_v, zt_c)
                idx = clamp(idx, 1, length(d_v) - 1)
                dz = d_v[idx+1] - d_v[idx]
                w = abs(dz) > 1e-10 ? clamp((zt_c - d_v[idx]) / dz, 0.0, 1.0) : 0.5
                out[name][it, iz] = c_v[idx] + w * (c_v[idx+1] - c_v[idx])
            end
        end
        n_used[it] = sum(valid_d)
    end

    return (; (name => out[name] for name in keys(out))...,
            n_instruments = n_used,
            time = time,
            depth = td)
end
