"""
    compute_metrics(obs, model; weights=nothing, prefix="")

Compute standard model-vs-observation validation metrics.

Returns a `Dict{String,Float64}` with keys:
`n`, `bias`, `mae`, `rmse`, `rmse_unbiased`, `std_obs`, `std_model`,
`correlation`, `scatter_index`, `skill_score`, `nse`.
"""
function compute_metrics(obs::AbstractVector{<:Real},
                         model::AbstractVector{<:Real};
                         weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
                         prefix::String="")
    o = Float64.(obs)
    m = Float64.(model)

    valid = isfinite.(o) .& isfinite.(m)
    nvalid = sum(valid)

    all_keys = ("n", "bias", "mae", "rmse", "rmse_unbiased",
                "std_obs", "std_model", "correlation",
                "scatter_index", "skill_score", "nse")

    if nvalid < 2
        return Dict{String,Float64}(prefix * k => NaN for k in all_keys)
    end

    ov = o[valid]
    mv = m[valid]
    n = length(ov)

    w = if weights !== nothing
        wv = Float64.(weights)[valid]
        wv ./ sum(wv) .* n
    else
        ones(n)
    end

    diff = mv .- ov
    wsum = sum(w)

    bias_val = sum(w .* diff) / wsum
    mae_val  = sum(w .* abs.(diff)) / wsum
    mse_val  = sum(w .* diff .^ 2) / wsum
    rmse_val = sqrt(mse_val)
    rmse_ub  = sqrt(max(mse_val - bias_val^2, 0.0))

    o_mean  = sum(w .* ov) / wsum
    m_mean  = sum(w .* mv) / wsum
    std_obs = sqrt(sum(w .* (ov .- o_mean) .^ 2) / wsum)
    std_mod = sqrt(sum(w .* (mv .- m_mean) .^ 2) / wsum)

    corr = if std_obs > 0 && std_mod > 0
        r = sum(w .* (ov .- o_mean) .* (mv .- m_mean)) / wsum / (std_obs * std_mod)
        clamp(r, -1.0, 1.0)
    else
        NaN
    end

    si  = std_obs > 0 ? rmse_ub / std_obs : NaN
    ss  = std_obs > 0 ? 1.0 - mse_val / std_obs^2 : NaN
    nse_val = std_obs > 0 ? 1.0 - sum((ov .- mv) .^ 2) / sum((ov .- o_mean) .^ 2) : NaN

    return Dict{String,Float64}(
        prefix * "n"              => Float64(nvalid),
        prefix * "bias"           => round(bias_val; digits=6),
        prefix * "mae"            => round(mae_val;  digits=6),
        prefix * "rmse"           => round(rmse_val; digits=6),
        prefix * "rmse_unbiased"  => round(rmse_ub;  digits=6),
        prefix * "std_obs"        => round(std_obs;  digits=6),
        prefix * "std_model"      => round(std_mod;  digits=6),
        prefix * "correlation"    => round(corr;     digits=6),
        prefix * "scatter_index"  => isfinite(si)  ? round(si;  digits=6) : NaN,
        prefix * "skill_score"    => isfinite(ss)  ? round(ss;  digits=6) : NaN,
        prefix * "nse"            => isfinite(nse_val) ? round(nse_val; digits=6) : NaN,
    )
end

compute_metrics(obs, model; kwargs...) =
    compute_metrics(vec(collect(Float64, obs)), vec(collect(Float64, model)); kwargs...)

# Strip units, converting `model` to `obs`'s unit first -- same auto-convert
# behavior already shipped in Dispatch._stripped_common_unit (the fix for
# commit 801b386's cm/s-vs-m/s skill_score bug), not the "reject on
# mismatch" design sketched in the original roadmap. Keeping this
# consistent with the already-tested Dispatch behavior rather than
# reintroducing the two-implementations-of-the-same-bug-class risk.
function _stripped_to_obs_unit(obs_value, model_value)
    ou = Unitful.unit(first(obs_value))
    return Unitful.ustrip.(obs_value), Unitful.ustrip.(Unitful.uconvert.(ou, model_value))
end

"""
    compute_metrics(obs::Types.TimeSeriesVector, model::Types.TimeSeriesVector; weights=nothing, prefix="")

Typed overload for unit-tagged series. `model` is converted to `obs`'s
unit before computing (see [`Dispatch._stripped_common_unit`](@ref) for
the same convention on the single-metric `Dispatch` functions) -- a
mismatched-but-compatible unit (e.g. cm/s vs m/s) is auto-converted, not
rejected. All metrics keyed in `obs`'s unit's numeric scale, same as the
plain-`Vector` method.
"""
function compute_metrics(obs::Types.TimeSeriesVector, model::Types.TimeSeriesVector;
                         weights::Union{AbstractVector{<:Real}, Nothing}=nothing,
                         prefix::String="")
    ov, mv = _stripped_to_obs_unit(obs.value, model.value)
    return compute_metrics(ov, mv; weights=weights, prefix=prefix)
end


"""
    taylor_stats(obs, model)

Return the three numbers needed for a Taylor diagram point:
`std_ref`, `std_test`, `correlation`, `rms_diff`.
"""
function taylor_stats(obs, model)
    m = compute_metrics(vec(collect(Float64, obs)), vec(collect(Float64, model)))
    return Dict{String,Float64}(
        "std_ref"     => m["std_obs"],
        "std_test"    => m["std_model"],
        "correlation" => m["correlation"],
        "rms_diff"    => m["rmse_unbiased"],
    )
end

"""
    taylor_stats(obs::Types.TimeSeriesVector, model::Types.TimeSeriesVector)

Typed overload for unit-tagged series, same auto-convert-to-`obs`'s-unit
convention as the typed [`compute_metrics`](@ref) above.
"""
function taylor_stats(obs::Types.TimeSeriesVector, model::Types.TimeSeriesVector)
    ov, mv = _stripped_to_obs_unit(obs.value, model.value)
    return taylor_stats(ov, mv)
end


"""
    bootstrap_metrics(obs, model; n_boot=1000, ci=0.95, seed=42)

Bootstrap confidence intervals for validation metrics.
Returns `Dict{String, NamedTuple{(:mean,:lo,:hi)}}`.
"""
function bootstrap_metrics(obs::AbstractVector{<:Real},
                           model::AbstractVector{<:Real};
                           n_boot::Int=1000, ci::Float64=0.95, seed::Int=42)
    rng = MersenneTwister(seed)
    n = length(obs)
    o = Float64.(obs)
    m = Float64.(model)

    boots = [begin
        idx = rand(rng, 1:n, n)
        compute_metrics(o[idx], m[idx])
    end for _ in 1:n_boot]

    keys_list = collect(keys(boots[1]))
    alpha = (1.0 - ci) / 2.0

    result = Dict{String, NamedTuple{(:mean,:lo,:hi), Tuple{Float64,Float64,Float64}}}()
    for k in keys_list
        vals = [b[k] for b in boots]
        finite_vals = filter(isfinite, vals)
        if length(finite_vals) < 2
            result[k] = (mean=NaN, lo=NaN, hi=NaN)
        else
            sorted = sort(finite_vals)
            n_f = length(sorted)
            result[k] = (
                mean = mean(finite_vals),
                lo   = sorted[max(1, round(Int, alpha * n_f))],
                hi   = sorted[min(n_f, round(Int, (1 - alpha) * n_f))],
            )
        end
    end
    return result
end


"""
    metrics_by_group(df, obs_col, model_col, group_col)

Compute metrics separately for each group. Returns a `DataFrame`.
"""
function metrics_by_group(df::DataFrame, obs_col::Symbol, model_col::Symbol,
                          group_col::Symbol)
    groups = unique(df[!, group_col])
    rows = Dict{String,Float64}[]
    group_vals = []
    for g in groups
        sub = filter(row -> row[group_col] == g, df)
        m = compute_metrics(sub[!, obs_col], sub[!, model_col])
        push!(rows, m)
        push!(group_vals, g)
    end

    isempty(rows) && return DataFrame()

    all_keys = sort(collect(keys(rows[1])))
    result = DataFrame()
    result[!, group_col] = group_vals
    for k in all_keys
        result[!, Symbol(k)] = [r[k] for r in rows]
    end
    return result
end


"""
    current_ellipse_metrics(obs_u, obs_v, model_u, model_v)

Compare observed and model current variance ellipses.
"""
function current_ellipse_metrics(obs_u::AbstractVector, obs_v::AbstractVector,
                                 model_u::AbstractVector, model_v::AbstractVector)
    valid = isfinite.(obs_u) .& isfinite.(obs_v) .& isfinite.(model_u) .& isfinite.(model_v)

    function _ellipse(u, v)
        u_anom = u .- mean(u)
        v_anom = v .- mean(v)
        C = [mean(u_anom .^ 2)        mean(u_anom .* v_anom);
             mean(u_anom .* v_anom)    mean(v_anom .^ 2)]
        tr_val = C[1,1] + C[2,2]
        det_val = C[1,1] * C[2,2] - C[1,2]^2
        disc = sqrt(max((tr_val / 2)^2 - det_val, 0.0))
        l1 = tr_val / 2 + disc
        l2 = tr_val / 2 - disc
        theta = 0.5 * atan(2 * C[1,2], C[1,1] - C[2,2])
        incl = mod(90.0 - rad2deg(theta), 180.0)
        eke = 0.5 * (var(u) + var(v))
        return sqrt(max(l1, 0)), sqrt(max(l2, 0)), incl, eke
    end

    ou, ov = Float64.(obs_u[valid]),   Float64.(obs_v[valid])
    mu_arr, mv_arr = Float64.(model_u[valid]), Float64.(model_v[valid])

    o_a, o_b, o_inc, o_eke = _ellipse(ou, ov)
    m_a, m_b, m_inc, m_eke = _ellipse(mu_arr, mv_arr)

    o_spd = hypot.(ou, ov)
    m_spd = hypot.(mu_arr, mv_arr)

    mu_m = compute_metrics(ou, mu_arr)
    mv_m = compute_metrics(ov, mv_arr)
    ms_m = compute_metrics(o_spd, m_spd)

    o_MKE = 0.5 * (mean(ou)^2 + mean(ov)^2)
    m_MKE = 0.5 * (mean(mu_arr)^2 + mean(mv_arr)^2)

    return Dict{String,Float64}(
        "obs_semi_major"  => round(o_a;   digits=4),
        "obs_semi_minor"  => round(o_b;   digits=4),
        "obs_inclination" => round(o_inc; digits=2),
        "obs_EKE"         => round(o_eke; digits=6),
        "obs_MKE"         => round(o_MKE; digits=6),
        "mod_semi_major"  => round(m_a;   digits=4),
        "mod_semi_minor"  => round(m_b;   digits=4),
        "mod_inclination" => round(m_inc; digits=2),
        "mod_EKE"         => round(m_eke; digits=6),
        "mod_MKE"         => round(m_MKE; digits=6),
        "u_rmse"          => mu_m["rmse"],
        "v_rmse"          => mv_m["rmse"],
        "u_bias"          => mu_m["bias"],
        "v_bias"          => mv_m["bias"],
        "u_corr"          => mu_m["correlation"],
        "v_corr"          => mv_m["correlation"],
        "speed_rmse"      => ms_m["rmse"],
        "speed_bias"      => ms_m["bias"],
        "speed_corr"      => ms_m["correlation"],
    )
end
