"""
    GEMBuilder

Gravest Empirical Mode (GEM) builder from Argo T/S profiles.
"""
mutable struct GEMBuilder
    depth_grid::Vector{Float64}
    tau_values::Vector{Float64}
    T_table::Union{Matrix{Float64}, Nothing}
    S_table::Union{Matrix{Float64}, Nothing}
    T_splines::Union{Vector{Any}, Nothing}
    S_splines::Union{Vector{Any}, Nothing}
    site::Union{String, Nothing}
end

function GEMBuilder(; depth_grid::Union{AbstractVector{<:Real}, Nothing}=nothing,
                     site::Union{String, Nothing}=nothing)
    dg = depth_grid !== nothing ? Float64.(depth_grid) : collect(0.0:10.0:2000.0)
    return GEMBuilder(dg, Float64[], nothing, nothing, nothing, nothing, site)
end

"""
    sound_speed_chen_millero(T, S, P)

Chen & Millero (1977) sound speed [m/s].
T [°C], S [PSU], P [dbar].
"""
function sound_speed_chen_millero(T::Real, S::Real, P::Real)
    return 1449.2 + 4.6*T - 0.055*T^2 + 0.00029*T^3 +
           (1.34 - 0.01*T) * (S - 35) + 0.016*P
end

function sound_speed_chen_millero(T::AbstractArray, S::AbstractArray, P::AbstractArray)
    return @. 1449.2 + 4.6*T - 0.055*T^2 + 0.00029*T^3 +
              (1.34 - 0.01*T) * (S - 35) + 0.016*P
end

function gem_from_argo(argo::ArgoLoader;
                       depth_grid::Union{AbstractVector{<:Real}, Nothing}=nothing,
                       good_qc::Bool=true,
                       site::Union{String, Nothing}=nothing)
    g = GEMBuilder(; depth_grid=depth_grid, site=site)

    T_raw = argo_temperature(argo; good_qc=good_qc)
    S_raw = argo_salinity(argo; good_qc=good_qc)
    P_raw = argo_pressure(argo)

    (T_raw === nothing || S_raw === nothing || P_raw === nothing) &&
        error("Cannot build GEM: missing T, S, or P data in Argo loader.")

    n_prof = size(T_raw, 1)
    nz = length(g.depth_grid)
    tau_list = Float64[]
    T_interp = Vector{Vector{Float64}}()
    S_interp = Vector{Vector{Float64}}()

    for ip in 1:n_prof
        p_prof = ndims(P_raw) >= 2 ? P_raw[ip, :] : [P_raw[ip]]
        t_prof = ndims(T_raw) >= 2 ? T_raw[ip, :] : [T_raw[ip]]
        s_prof = ndims(S_raw) >= 2 ? S_raw[ip, :] : [S_raw[ip]]

        valid = isfinite.(p_prof) .& isfinite.(t_prof) .& isfinite.(s_prof)
        sum(valid) < 3 && continue

        p_v = p_prof[valid]
        t_v = t_prof[valid]
        s_v = s_prof[valid]
        order = sortperm(p_v)
        p_v, t_v, s_v = p_v[order], t_v[order], s_v[order]

        d_v = p_v .* _DEPTH_PER_DBAR_OBS

        c = sound_speed_chen_millero(t_v, s_v, p_v)
        dz = diff(d_v)
        c_mid = 0.5 .* (c[1:end-1] .+ c[2:end])
        tau_cumul = vcat(0.0, cumsum(2.0 .* dz ./ c_mid))
        tau_total = tau_cumul[end]
        tau_total <= 0 && continue

        T_on_grid = _interp_profile_to_grid(d_v, t_v, g.depth_grid)
        S_on_grid = _interp_profile_to_grid(d_v, s_v, g.depth_grid)

        push!(tau_list, tau_total)
        push!(T_interp, T_on_grid)
        push!(S_interp, S_on_grid)
    end

    isempty(tau_list) && error("No valid profiles for GEM construction.")

    order = sortperm(tau_list)
    g.tau_values = tau_list[order]
    g.T_table = hcat(T_interp[order]...)'  # (n_prof, nz)
    g.S_table = hcat(S_interp[order]...)'

    return g
end

function gem_fit!(g::GEMBuilder)
    g.T_table === nothing && error("No T/S table. Call gem_from_argo first.")
    n_tau = length(g.tau_values)
    nz = length(g.depth_grid)

    g.T_splines = Vector{Any}(undef, nz)
    g.S_splines = Vector{Any}(undef, nz)

    for iz in 1:nz
        t_col = g.T_table[:, iz]
        s_col = g.S_table[:, iz]
        valid = isfinite.(t_col) .& isfinite.(g.tau_values)
        if sum(valid) >= 4
            tau_v = g.tau_values[valid]
            order = sortperm(tau_v)
            g.T_splines[iz] = (tau=tau_v[order], vals=t_col[valid][order])
            g.S_splines[iz] = (tau=tau_v[order], vals=s_col[valid][order])
        else
            g.T_splines[iz] = nothing
            g.S_splines[iz] = nothing
        end
    end
    return g
end

function gem_tau_to_profiles(g::GEMBuilder, tau_series::AbstractVector{<:Real})
    g.T_splines === nothing && error("GEM not fitted. Call gem_fit! first.")
    nz = length(g.depth_grid)
    nt = length(tau_series)

    T_out = fill(NaN, nt, nz)
    S_out = fill(NaN, nt, nz)

    for iz in 1:nz
        g.T_splines[iz] === nothing && continue
        tau_ref = g.T_splines[iz].tau
        t_ref = g.T_splines[iz].vals
        s_ref = g.S_splines[iz].vals

        for it in 1:nt
            tau = tau_series[it]
            !isfinite(tau) && continue
            T_out[it, iz] = _interp1_clamp(tau_ref, t_ref, tau)
            S_out[it, iz] = _interp1_clamp(tau_ref, s_ref, tau)
        end
    end

    return (temperature=T_out, salinity=S_out, depth=g.depth_grid)
end

function gem_save(g::GEMBuilder, path::String)
    ds = NCDataset(path, "c")
    nz = length(g.depth_grid)
    n_tau = length(g.tau_values)

    defDim(ds, "depth", nz)
    defDim(ds, "tau", n_tau)

    defVar(ds, "depth", g.depth_grid, ("depth",))
    defVar(ds, "tau", g.tau_values, ("tau",))
    g.T_table !== nothing && defVar(ds, "T_table", g.T_table, ("tau", "depth"))
    g.S_table !== nothing && defVar(ds, "S_table", g.S_table, ("tau", "depth"))

    g.site !== nothing && (ds.attrib["site"] = g.site)
    close(ds)
end

function gem_load(path::String; fit::Bool=true)
    ds = NCDataset(path, "r")
    depth_grid = Float64.(Array(ds["depth"]))
    tau_values = Float64.(Array(ds["tau"]))
    T_table = haskey(ds, "T_table") ? Float64.(Array(ds["T_table"])) : nothing
    S_table = haskey(ds, "S_table") ? Float64.(Array(ds["S_table"])) : nothing
    site = haskey(ds.attrib, "site") ? ds.attrib["site"] : nothing
    close(ds)

    g = GEMBuilder(depth_grid, tau_values, T_table, S_table, nothing, nothing, site)
    fit && gem_fit!(g)
    return g
end

function gem_model_tau(g::GEMBuilder, reader, lon::Real, lat::Real)
    error("gem_model_tau requires a reader with temperature/salinity methods — use with CROCOReader/NEMOReader")
end

function _interp_profile_to_grid(z_in, v_in, z_grid)
    n = length(z_grid)
    out = fill(NaN, n)
    nz_in = length(z_in)
    nz_in < 2 && return out

    for i in 1:n
        zt = z_grid[i]
        zt < z_in[1] && continue
        zt > z_in[end] && continue
        k = searchsortedlast(z_in, zt)
        k = clamp(k, 1, nz_in - 1)
        dz = z_in[k+1] - z_in[k]
        w = abs(dz) > 1e-10 ? clamp((zt - z_in[k]) / dz, 0.0, 1.0) : 0.5
        out[i] = v_in[k] + w * (v_in[k+1] - v_in[k])
    end
    return out
end

function _interp1_clamp(xp, yp, x)
    x <= xp[1] && return yp[1]
    x >= xp[end] && return yp[end]
    k = searchsortedlast(xp, x)
    k = clamp(k, 1, length(xp) - 1)
    t = (x - xp[k]) / (xp[k+1] - xp[k])
    return yp[k] + t * (yp[k+1] - yp[k])
end
