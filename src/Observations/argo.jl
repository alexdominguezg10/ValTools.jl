"""
    ArgoLoader

Loader for Argo float profile data (GDAC NetCDF format).

# Fields
- `ds::NCDataset`
- `lon::Vector{Float64}`, `lat::Vector{Float64}`, `time::Vector{DateTime}`
- `n_profiles::Int`
"""
mutable struct ArgoLoader
    ds::NCDataset
    lon::Vector{Float64}
    lat::Vector{Float64}
    time::Vector{DateTime}
    n_profiles::Int
end

function ArgoLoader(files::AbstractString;
                    bbox::Union{NTuple{4,Real}, Nothing}=nothing,
                    date_range::Union{Tuple{String,String}, Nothing}=nothing)
    ds = _obs_open_nc(files)

    lon = _obs_read_var_f64(ds, ("LONGITUDE", "longitude", "lon"))
    lat = _obs_read_var_f64(ds, ("LATITUDE", "latitude", "lat"))
    time = _obs_read_time(ds, ("JULD", "TIME", "time"))

    mask = trues(length(lon))
    if bbox !== nothing
        mask .&= (lon .>= bbox[1]) .& (lon .<= bbox[3]) .&
                 (lat .>= bbox[2]) .& (lat .<= bbox[4])
    end
    if date_range !== nothing
        t0, t1 = DateTime(date_range[1]), DateTime(date_range[2])
        mask .&= (time .>= t0) .& (time .<= t1)
    end

    idx = findall(mask)
    return ArgoLoader(ds, lon[idx], lat[idx], time[idx], length(idx))
end

function argo_temperature(r::ArgoLoader; good_qc::Bool=true)
    data = _obs_read_var_f64(r.ds, ("TEMP_ADJUSTED", "TEMP", "temperature"))
    if good_qc
        qc = _obs_read_qc(r.ds, ("TEMP_QC", "TEMP_ADJUSTED_QC"))
        qc !== nothing && _apply_qc_mask!(data, qc)
    end
    return data
end

function argo_salinity(r::ArgoLoader; good_qc::Bool=true)
    data = _obs_read_var_f64(r.ds, ("PSAL_ADJUSTED", "PSAL", "salinity"))
    if good_qc
        qc = _obs_read_qc(r.ds, ("PSAL_QC", "PSAL_ADJUSTED_QC"))
        qc !== nothing && _apply_qc_mask!(data, qc)
    end
    return data
end

function argo_pressure(r::ArgoLoader)
    return _obs_read_var_f64(r.ds, ("PRES_ADJUSTED", "PRES", "pressure"))
end

function argo_to_dataframe(r::ArgoLoader; good_qc::Bool=true)
    T = argo_temperature(r; good_qc=good_qc)
    S = argo_salinity(r; good_qc=good_qc)
    P = argo_pressure(r)

    T === nothing && return DataFrame()

    rows = NamedTuple[]
    n_prof = min(r.n_profiles, size(T, 1))
    n_lev = ndims(T) >= 2 ? size(T, 2) : 1

    for ip in 1:n_prof, il in 1:n_lev
        t_val = ndims(T) >= 2 ? T[ip, il] : T[ip]
        s_val = S !== nothing ? (ndims(S) >= 2 ? S[ip, il] : S[ip]) : NaN
        p_val = P !== nothing ? (ndims(P) >= 2 ? P[ip, il] : P[ip]) : NaN
        (isnan(t_val) && isnan(s_val)) && continue
        push!(rows, (lon=r.lon[ip], lat=r.lat[ip], time=r.time[ip],
                      pressure=p_val, temperature=t_val, salinity=s_val))
    end
    return DataFrame(rows)
end

Base.close(r::ArgoLoader) = close(r.ds)
