"""
    RAFOSLoader

Loader for RAFOS/subsurface Lagrangian float data.
"""
const _RAFOS_FILL = (99999.0, 9999.0, 999.0, -999.0, -9999.0)
const _DEPTH_PER_DBAR_OBS = 1.019716

mutable struct RAFOSLoader
    df::DataFrame
end

function RAFOSLoader(files::AbstractString;
                     bbox::Union{NTuple{4,Real}, Nothing}=nothing,
                     date_range::Union{Tuple{String,String}, Nothing}=nothing,
                     pressure_range::Union{Tuple{Real,Real}, Nothing}=nothing)
    expanded = glob(files)
    isempty(expanded) && (expanded = [files])

    all_dfs = DataFrame[]
    for f in expanded
        if endswith(f, ".nc") || endswith(f, ".nc4")
            push!(all_dfs, _read_rafos_nc(f))
        else
            push!(all_dfs, _read_rafos_csv(f))
        end
    end

    df = isempty(all_dfs) ? DataFrame() : vcat(all_dfs...; cols=:union)
    isempty(df) && return RAFOSLoader(df)

    for col in (:lon, :lat, :pressure)
        hasproperty(df, col) || continue
        for fv in _RAFOS_FILL
            df[!, col] = replace(df[!, col], fv => NaN)
        end
    end

    if hasproperty(df, :pressure)
        df[!, :depth] = df.pressure .* _DEPTH_PER_DBAR_OBS
    end

    if bbox !== nothing && hasproperty(df, :lon) && hasproperty(df, :lat)
        filter!(row -> row.lon >= bbox[1] && row.lon <= bbox[3] &&
                       row.lat >= bbox[2] && row.lat <= bbox[4], df)
    end
    if date_range !== nothing && hasproperty(df, :time)
        t0, t1 = DateTime(date_range[1]), DateTime(date_range[2])
        filter!(row -> row.time >= t0 && row.time <= t1, df)
    end
    if pressure_range !== nothing && hasproperty(df, :pressure)
        filter!(row -> isfinite(row.pressure) &&
                       row.pressure >= pressure_range[1] &&
                       row.pressure <= pressure_range[2], df)
    end

    return RAFOSLoader(df)
end

function rafos_positions(r::RAFOSLoader)
    cols = intersect(names(r.df), ["float_id", "cycle", "time", "lon", "lat", "pressure", "depth"])
    return select(r.df, Symbol.(cols))
end

function rafos_velocity_estimates(r::RAFOSLoader;
                                  min_dt_hours::Real=12.0,
                                  max_dt_hours::Real=96.0)
    hasproperty(r.df, :float_id) || return DataFrame()
    R_earth = 6371.0e3

    rows = NamedTuple[]
    for gdf in groupby(r.df, :float_id)
        sorted = sort(gdf, :time)
        n = nrow(sorted)
        for i in 2:n
            dt_s = Dates.value(sorted.time[i] - sorted.time[i-1]) / 1000.0
            dt_h = dt_s / 3600.0
            (dt_h < min_dt_hours || dt_h > max_dt_hours) && continue

            dlon = sorted.lon[i] - sorted.lon[i-1]
            dlat = sorted.lat[i] - sorted.lat[i-1]
            mean_lat = 0.5 * (sorted.lat[i] + sorted.lat[i-1])

            dx = dlon * cosd(mean_lat) * π / 180 * R_earth
            dy = dlat * π / 180 * R_earth
            u = dx / dt_s
            v = dy / dt_s

            push!(rows, (float_id=sorted.float_id[i],
                         time=sorted.time[i],
                         lon=sorted.lon[i], lat=sorted.lat[i],
                         u=u, v=v, speed=hypot(u, v),
                         dt_hours=dt_h))
        end
    end
    return DataFrame(rows)
end

rafos_to_dataframe(r::RAFOSLoader) = copy(r.df)

function _read_rafos_nc(path::String)
    ds = NCDataset(path, "r")
    df = DataFrame()
    for (nc_name, col) in [("LONGITUDE","lon"), ("LATITUDE","lat"),
                            ("PRES","pressure"), ("TEMP","temperature")]
        haskey(ds, nc_name) && (df[!, Symbol(col)] = vec(Float64.(Array(ds[nc_name]))))
    end
    for tname in ("JULD", "TIME", "time")
        if haskey(ds, tname)
            df[!, :time] = vec(Array(ds[tname]))
            break
        end
    end
    for idname in ("PLATFORM_NUMBER", "float_id", "WMO")
        if haskey(ds, idname)
            df[!, :float_id] = vec(string.(Array(ds[idname])))
            break
        end
    end
    close(ds)
    return df
end

function _read_rafos_csv(path::String)
    lines = readlines(path)
    isempty(lines) && return DataFrame()
    header = Symbol.(split(strip(replace(lines[1], "#" => "")), r"[,\t\s]+"))
    rows = NamedTuple[]
    for line in lines[2:end]
        parts = split(strip(line), r"[,\t\s]+")
        length(parts) < length(header) && continue
        row = Dict{Symbol, Any}()
        for (i, h) in enumerate(header)
            i > length(parts) && continue
            val = tryparse(Float64, parts[i])
            row[h] = val !== nothing ? val : parts[i]
        end
        push!(rows, NamedTuple(row))
    end
    return DataFrame(rows)
end

Base.close(r::RAFOSLoader) = nothing
