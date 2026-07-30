"""
    NDBCLoader

Loader for NDBC meteorological buoy data (CSV/TXT format).
"""
const _NDBC_FILL_VALUES = (99.0, 999.0, 9999.0, 99.00, 999.00, 9999.00)

const _NDBC_COL_MAP = Dict(
    "WVHT" => :Hs, "DPD" => :Tp, "APD" => :Ta, "MWD" => :Mwd,
    "WSPD" => :wspd, "WDIR" => :wdir, "GST" => :gst,
    "ATMP" => :atmp, "WTMP" => :wtmp, "PRES" => :slp, "BAR" => :slp,
)

const _NDBC_GOM_STATIONS = Dict(
    "42001" => (lon=-89.658, lat=25.888),
    "42002" => (lon=-93.666, lat=25.790),
    "42019" => (lon=-95.360, lat=27.910),
    "42020" => (lon=-96.694, lat=26.966),
    "42055" => (lon=-94.047, lat=22.120),
    "42056" => (lon=-85.059, lat=19.874),
)

mutable struct NDBCLoader
    df::DataFrame
    stations::Vector{String}
    station_meta::Dict{String, NamedTuple{(:lon,:lat), Tuple{Float64,Float64}}}
end

function NDBCLoader(source::AbstractString;
                    station_meta::Union{Dict, Nothing}=nothing)
    files = glob(source)
    isempty(files) && (files = [source])

    all_dfs = DataFrame[]
    for f in files
        df = _parse_ndbc_file(f)
        isempty(df) && continue
        push!(all_dfs, df)
    end

    df = isempty(all_dfs) ? DataFrame() : vcat(all_dfs...)
    stations = sort(unique(string.(df[!, :station])))

    meta = station_meta !== nothing ?
        Dict(k => (lon=Float64(v.lon), lat=Float64(v.lat)) for (k, v) in station_meta) :
        copy(_NDBC_GOM_STATIONS)

    return NDBCLoader(df, stations, meta)
end

function ndbc_station(r::NDBCLoader, stn_id::String)
    return filter(row -> string(row.station) == stn_id, r.df)
end

function ndbc_waves(r::NDBCLoader)
    result = Dict{String, DataFrame}()
    for stn in r.stations
        sub = ndbc_station(r, stn)
        if hasproperty(sub, :Hs)
            result[stn] = select(sub, [:time, :Hs])
        end
    end
    return result
end

function ndbc_winds(r::NDBCLoader)
    result = Dict{String, DataFrame}()
    for stn in r.stations
        sub = ndbc_station(r, stn)
        cols = intersect(names(sub), ["time", "wspd", "wdir", "gst"])
        !isempty(cols) && (result[stn] = select(sub, Symbol.(cols)))
    end
    return result
end

function ndbc_to_dataframe(r::NDBCLoader; add_position::Bool=true)
    df = copy(r.df)
    if add_position
        df[!, :lon] = fill(NaN, nrow(df))
        df[!, :lat] = fill(NaN, nrow(df))
        for i in 1:nrow(df)
            stn = string(df[i, :station])
            if haskey(r.station_meta, stn)
                df[i, :lon] = r.station_meta[stn].lon
                df[i, :lat] = r.station_meta[stn].lat
            end
        end
    end
    return df
end

function _parse_ndbc_file(filepath::String)
    lines = readlines(filepath)
    isempty(lines) && return DataFrame()

    header = split(strip(replace(lines[1], "#" => "")))
    skip = startswith(lines[2], "#") ? 2 : 1

    rows = NamedTuple[]
    stn_id = _extract_station_id(filepath)

    for line in lines[skip+1:end]
        parts = split(strip(line))
        length(parts) < length(header) && continue

        yr = parse(Int, parts[1])
        yr < 100 && (yr += yr < 70 ? 2000 : 1900)
        mo = parse(Int, parts[2])
        da = parse(Int, parts[3])
        hr = length(header) >= 4 && header[4] in ("hh", "HH", "hr") ?
             parse(Int, parts[4]) : 0
        mn = length(header) >= 5 && header[5] in ("mm", "MM", "mn") ?
             parse(Int, parts[5]) : 0

        try
            t = DateTime(yr, mo, da, hr, mn)
            row = Dict{Symbol, Any}(:time => t, :station => stn_id)

            for (ci, col) in enumerate(header)
                ci <= 5 && continue
                ci > length(parts) && continue
                val = tryparse(Float64, parts[ci])
                val === nothing && continue
                val in _NDBC_FILL_VALUES && (val = NaN)
                sym = get(_NDBC_COL_MAP, col, Symbol(col))
                row[sym] = val
            end
            push!(rows, NamedTuple(row))
        catch
            continue
        end
    end
    isempty(rows) && return DataFrame()
    return DataFrame(rows)
end

function _extract_station_id(filepath::String)
    bn = basename(filepath)
    m = match(r"(\d{5})", bn)
    return m !== nothing ? m[1] : splitext(bn)[1]
end

"""
    ndbc_winds_ts(r::NDBCLoader) -> TimeSeriesCollection

Typed counterpart to [`ndbc_winds`](@ref)'s wind-speed column: one
`TimeSeriesVector` per station (named by station id), bundled as a
`TimeSeriesCollection` since each NDBC station has its own independent
timestamps -- there is no shared time axis across stations.

NDBC's plain-text format carries no per-variable units metadata (unlike
the NetCDF-backed loaders, there is no attribute to read), so `m/s` is
used directly per NDBC's documented station format, not inferred from the
data. Errors if no station yields a usable, DateTime-decoded wind-speed
series.
"""
function ndbc_winds_ts(r::NDBCLoader)
    # Built as Vector{Any} then narrowed below -- `Types.TimeSeriesVector[]`
    # (the empty-literal idiom) creates a Vector of the ABSTRACT,
    # unparametrized TimeSeriesVector type, which can never satisfy
    # TimeSeriesCollection's `Vector{TimeSeriesVector{Q}}` constructor even
    # after concrete elements are pushed into it.
    raw = Any[]
    for stn in r.stations
        sub = ndbc_station(r, stn)
        hasproperty(sub, :wspd) || continue
        isempty(sub.time) && continue
        eltype(sub.time) <: Dates.AbstractDateTime || continue
        push!(raw, Types.TimeSeriesVector(Vector{DateTime}(sub.time), Float64.(sub.wspd) .* u"m/s",
                                          stn, (source="NDBC",)))
    end
    isempty(raw) && error("ndbc_winds_ts: no station produced a usable wind-speed series")
    series = [s for s in raw]  # narrows Vector{Any} -> concrete Vector{TimeSeriesVector{Q}}
    return Types.TimeSeriesCollection(series, "ndbc_wspd", (;))
end

Base.close(r::NDBCLoader) = nothing
