function _obs_open_nc(path::AbstractString)
    expanded = glob(path)
    if length(expanded) > 1
        sort!(expanded)
        return NCDataset(expanded, "r")
    elseif length(expanded) == 1
        return NCDataset(expanded[1], "r")
    else
        return NCDataset(path, "r")
    end
end

function _first_present_varname(ds, candidates::Tuple)
    for name in candidates
        haskey(ds, name) && return name
    end
    return nothing
end

function _obs_read_var_f64(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            return Float64.(Array(ds[name]))
        end
    end
    return nothing
end

function _obs_read_time(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            raw = Array(ds[name])
            if eltype(raw) <: Dates.AbstractDateTime
                return Vector{DateTime}(raw)
            elseif eltype(raw) <: Dates.AbstractTime
                return Vector{DateTime}(raw)
            else
                return raw
            end
        end
    end
    return DateTime[]
end

function _obs_try_scalar(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            val = Array(ds[name])
            return Float64(val isa AbstractArray ? val[1] : val)
        end
    end
    return nothing
end

function _obs_read_qc(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            raw = Array(ds[name])
            if eltype(raw) <: AbstractChar || eltype(raw) <: AbstractString
                return parse.(Int, string.(raw))
            else
                return Int.(raw)
            end
        end
    end
    return nothing
end

# CF/COARDS `units` attribute strings mapped to Unitful units. Deliberately
# conservative — no attempt to cover every CF string, only the ones actually
# seen in the loaders this package reads. An unrecognized string is never
# guessed at; `_var_units` falls back to the caller-supplied default and
# still returns the raw string so callers can preserve it in metadata.
const _CF_UNIT_MAP = Dict{String,Unitful.Units}(
    "m s-1" => u"m/s", "m/s" => u"m/s", "meters per second" => u"m/s",
    "cm s-1" => u"cm/s", "cm/s" => u"cm/s",
    "degree_Celsius" => u"°C", "degrees_C" => u"°C", "degC" => u"°C", "Celsius" => u"°C",
    "K" => u"K", "kelvin" => u"K",
    "m" => u"m", "meter" => u"m", "meters" => u"m",
    "Pa" => u"Pa", "hPa" => u"hPa",
    "s" => u"s", "second" => u"s", "seconds" => u"s",
    "hours" => u"hr", "hour" => u"hr",
    "degree" => u"°", "degrees" => u"°",
)

"""
    _var_units(ds, varname, fallback)

Read the CF `units` attribute of `ds[varname]` and map it to a Unitful
unit via `_CF_UNIT_MAP`. Returns `(unit, raw_string)`: `unit` is `fallback`
when the variable has no `units` attribute, or when the attribute string
isn't in the map (this never guesses a unit from the variable name or
typical values — only from an explicit, recognized attribute string);
`raw_string` is the attribute's literal text (or `nothing` if absent),
returned even when unrecognized so callers can still record it.
"""
function _var_units(ds, varname::AbstractString, fallback::Unitful.Units)
    haskey(ds, varname) || return fallback, nothing
    attrib = ds[varname].attrib
    haskey(attrib, "units") || return fallback, nothing
    raw = string(attrib["units"])
    mapped = get(_CF_UNIT_MAP, raw, nothing)
    return (mapped === nothing ? fallback : mapped), raw
end

# Guard used by every typed (`_ts`) accessor before constructing a
# TimeSeriesVector/Matrix/Collection. `_obs_read_time` can return an empty
# `DateTime[]` (no time variable found at all) or a raw, undecoded array
# (a time variable was found but wasn't already a recognized date/time
# type) -- both must fail loudly here rather than propagate into a typed
# struct that promises `Vector{DateTime}`.
function _require_datetime(time, context::AbstractString)
    isempty(time) && error("$context: no time variable found")
    eltype(time) <: Dates.AbstractDateTime ||
        error("$context: time variable found but not decoded to DateTime (got element type $(eltype(time))) -- refusing to guess a time unit/epoch")
    return Vector{DateTime}(time)
end

function _apply_qc_mask!(data::AbstractArray, qc::AbstractArray)
    for i in eachindex(data)
        if checkbounds(Bool, qc, i)
            flag = qc[i]
            if flag != 1 && flag != 2
                data[i] = NaN
            end
        end
    end
    return data
end
