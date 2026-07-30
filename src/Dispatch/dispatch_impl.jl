# Polymorphic operations for Types

import Statistics: mean, std, var, cov

"""
    mean(ts::Types.TimeSeriesVector)

Mean of `ts.value`, preserving its unit (returns a `Unitful.Quantity` if
`ts` carries one, a bare number otherwise).
"""
function mean(ts::Types.TimeSeriesVector)
    m = Statistics.mean(ts.value)
    return m
end

"""
    mean(ts::Types.TimeSeriesMatrix)

Per-channel mean of `ts.value` (`dims=1`), preserving units. Returns a
`(1, n_channels)` matrix.
"""
function mean(ts::Types.TimeSeriesMatrix)
    m = Statistics.mean(ts.value; dims=1)
    return m
end

"""
    std(ts::Types.TimeSeriesVector)

Standard deviation of `ts.value`, preserving its unit.
"""
function std(ts::Types.TimeSeriesVector)
    s = Statistics.std(ts.value)
    return s
end

"""
    std(ts::Types.TimeSeriesMatrix)

Per-channel standard deviation of `ts.value` (`dims=1`), preserving units.
"""
function std(ts::Types.TimeSeriesMatrix)
    s = Statistics.std(ts.value; dims=1)
    return s
end

"""
    var(ts::Types.TimeSeriesVector)

Variance of `ts.value`. Units are squared relative to `ts` (e.g. m/s → m²/s²).
"""
function var(ts::Types.TimeSeriesVector)
    v = Statistics.var(ts.value)
    return v
end

"""
    var(ts::Types.TimeSeriesMatrix)

Per-channel variance of `ts.value` (`dims=1`). Units are squared relative to `ts`.
"""
function var(ts::Types.TimeSeriesMatrix)
    v = Statistics.var(ts.value; dims=1)
    return v
end

"""
    cov(ts::Types.TimeSeriesMatrix)

Covariance matrix (`n_channels × n_channels`) across the channels of `ts.value`.
"""
function cov(ts::Types.TimeSeriesMatrix)
    c = Statistics.cov(ts.value)
    return c
end

# ── Validation metrics ─────────────────────────────────────────────

"""
    rmse(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector)

Root-mean-square error between `model` and `obs`, requiring compatible
units (Unitful checks/broadcasts this on subtraction). Result carries the
same unit as the inputs.
"""
function rmse(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector)
    diff = model.value .- obs.value
    return sqrt(Statistics.mean(diff .^ 2))
end

# Strip units from `a` and `b`, converting `b` to `a`'s unit first. Plain
# `ustrip.(a), ustrip.(b)` is wrong whenever `a` and `b` carry different but
# compatible units (e.g. m/s vs cm/s): each value would be stripped in its
# own scale, silently comparing numbers that aren't on the same footing.
function _stripped_common_unit(a, b)
    ua = Unitful.unit(first(a))
    return Unitful.ustrip.(a), Unitful.ustrip.(Unitful.uconvert.(ua, b))
end

"""
    correlation(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)

Pearson correlation coefficient between `ts1.value` and `ts2.value`.
Units are converted to a common scale and stripped before comparison
(correlation is dimensionless and invariant to a shared unit rescaling).
"""
function correlation(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)
    a, b = _stripped_common_unit(ts1.value, ts2.value)
    return Statistics.cor(a, b)
end

"""
    skill_score(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector; ref=nothing)

Murphy (1988) skill score: `1 - mse(model, obs) / ref_var`, where `ref_var`
defaults to `var(obs)` (unitless) unless `ref` is supplied explicitly.
A score of 1 is a perfect match; 0 means no better than the reference
variance; negative means worse.
"""
function skill_score(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector; ref=nothing)
    m, o = _stripped_common_unit(model.value, obs.value)
    mse = Statistics.mean((m .- o) .^ 2)
    if isnothing(ref)
        ref_var = Statistics.var(o)
    else
        ref_var = ref
    end
    return 1.0 - mse / ref_var
end

"""
    validate(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector)

Bundle of standard validation metrics between `model` and `obs`, returned
as a `NamedTuple`: `(rmse, correlation, skill, bias)`. `rmse` carries units;
`correlation` and `skill` are dimensionless ratios (unaffected by which
unit was used internally). `bias = mean(model - obs)` is a bare number
expressed in **`model`'s unit** (obs is converted to match before
subtracting) — if `model` and `obs` carry different-but-compatible units
(e.g. cm/s vs m/s), remember to interpret `bias` against `model`'s unit,
not `obs`'s. Suitable for storing directly in [`ColocatedObservation`](@ref)'s
`metrics` field.
"""
function validate(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector)
    m, o = _stripped_common_unit(model.value, obs.value)
    return (
        rmse = rmse(model, obs),
        correlation = correlation(model, obs),
        skill = skill_score(model, obs),
        bias = Statistics.mean(m .- o)
    )
end

# ── Multi-series validation (TimeSeriesMatrix / TimeSeriesCollection) ──
#
# Both containers hold several series identified by NAME (channel names on
# TimeSeriesMatrix, each series' own .name on TimeSeriesCollection). Every
# method below matches `model` against `obs` BY NAME, not by column/vector
# position -- comparing entry i of one against entry i of the other is
# silently wrong whenever the two were built or ordered independently
# (e.g. two station lists that don't happen to share an iteration order).

function _matched_names(a_names::AbstractVector{<:AbstractString}, b_names::AbstractVector{<:AbstractString})
    length(unique(a_names)) == length(a_names) ||
        error("duplicate names in first argument ($(a_names)) -- name-based matching is ambiguous")
    length(unique(b_names)) == length(b_names) ||
        error("duplicate names in second argument ($(b_names)) -- name-based matching is ambiguous")
    common = intersect(a_names, b_names)
    isempty(common) && error("no names in common between $(a_names) and $(b_names)")
    return common
end

function _channel_series(tm::Types.TimeSeriesMatrix, name::AbstractString)
    idx = findfirst(==(name), tm.channels)
    idx === nothing && error("channel \"$name\" not found in $(tm.channels)")
    return Types.TimeSeriesVector(tm.time, tm.value[:, idx], name, tm.metadata)
end

_collection_names(c::Types.TimeSeriesCollection) = [ts.name for ts in c.series]

"""
    mean/std/var(c::Types.TimeSeriesCollection)

Per-series mean/std/var, as a `Dict` keyed by each series' `.name`.
"""
mean(c::Types.TimeSeriesCollection) = Dict(ts.name => mean(ts) for ts in c.series)
std(c::Types.TimeSeriesCollection) = Dict(ts.name => std(ts) for ts in c.series)
var(c::Types.TimeSeriesCollection) = Dict(ts.name => var(ts) for ts in c.series)

for fn in (:rmse, :correlation)
    @eval begin
        """
            $($fn)(model::Types.TimeSeriesMatrix, obs::Types.TimeSeriesMatrix)

        Per-channel $($fn), matched by channel name. Returns a `Dict`
        keyed by channel name.
        """
        function $fn(model::Types.TimeSeriesMatrix, obs::Types.TimeSeriesMatrix)
            names = _matched_names(model.channels, obs.channels)
            return Dict(name => $fn(_channel_series(model, name), _channel_series(obs, name)) for name in names)
        end

        """
            $($fn)(model::Types.TimeSeriesCollection, obs::Types.TimeSeriesCollection)

        Per-series $($fn), matched by series name (not position). Returns
        a `Dict` keyed by series name.
        """
        function $fn(model::Types.TimeSeriesCollection, obs::Types.TimeSeriesCollection)
            names = _matched_names(_collection_names(model), _collection_names(obs))
            return Dict(name => $fn(model[name], obs[name]) for name in names)
        end
    end
end

function skill_score(model::Types.TimeSeriesMatrix, obs::Types.TimeSeriesMatrix; ref=nothing)
    names = _matched_names(model.channels, obs.channels)
    return Dict(name => skill_score(_channel_series(model, name), _channel_series(obs, name); ref=ref) for name in names)
end

function skill_score(model::Types.TimeSeriesCollection, obs::Types.TimeSeriesCollection; ref=nothing)
    names = _matched_names(_collection_names(model), _collection_names(obs))
    return Dict(name => skill_score(model[name], obs[name]; ref=ref) for name in names)
end

"""
    validate(model::Types.TimeSeriesMatrix, obs::Types.TimeSeriesMatrix)

Per-channel `(rmse, correlation, skill, bias)`, matched by channel name
(not column position). Returns a `Dict{String,NamedTuple}` keyed by
channel name; each value is the same bundle as the single-series
[`validate`](@ref).
"""
function validate(model::Types.TimeSeriesMatrix, obs::Types.TimeSeriesMatrix)
    names = _matched_names(model.channels, obs.channels)
    return Dict(name => validate(_channel_series(model, name), _channel_series(obs, name)) for name in names)
end

"""
    validate(model::Types.TimeSeriesCollection, obs::Types.TimeSeriesCollection)

Per-series `(rmse, correlation, skill, bias)`, matched by each series'
`.name` (not position -- ragged collections built independently, e.g.
two station lists, have no reliable shared order). Returns a
`Dict{String,NamedTuple}` keyed by series name.
"""
function validate(model::Types.TimeSeriesCollection, obs::Types.TimeSeriesCollection)
    names = _matched_names(_collection_names(model), _collection_names(obs))
    return Dict(name => validate(model[name], obs[name]) for name in names)
end
