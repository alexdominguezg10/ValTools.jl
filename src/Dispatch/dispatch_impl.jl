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
