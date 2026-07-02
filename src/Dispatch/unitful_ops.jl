# Unit-aware arithmetic operations for TimeSeriesVector/Matrix

import Base: +, -, *, /, ^
import Unitful

"""
    ts1 + ts2

Elementwise sum of two [`Types.TimeSeriesVector`](@ref)s sharing the same
time axis. Dimensions must be compatible (Unitful raises a
`DimensionError` otherwise; compatible-but-different units are converted
automatically). Errors if lengths or timestamps differ. `name` and
`metadata` are combined (`"a + b"`, `merge(meta1, meta2)`).
"""
function +(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)
    if length(ts1.time) != length(ts2.time)
        error("Time series must have same length for addition")
    end
    if ts1.time != ts2.time
        error("Time series must have same time axis for addition")
    end
    return Types.TimeSeriesVector(
        ts1.time,
        ts1.value .+ ts2.value,  # Unitful handles unit checking
        "$(ts1.name) + $(ts2.name)",
        merge(ts1.metadata, ts2.metadata)
    )
end

"""
    ts1 - ts2

Elementwise difference of two [`Types.TimeSeriesVector`](@ref)s. Same
requirements and semantics as [`+`](@ref) (shared time axis, compatible
units).
"""
function -(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)
    if length(ts1.time) != length(ts2.time)
        error("Time series must have same length for subtraction")
    end
    if ts1.time != ts2.time
        error("Time series must have same time axis for subtraction")
    end
    return Types.TimeSeriesVector(
        ts1.time,
        ts1.value .- ts2.value,
        "$(ts1.name) - $(ts2.name)",
        merge(ts1.metadata, ts2.metadata)
    )
end

"""
    ts * scalar
    scalar * ts

Scale a [`Types.TimeSeriesVector`](@ref) by a real scalar. Preserves the
unit of `ts`.
"""
function *(ts::Types.TimeSeriesVector, scalar::Real)
    return Types.TimeSeriesVector(
        ts.time,
        ts.value .* scalar,
        "$(ts.name) × $scalar",
        ts.metadata
    )
end

function *(scalar::Real, ts::Types.TimeSeriesVector)
    return ts * scalar
end

"""
    ts / scalar

Divide a [`Types.TimeSeriesVector`](@ref) by a real scalar. Preserves the
unit of `ts`.
"""
function /(ts::Types.TimeSeriesVector, scalar::Real)
    return Types.TimeSeriesVector(
        ts.time,
        ts.value ./ scalar,
        "$(ts.name) / $scalar",
        ts.metadata
    )
end

"""
    convert_units(ts::Types.TimeSeriesVector, target_unit)

Convert `ts.value` to `target_unit` via `Unitful.uconvert`, e.g.
`convert_units(ts, u"cm/s")`. Errors if `target_unit` has incompatible
dimensions.
"""
function convert_units(ts::Types.TimeSeriesVector, target_unit)
    converted_value = Unitful.uconvert.(target_unit, ts.value)
    return Types.TimeSeriesVector(
        ts.time,
        converted_value,
        ts.name,
        ts.metadata
    )
end

"""
    strip_units(ts::Types.TimeSeriesVector)

Return a copy of `ts` with `Unitful.ustrip` applied to every value,
producing a bare `Float64` series. Use when passing data to code that
doesn't understand `Unitful.Quantity` (e.g. [`correlation`](@ref)-style
comparisons already do this internally).
"""
function strip_units(ts::Types.TimeSeriesVector)
    return Types.TimeSeriesVector(
        ts.time,
        Unitful.ustrip.(ts.value),
        ts.name,
        ts.metadata
    )
end

"""
    unit_of(ts::Types.TimeSeriesVector)

The `Unitful` unit of `ts.value`'s first element, or `nothing` if `ts` is
empty.
"""
function unit_of(ts::Types.TimeSeriesVector)
    if isempty(ts.value)
        return nothing
    end
    return Unitful.unit(first(ts.value))
end

"""
    scale_to_unit(ts::Types.TimeSeriesVector, ref_unit)

Convert `ts` to whatever unit `ref_unit` carries, e.g.
`scale_to_unit(ts, 1.0u"cm/s")` converts to cm/s. Convenience wrapper
around [`convert_units`](@ref) for when you have a reference quantity
rather than a bare unit.
"""
function scale_to_unit(ts::Types.TimeSeriesVector, ref_unit)
    target_unit = Unitful.unit(1 * ref_unit)
    return convert_units(ts, target_unit)
end
