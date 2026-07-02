# Unit-aware arithmetic operations for TimeSeriesVector/Matrix

import Base: +, -, *, /, ^
import Unitful

# Addition: ts1 + ts2 (units must match)
function +(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)
    if length(ts1.time) != length(ts2.time)
        error("Time series must have same length for addition")
    end
    if ts1.time != ts2.time
        error("Time series must have same time axis for addition")
    end
    return Types.TimeSeriesVector(
        time=ts1.time,
        value=ts1.value .+ ts2.value,  # Unitful handles unit checking
        name="$(ts1.name) + $(ts2.name)",
        metadata=merge(ts1.metadata, ts2.metadata)
    )
end

# Subtraction: ts1 - ts2 (units must match)
function -(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)
    if length(ts1.time) != length(ts2.time)
        error("Time series must have same length for subtraction")
    end
    if ts1.time != ts2.time
        error("Time series must have same time axis for subtraction")
    end
    return Types.TimeSeriesVector(
        time=ts1.time,
        value=ts1.value .- ts2.value,
        name="$(ts1.name) - $(ts2.name)",
        metadata=merge(ts1.metadata, ts2.metadata)
    )
end

# Scalar multiplication: ts * scalar (preserves units)
function *(ts::Types.TimeSeriesVector, scalar::Real)
    return Types.TimeSeriesVector(
        time=ts.time,
        value=ts.value .* scalar,
        name="$(ts.name) × $scalar",
        metadata=ts.metadata
    )
end

function *(scalar::Real, ts::Types.TimeSeriesVector)
    return ts * scalar
end

# Scalar division: ts / scalar (preserves units)
function /(ts::Types.TimeSeriesVector, scalar::Real)
    return Types.TimeSeriesVector(
        time=ts.time,
        value=ts.value ./ scalar,
        name="$(ts.name) / $scalar",
        metadata=ts.metadata
    )
end

# Unit conversion helper
function convert_units(ts::Types.TimeSeriesVector, target_unit)
    converted_value = Unitful.uconvert.(target_unit, ts.value)
    return Types.TimeSeriesVector(
        time=ts.time,
        value=converted_value,
        name=ts.name,
        metadata=ts.metadata
    )
end

# Unit stripping (for operations that need dimensionless values)
function strip_units(ts::Types.TimeSeriesVector)
    return Types.TimeSeriesVector(
        time=ts.time,
        value=Unitful.ustrip.(ts.value),
        name=ts.name,
        metadata=ts.metadata
    )
end

# Get unit information
function unit_of(ts::Types.TimeSeriesVector)
    if isempty(ts.value)
        return nothing
    end
    return Unitful.unit(first(ts.value))
end

# Scale to reference unit
function scale_to_unit(ts::Types.TimeSeriesVector, ref_unit)
    target_unit = Unitful.unit(1 * ref_unit)
    return convert_units(ts, target_unit)
end
