"""
    Dispatch

Polymorphic methods for ValTools types.

Enables elegant operations:
- mean(ts::TimeSeriesVector) → preserves units
- cov(ts::TimeSeriesMatrix) → covariance with units
- validate(model, obs) → comprehensive metrics

All methods work via multiple dispatch on type signature.
"""

module Dispatch

using Statistics
using Unitful
using Dates
using ..Types

include("dispatch_impl.jl")
include("unitful_ops.jl")

export mean, std, var, cov, rmse, correlation, skill_score, validate
export convert_units, strip_units, unit_of, scale_to_unit

end # module
