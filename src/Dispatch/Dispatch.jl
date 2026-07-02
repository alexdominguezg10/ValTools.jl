"""
    Dispatch

Polymorphic, unit-aware methods for ValTools' [`Types`](@ref) (see
`dispatch_impl.jl` and `unitful_ops.jl`).

Enables elegant operations:
- `mean(ts::TimeSeriesVector)`, `std`, `var` → preserve units
- `cov(ts::TimeSeriesMatrix)` → covariance across channels
- `rmse`, `correlation`, `skill_score`, `validate(model, obs)` → validation metrics
- `ts1 + ts2`, `ts1 - ts2`, `ts * scalar`, `ts / scalar` → unit-checked arithmetic
- `convert_units`, `scale_to_unit`, `strip_units`, `unit_of` → unit utilities

All methods work via multiple dispatch on type signature. The arithmetic
operators extend `Base` and are available globally once ValTools is
loaded; the named functions (`rmse`, `convert_units`, etc.) require
`using ValTools.Dispatch` since they are not re-exported at the top level.
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
