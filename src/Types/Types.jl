"""
    Types

Core type definitions for ValTools.jl.

All types are Unitful-aware (carry units via a type parameter `Q<:Number`,
normally a `Unitful.Quantity`) and use typed or free-form metadata.
Designed for polymorphic dispatch (see [`Dispatch`](@ref)) and clean
oceanographic validation workflows.

## Types Exported
- `TimeSeriesVector{Q<:Number}`: Single time series, unit-tagged via `Q`
- `TimeSeriesMatrix{Q<:Number}`: Multiple time series, shared time axis
- `SpectralEstimate{Q<:Number}`: Power spectral density with uncertainties
- `RotarySpectralEstimate`: Rotary (CW/CCW) spectral decomposition with uncertainties
- `RotaryCoherenceEstimate`: Rotary cross-spectral coherence between two velocity series
- `CrossSpectralEstimate`: Multitaper cross-spectrum/coherence between two real time series
- `ColocatedObservation`: Model-obs colocation result
- `ObsMetadata`: Structured metadata (source, QC, location, etc.)
- `WaveletTransform{Q<:Number, N}`: Continuous wavelet transform coefficients + parameters

`Q` is inferred automatically from the `value`/`power` array passed to
the constructor — pass Unitful-quantity data (e.g. `randn(10) * u"m/s"`)
for unit tracking, or bare `Float64` data for dimensionless series (e.g.
after [`Dispatch.strip_units`](@ref)).
"""

module Types

using Dates
using Unitful

include("types_def.jl")

export TimeSeriesVector, TimeSeriesMatrix, TimeSeriesCollection, SpectralEstimate,
       RotarySpectralEstimate, RotaryCoherenceEstimate, CrossSpectralEstimate,
       EllipsePolarizationEstimate, ColocatedObservation, ObsMetadata,
       WaveletTransform

end # module
