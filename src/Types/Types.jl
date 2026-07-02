"""
    Types

Core type definitions for ValTools.jl.

All types are Unitful-aware (carry units) and use typed metadata.
Designed for polymorphic dispatch and clean oceanographic validation workflows.

## Types Exported
- TimeSeriesVector{T, U}: Single time series with units
- TimeSeriesMatrix{T, U}: Multiple time series, shared time
- SpectralEstimate{T, U}: Power spectral density with uncertainties
- ColocatedObservation: Model-obs colocation result
- ObsMetadata: Structured metadata (source, QC, location, etc.)
"""

module Types

using Dates
using Unitful

include("types_def.jl")

export TimeSeriesVector, TimeSeriesMatrix, SpectralEstimate, ColocatedObservation, ObsMetadata

end # module
