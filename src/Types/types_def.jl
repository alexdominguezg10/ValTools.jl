import Dates, Unitful

"""
    TimeSeriesVector{Q<:Number}

A single time series carrying its physical unit in the element type.

`Q` is normally a `Unitful.Quantity` (e.g. `Quantity{Float64, 𝐋 𝐓⁻¹, ...}`
for m/s), inferred automatically from `value`. Use a bare `Q<:Real` (no
units) for dimensionless series, e.g. after [`strip_units`](@ref).

# Fields
- `time`: sample timestamps
- `value`: samples, unit-tagged via Unitful (or bare `Float64` if dimensionless)
- `name`: human-readable label (e.g. "velocity_u", "model - obs")
- `metadata`: free-form `NamedTuple` for QC flags, source, provenance, etc.

# Example
```julia
using Unitful, Dates
t = Dates.now() .+ Dates.Second.(0:9)
ts = TimeSeriesVector(t, randn(10) * u"m/s", "velocity", (;))
```
"""
struct TimeSeriesVector{Q<:Number}
    time::Vector{Dates.DateTime}
    value::Vector{Q}
    name::String
    metadata::NamedTuple
end

"""
    TimeSeriesMatrix{Q<:Number}

Multiple time series sharing a common time axis, e.g. velocity at several
mooring depths or channels. Same unit-tracking convention as
[`TimeSeriesVector`](@ref): `Q` is inferred from `value`.

# Fields
- `time`: sample timestamps, shared across all channels
- `value`: `(n_time, n_channels)` matrix, unit-tagged via Unitful
- `channels`: per-column labels (e.g. depth or station names)
- `name`: human-readable label for the whole series
- `metadata`: free-form `NamedTuple`

# Example
```julia
using Unitful, Dates
t = Dates.now() .+ Dates.Second.(0:9)
tm = TimeSeriesMatrix(t, randn(10, 3) * u"m/s", ["10m", "50m", "100m"], "currents", (;))
```
"""
struct TimeSeriesMatrix{Q<:Number}
    time::Vector{Dates.DateTime}
    value::Matrix{Q}
    channels::Vector{String}
    name::String
    metadata::NamedTuple
end

"""
    ObsMetadata

Structured provenance/QC metadata for an observational dataset, kept
separate from [`TimeSeriesVector`](@ref)'s free-form `metadata` field for
cases where a fixed, typed schema is preferred.

# Fields
- `source`: dataset origin (e.g. "NDBC", "Argo")
- `units`: original units as reported by the source
- `qc_flags`: per-sample QC pass/fail
- `timestamp`: retrieval or processing time
- `instrument`: instrument or sensor name
- `location`: free-form `NamedTuple`, typically `(lat=..., lon=...)`
"""
struct ObsMetadata
    source::String
    units::String
    qc_flags::Vector{Bool}
    timestamp::Dates.DateTime
    instrument::String
    location::NamedTuple
end

"""
    SpectralEstimate{Q<:Number}

Power spectral density estimate with optional uncertainty diagnostics,
as produced by [`spectral_multitaper`](@ref).

`Q` follows the same unit convention as [`TimeSeriesVector`](@ref): pass
a `unit` keyword to the estimator to get `power` in physical units
(typically `unit^2`, i.e. a variance density), or leave it dimensionless.

# Fields
- `freq`: frequency vector (positive frequencies)
- `power`: power spectral density, unit-tagged via Unitful or bare `Float64`
- `ftest_pval`: F-test p-values for line components, or `nothing`
- `jkvar`: jackknifed variance (confidence intervals), or `nothing`
- `params`: free-form `NamedTuple` of estimation parameters (nw, ntapers, dt, ...)
"""
struct SpectralEstimate{Q<:Number}
    freq::Vector{Float64}
    power::Vector{Q}
    ftest_pval::Union{Vector{Float64}, Nothing}
    jkvar::Union{Vector{Float64}, Nothing}
    params::NamedTuple
end

"""
    RotarySpectralEstimate

Rotary (CW/CCW) spectral decomposition of a velocity time series `w = u + iv`,
as produced by [`rotary_spectrum`](@ref). Unifies what used to be two
independent implementations (`Metrics.rotary_spectrum` and `JLab.rotary`).

# Fields
- `freq`: positive frequency vector
- `S_ccw`: counter-clockwise (positive frequency) power spectral density
- `S_cw`: clockwise (negative frequency) power spectral density
- `ci_ccw`: `(lower, upper)` jackknife confidence bounds for `S_ccw`, or `nothing`
- `ci_cw`: `(lower, upper)` jackknife confidence bounds for `S_cw`, or `nothing`
- `rotary_coefficient`: per-frequency `(S_ccw .- S_cw) ./ (S_ccw .+ S_cw)`,
  in `[-1, 1]`; positive means CCW-dominant, negative means CW-dominant
- `ftest_ccw`: Thomson (1982) harmonic F-test p-values for a line component
  in the CCW branch, or `nothing` if `ftest=false` / `ntapers <= 1`. Small
  p-values (e.g. `< 0.05`) indicate the CCW power at that frequency is
  better explained by a coherent line component than by a stochastic
  background — i.e. a genuine rotary peak rather than noise.
- `ftest_cw`: same as `ftest_ccw`, for the CW branch. Computed natively on
  the (unevenly-spaced) negative-frequency FFT bins, then linearly
  interpolated onto `freq` for a common grid with `ftest_ccw` — an
  approximation near sharp features, consistent with how `S_cw` itself is
  already interpolated in this function.
- `params`: free-form `NamedTuple` of estimation parameters (nw, ntapers, dt, ...)

Supports tuple destructuring for backward compatibility with the old
`(freqs, S_ccw, S_cw)` return convention, e.g. `f, ccw, cw = spec`.
"""
struct RotarySpectralEstimate
    freq::Vector{Float64}
    S_ccw::Vector{Float64}
    S_cw::Vector{Float64}
    ci_ccw::Union{Tuple{Vector{Float64},Vector{Float64}}, Nothing}
    ci_cw::Union{Tuple{Vector{Float64},Vector{Float64}}, Nothing}
    rotary_coefficient::Vector{Float64}
    ftest_ccw::Union{Vector{Float64}, Nothing}
    ftest_cw::Union{Vector{Float64}, Nothing}
    params::NamedTuple
end

Base.iterate(r::RotarySpectralEstimate) = (r.freq, Val(:S_ccw))
Base.iterate(r::RotarySpectralEstimate, ::Val{:S_ccw}) = (r.S_ccw, Val(:S_cw))
Base.iterate(r::RotarySpectralEstimate, ::Val{:S_cw}) = (r.S_cw, Val(:done))
Base.iterate(r::RotarySpectralEstimate, ::Val{:done}) = nothing

"""
    RotaryCoherenceEstimate

Rotary cross-spectral coherence between two velocity time series
`w1 = u1 + iv1` and `w2 = u2 + iv2`, as produced by [`rotary_coherence`](@ref).
Follows the CW/CCW decomposition of Gonella (1972) applied separately to
each rotary component, per Mooers (1973) and Kundu (1976).

# Fields
- `freq`: positive frequency vector
- `coh_ccw`: magnitude-squared coherence of the CCW (positive frequency) components, in `[0, 1]`
- `coh_cw`: magnitude-squared coherence of the CW (negative frequency) components, in `[0, 1]`
- `phase_ccw`: cross-spectral phase (radians) of the CCW components
- `phase_cw`: cross-spectral phase (radians) of the CW components
- `significance_level`: critical coherence value above which `coh_ccw`/`coh_cw`
  is significantly nonzero at the requested confidence level, under the null
  hypothesis of no true coherence (frequency-independent; `NaN` if `ntapers <= 1`)
- `params`: free-form `NamedTuple` of estimation parameters (nw, ntapers, dt, ...)
"""
struct RotaryCoherenceEstimate
    freq::Vector{Float64}
    coh_ccw::Vector{Float64}
    coh_cw::Vector{Float64}
    phase_ccw::Vector{Float64}
    phase_cw::Vector{Float64}
    significance_level::Float64
    params::NamedTuple
end

"""
    CrossSpectralEstimate

Multitaper cross-spectral estimate between two ordinary real time series
`x` and `y`, as produced by [`cross_coherence`](@ref). The non-rotary
counterpart of [`RotaryCoherenceEstimate`](@ref) — no CW/CCW split, since
there's only one sense of "frequency" for a real-valued signal.

# Fields
- `freq`: positive frequency vector
- `cross_power`: complex cross-spectrum `Sxy(f)`, averaged over tapers
- `coherence`: magnitude-squared coherence `|Sxy|^2 / (Sxx*Syy)`, in `[0, 1]`
- `phase`: cross-spectral phase (radians)
- `significance_level`: critical coherence value above which `coherence`
  is significantly nonzero at the requested confidence level, under the
  null hypothesis of no true coherence (frequency-independent; `NaN` if
  `ntapers <= 1`)
- `params`: free-form `NamedTuple` of estimation parameters (nw, ntapers, dt, ...)
"""
struct CrossSpectralEstimate
    freq::Vector{Float64}
    cross_power::Vector{ComplexF64}
    coherence::Vector{Float64}
    phase::Vector{Float64}
    significance_level::Float64
    params::NamedTuple
end

"""
    ColocatedObservation

Result of pairing a model [`TimeSeriesVector`](@ref) with an observational
one at a common location, plus the validation metrics computed between them
(typically via [`validate`](@ref)).

# Fields
- `model`: model time series at the colocation point
- `obs`: observational time series at the colocation point
- `distance`: colocation distance (e.g. km from obs to nearest model point)
- `metrics`: free-form `NamedTuple`, typically `(rmse=..., correlation=..., skill=..., bias=...)`
"""
struct ColocatedObservation
    model::TimeSeriesVector
    obs::TimeSeriesVector
    distance::Float64
    metrics::NamedTuple
end

