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
    TimeSeriesCollection{Q<:Number}

Multiple time series that do **not** share a common time axis — e.g. Argo
floats, NDBC stations, or RAFOS/CloudDrift drifters, each with its own
independent timestamps. This is the ragged counterpart to
[`TimeSeriesMatrix`](@ref) (which requires one shared time vector): use
`TimeSeriesMatrix` when every series is sampled on the same clock (mooring
depths, ensemble members), and `TimeSeriesCollection` when it isn't.
Forcing ragged data into `TimeSeriesMatrix` would require resampling each
series onto a common grid, silently altering the data just to fit the
container — this type exists specifically to avoid that.

# Fields
- `series`: the individual `TimeSeriesVector`s, each with its own `.time`
- `name`: human-readable label for the whole collection
- `metadata`: free-form `NamedTuple`

# Indexing
`length(c)`, `c[i]` (positional), `c["station_id"]` (by `series[i].name`,
errors if not found or not unique), and iteration are all supported.

# Example
```julia
using Unitful, Dates
t1 = Dates.now() .+ Dates.Second.(0:9)
t2 = Dates.now() .+ Dates.Hour.(0:5)  # different sampling, different length
c = TimeSeriesCollection([
    TimeSeriesVector(t1, randn(10) * u"m/s", "42001", (;)),
    TimeSeriesVector(t2, randn(6) * u"m/s", "42002", (;)),
], "ndbc_wspd", (;))
c["42001"]  # -> the first TimeSeriesVector
```
"""
struct TimeSeriesCollection{Q<:Number}
    series::Vector{TimeSeriesVector{Q}}
    name::String
    metadata::NamedTuple
end

Base.length(c::TimeSeriesCollection) = length(c.series)
Base.getindex(c::TimeSeriesCollection, i::Integer) = c.series[i]
Base.iterate(c::TimeSeriesCollection, state=1) =
    state > length(c.series) ? nothing : (c.series[state], state + 1)

function Base.getindex(c::TimeSeriesCollection, name::AbstractString)
    matches = findall(ts -> ts.name == name, c.series)
    isempty(matches) && error("No series named \"$name\" in TimeSeriesCollection \"$(c.name)\"")
    length(matches) > 1 && error("Multiple series named \"$name\" in TimeSeriesCollection \"$(c.name)\" — names are not unique")
    return c.series[only(matches)]
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
    EllipsePolarizationEstimate

Frequency-domain polarization decomposition of a bivariate (u,v) velocity
spectral matrix, as produced by [`ellipse_polarization`](@ref). Port of
jLab's `jSpectral/polparams.m` + `jSpectral/specdiag.m` (Lilly), applied to
the multitaper spectral matrix `S = [[Sxx, Sxy]; [conj(Sxy), Syy]]` built
from the same per-taper machinery as [`CrossSpectralEstimate`](@ref).

Unlike [`RotarySpectralEstimate`](@ref) (which decomposes into CW/CCW
rotary power) this diagonalizes the full Hermitian spectral matrix into an
elliptical eigenbasis — major/minor axis power and orientation — and
separately reports Cartesian polarization ratios. `imag(beta)` is the same
physical quantity as `RotarySpectralEstimate.rotary_coefficient` derived a
different way (spectral-matrix eigendecomposition vs. direct CW/CCW power
ratio); the two should agree, which is a useful independent correctness
check on both implementations.

# Fields
- `freq`: positive frequency vector
- `d1`: major-axis eigen-power spectrum (`d1 >= d2` by construction)
- `d2`: minor-axis eigen-power spectrum
- `theta`: ellipse orientation angle (radians), from `specdiag`
- `nu`: secondary `specdiag` angle (relates to the circular/rotary phase of
  the eigenbasis; `nu = 0` identically for a real (non-rotary) spectral
  matrix, since `nu` requires a nonzero imaginary cross-term)
- `P`: total polarization, `(d1-d2)/(d1+d2)`, in `[0, 1]`. `P = 0` for a
  circular (unpolarized) signal, `P = 1` for a purely linear (rectilinear)
  one.
- `alpha`: Cartesian anisotropy `(Sxx-Syy)/(Sxx+Syy)`, in `[-1, 1]`
- `beta`: complex `2*Sxy/(Sxx+Syy)`; `real(beta)` reflects the linear u-v
  correlation orientation, `imag(beta)` is the rotary sense (matches
  `RotarySpectralEstimate.rotary_coefficient`, see above)
- `ci_d1`, `ci_d2`: `(lower, upper)` jackknife confidence bounds for
  `d1`/`d2`, or `nothing`
- `ci_P`: `(lower, upper)` jackknife confidence bounds for `P`, or
  `nothing`. Orientation (`theta`) confidence intervals are not provided —
  jackknifing an angle needs circular-statistics care near the wraparound
  boundary that a plain linear jackknife doesn't handle correctly, and
  that hasn't been implemented yet.
- `params`: free-form `NamedTuple` of estimation parameters (nw, ntapers, dt_hours, ...)
"""
struct EllipsePolarizationEstimate
    freq::Vector{Float64}
    d1::Vector{Float64}
    d2::Vector{Float64}
    theta::Vector{Float64}
    nu::Vector{Float64}
    P::Vector{Float64}
    alpha::Vector{Float64}
    beta::Vector{ComplexF64}
    ci_d1::Union{Tuple{Vector{Float64},Vector{Float64}}, Nothing}
    ci_d2::Union{Tuple{Vector{Float64},Vector{Float64}}, Nothing}
    ci_P::Union{Tuple{Vector{Float64},Vector{Float64}}, Nothing}
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

