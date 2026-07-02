import Dates, Unitful

struct TimeSeriesVector{T, U}
    time::Vector{Dates.DateTime}
    value::Vector{Unitful.Quantity{T, U}}
    name::String
    metadata::NamedTuple
end

struct TimeSeriesMatrix{T, U}
    time::Vector{Dates.DateTime}
    value::Matrix{Unitful.Quantity{T, U}}
    channels::Vector{String}
    name::String
    metadata::NamedTuple
end

struct ObsMetadata
    source::String
    units::String
    qc_flags::Vector{Bool}
    timestamp::Dates.DateTime
    instrument::String
    location::NamedTuple
end

struct SpectralEstimate{T, U}
    freq::Vector{Float64}
    power::Vector{Unitful.Quantity{T, U}}
    ftest_pval::Union{Vector{Float64}, Nothing}
    jkvar::Union{Vector{Float64}, Nothing}
    params::NamedTuple
end

struct ColocatedObservation
    model::TimeSeriesVector
    obs::TimeSeriesVector
    distance::Float64
    metrics::NamedTuple
end

# Convenience constructors
TimeSeriesVector(time, value, name) = TimeSeriesVector(time, value, name, NamedTuple())
TimeSeriesMatrix(time, value, channels, name) = TimeSeriesMatrix(time, value, channels, name, NamedTuple())
TimeSeriesVector(; time, value, name, metadata=NamedTuple()) = TimeSeriesVector(time, value, name, metadata)
TimeSeriesMatrix(; time, value, channels, name, metadata=NamedTuple()) = TimeSeriesMatrix(time, value, channels, name, metadata)
