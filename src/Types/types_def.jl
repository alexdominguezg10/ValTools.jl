import Dates, Unitful

struct TimeSeriesVector{Q<:Number}
    time::Vector{Dates.DateTime}
    value::Vector{Q}
    name::String
    metadata::NamedTuple
end

struct TimeSeriesMatrix{Q<:Number}
    time::Vector{Dates.DateTime}
    value::Matrix{Q}
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

struct SpectralEstimate{Q<:Number}
    freq::Vector{Float64}
    power::Vector{Q}
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

