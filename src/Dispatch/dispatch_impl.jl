# Polymorphic operations for Types

import Statistics: mean, std, var, cov

function mean(ts::Types.TimeSeriesVector)
    m = Statistics.mean(ts.value)
    return m
end

function mean(ts::Types.TimeSeriesMatrix)
    m = Statistics.mean(ts.value; dims=1)
    return m
end

function std(ts::Types.TimeSeriesVector)
    s = Statistics.std(ts.value)
    return s
end

function std(ts::Types.TimeSeriesMatrix)
    s = Statistics.std(ts.value; dims=1)
    return s
end

function var(ts::Types.TimeSeriesVector)
    v = Statistics.var(ts.value)
    return v
end

function var(ts::Types.TimeSeriesMatrix)
    v = Statistics.var(ts.value; dims=1)
    return v
end

function cov(ts::Types.TimeSeriesMatrix)
    c = Statistics.cov(ts.value)
    return c
end

# Validation metrics
function rmse(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector)
    diff = model.value .- obs.value
    return sqrt(Statistics.mean(diff .^ 2))
end

function correlation(ts1::Types.TimeSeriesVector, ts2::Types.TimeSeriesVector)
    return Statistics.cor(Unitful.ustrip.(ts1.value), Unitful.ustrip.(ts2.value))
end

function skill_score(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector; ref=nothing)
    mse = Statistics.mean((Unitful.ustrip.(model.value) .- Unitful.ustrip.(obs.value)) .^ 2)
    if isnothing(ref)
        ref_var = Statistics.var(Unitful.ustrip.(obs.value))
    else
        ref_var = ref
    end
    return 1.0 - mse / ref_var
end

function validate(model::Types.TimeSeriesVector, obs::Types.TimeSeriesVector)
    return (
        rmse = rmse(model, obs),
        correlation = correlation(model, obs),
        skill = skill_score(model, obs),
        bias = Statistics.mean(Unitful.ustrip.(model.value) .- Unitful.ustrip.(obs.value))
    )
end
