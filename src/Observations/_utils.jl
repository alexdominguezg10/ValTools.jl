function _obs_open_nc(path::AbstractString)
    expanded = glob(path)
    if length(expanded) > 1
        sort!(expanded)
        return NCDataset(expanded, "r")
    elseif length(expanded) == 1
        return NCDataset(expanded[1], "r")
    else
        return NCDataset(path, "r")
    end
end

function _obs_read_var_f64(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            return Float64.(Array(ds[name]))
        end
    end
    return nothing
end

function _obs_read_time(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            raw = Array(ds[name])
            if eltype(raw) <: Dates.AbstractDateTime
                return Vector{DateTime}(raw)
            elseif eltype(raw) <: Dates.AbstractTime
                return Vector{DateTime}(raw)
            else
                return raw
            end
        end
    end
    return DateTime[]
end

function _obs_try_scalar(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            val = Array(ds[name])
            return Float64(val isa AbstractArray ? val[1] : val)
        end
    end
    return nothing
end

function _obs_read_qc(ds, candidates::Tuple)
    for name in candidates
        if haskey(ds, name)
            raw = Array(ds[name])
            if eltype(raw) <: AbstractChar || eltype(raw) <: AbstractString
                return parse.(Int, string.(raw))
            else
                return Int.(raw)
            end
        end
    end
    return nothing
end

function _apply_qc_mask!(data::AbstractArray, qc::AbstractArray)
    for i in eachindex(data)
        if checkbounds(Bool, qc, i)
            flag = qc[i]
            if flag != 1 && flag != 2
                data[i] = NaN
            end
        end
    end
    return data
end
