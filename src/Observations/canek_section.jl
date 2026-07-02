"""
    CANEKSectionLoader

Loader for CANEK Yucatan Channel velocity section data (MATLAB HDF5 .mat files).

Requires HDF5.jl for reading MATLAB v7.3 format files. If not available,
provides a stub that errors with a helpful message.
"""
mutable struct CANEKSectionLoader
    lon::Vector{Float64}
    depth::Vector{Float64}
    time::Vector{DateTime}
    velocity::Array{Float64, 3}  # (nt, nx, nz)
    cell_area::Union{Float64, Nothing}
end

function CANEKSectionLoader(velfile::AbstractString;
                            errfile::Union{AbstractString, Nothing}=nothing)
    _check_hdf5_available()

    data = _read_canek_mat(velfile)

    cell_area = nothing
    if errfile !== nothing
        err_data = _read_canek_mat(errfile)
        if haskey(err_data, :da)
            cell_area = Float64(err_data[:da][1])
        end
    end

    return CANEKSectionLoader(data[:lon], data[:depth], data[:time],
                              data[:velocity], cell_area)
end

function canek_to_dataset(r::CANEKSectionLoader)
    return (v=r.velocity, lon=r.lon, depth=r.depth, time=r.time)
end

function canek_transport(r::CANEKSectionLoader;
                         cell_area::Union{Real, Nothing}=nothing)
    ca = cell_area !== nothing ? Float64(cell_area) :
         r.cell_area !== nothing ? r.cell_area :
         error("cell_area not available. Provide it explicitly or load the error file.")

    nt = size(r.velocity, 1)
    transport = Vector{Float64}(undef, nt)
    for it in 1:nt
        v_slice = r.velocity[it, :, :]
        valid = isfinite.(v_slice)
        transport[it] = sum(v_slice[valid]) * ca / 1e6  # Sverdrups
    end
    return (data=transport, time=r.time)
end

const _MATLAB_EPOCH_OFFSET = 719529  # days between 0000-01-01 and 1970-01-01

function _matlab_datenum_to_datetime(datenum::Real)
    days_since_epoch = datenum - _MATLAB_EPOCH_OFFSET
    ms = round(Int64, days_since_epoch * 86400 * 1000)
    return DateTime(1970) + Millisecond(ms)
end

function _check_hdf5_available()
    try
        @eval import HDF5
    catch
        error("HDF5.jl is required for reading CANEK .mat files. Install with: using Pkg; Pkg.add(\"HDF5\")")
    end
end

function _read_canek_mat(filepath::String)
    HDF5_mod = @eval import HDF5; HDF5

    h5 = HDF5_mod.h5open(filepath, "r")
    result = Dict{Symbol, Any}()

    if haskey(h5, "X")
        X = Float64.(read(h5["X"]))
        result[:lon] = vec(X[:, 1])
    end
    if haskey(h5, "Z")
        Z = Float64.(read(h5["Z"]))
        result[:depth] = vec(Z[1, :])
    end

    if haskey(h5, "ttd")
        ttd = Float64.(read(h5["ttd"]))
        result[:time] = [_matlab_datenum_to_datetime(t) for t in vec(ttd)]
    end

    if haskey(h5, "Veldy") && haskey(h5, "celda") && haskey(h5, "mask")
        Veldy = Float64.(read(h5["Veldy"]))
        celda = Int.(read(h5["celda"]))
        mask_raw = Float64.(read(h5["mask"]))
        nx, nz = size(mask_raw)
        nt = size(Veldy, 1)

        velocity = fill(NaN, nt, nx, nz)
        celda_vec = vec(celda)
        for ic in 1:length(celda_vec)
            ci = celda_vec[ic]
            ci < 1 && continue
            ix = mod1(ci, nx)
            iz = div(ci - 1, nx) + 1
            iz > nz && continue
            for it in 1:nt
                velocity[it, ix, iz] = Veldy[it, ic]
            end
        end
        result[:velocity] = velocity
    end

    if haskey(h5, "da")
        result[:da] = Float64.(read(h5["da"]))
    end

    close(h5)
    return result
end

Base.close(r::CANEKSectionLoader) = nothing
