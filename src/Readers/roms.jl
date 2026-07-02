"""
    ROMSReader

Thin wrapper around `CROCOReader` for native ROMS output files.
Defaults `vtransform` to 1 (Song & Haidvogel) when the file attribute
is absent.
"""
function ROMSReader(files::AbstractString;
                    grid_file::Union{AbstractString, Nothing}=nothing,
                    vtransform::Union{Int, Nothing}=nothing,
                    decode_times::Bool=true)
    r = CROCOReader(files; grid_file=grid_file, vtransform=vtransform,
                    decode_times=decode_times)
    if vtransform === nothing && _get_attr(r.ds, "Vtransform", nothing) === nothing
        r.vtransform = 1
    end
    return r
end
