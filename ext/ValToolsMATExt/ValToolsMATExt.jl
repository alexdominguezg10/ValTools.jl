"""
ValToolsMATExt — MAT.jl-backed loaders for plain MATLAB (v5/v7, non-HDF5)
data files. Currently just `ANCMooringLoader`
(`src/Observations/anc_gomw_mooring.jl`); see that file's docstring for
the format itself.

Loaded automatically when both ValTools and MAT are in the environment
(weakdep, matching how ValToolsCairoMakieExt/ValToolsMultitaperExt work).
"""
module ValToolsMATExt

using ValTools
using ValTools.Observations: ANCMooringLoader, _parse_anc_filename, _matlab_datenum_to_datetime
using MAT
using Statistics: mean
using Dates: DateTime

function ValTools.Observations.ANCMooringLoader(filepath::AbstractString;
                                                 looking::Union{Symbol, Nothing}=nothing,
                                                 site::Union{String, Nothing}=nothing)
    meta = _parse_anc_filename(filepath)
    look = looking !== nothing ? looking :
           (meta !== nothing && meta.looking !== nothing) ? meta.looking : :down
    look in (:down, :up) || error("looking must be :down or :up, got $look")

    data = matread(String(filepath))
    haskey(data, "u") && haskey(data, "v") || error("$filepath: expected variables 'u' and 'v' not found")

    u = Float64.(data["u"])
    v = Float64.(data["v"])
    w = haskey(data, "w") ? Float64.(data["w"]) : nothing
    bins = haskey(data, "bins") ? vec(Float64.(data["bins"])) : collect(1.0:size(u, 2))
    instrument_depth = haskey(data, "profinst") ? vec(Float64.(data["profinst"])) :
                        error("$filepath: expected variable 'profinst' (instrument depth) not found")
    temp = haskey(data, "Temp") ? vec(Float64.(data["Temp"])) : nothing

    haskey(data, "jd") || error("$filepath: expected variable 'jd' (MATLAB datenum time) not found")
    time = DateTime[_matlab_datenum_to_datetime(t) for t in vec(Float64.(data["jd"]))]

    sign = look === :down ? 1.0 : -1.0
    depths = mean(instrument_depth) .+ sign .* bins

    lon_val = haskey(data, "Lon") ? Float64(data["Lon"]) : error("$filepath: expected variable 'Lon' not found")
    lat_val = haskey(data, "Lat") ? Float64(data["Lat"]) : error("$filepath: expected variable 'Lat' not found")
    site_val = site !== nothing ? site :
               meta !== nothing ? meta.site :
               haskey(data, "nam") ? String(data["nam"]) : "mooring"

    return ANCMooringLoader(String(filepath), site_val, lon_val, lat_val, time, u, v, w,
                            bins, instrument_depth, depths, temp, meta)
end

end # module
