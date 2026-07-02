module Readers

using NCDatasets
using Glob
using Dates
using Statistics

include("sigma.jl")
include("croco.jl")
include("roms.jl")
include("nemo.jl")

export CROCOReader, ROMSReader, NEMOReader
export ssh, sst, sss, temperature, salinity, velocities, z_levels
export build_stretching, sigma_to_z, interp_z

end # module
