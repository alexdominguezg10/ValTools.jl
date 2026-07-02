module ValToolsCairoMakieExt

using ValTools
using CairoMakie
using Statistics
using Random

include("taylor.jl")
include("maps.jl")
include("timeseries.jl")
include("wind_rose.jl")
include("lic.jl")
include("streamlines.jl")
include("ocean_panel.jl")

end # module
