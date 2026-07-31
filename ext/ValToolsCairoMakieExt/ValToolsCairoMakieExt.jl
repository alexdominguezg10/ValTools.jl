module ValToolsCairoMakieExt

using ValTools
using CairoMakie
using Statistics
using Random
using Dates
using Unitful

include("taylor.jl")
include("maps.jl")
include("timeseries.jl")
include("wind_rose.jl")
include("lic.jl")
include("streamlines.jl")
include("ocean_panel.jl")
include("typed_plots.jl")

end # module
