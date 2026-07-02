module Colocation

using NearestNeighbors
using Interpolations
using DataFrames
using Dates
using Statistics
using LinearAlgebra

include("interpolate.jl")
include("mooring_dynamics.jl")

export colocate_model_obs, colocate_model_grid,
       colocate_model_trajectory, colocate_model_mooring,
       colocate_model_section
export pressure_to_depth, depth_to_pressure
export MooringKnockdownModel, fit_knockdown,
       delta_z, horizontal_excursion, theta
export virtual_mooring, project_to_fixed_depths

end # module
