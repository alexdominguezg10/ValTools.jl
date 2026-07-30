module Observations

using NCDatasets
using DataFrames
using Dates
using Statistics
using Glob
using LinearAlgebra
using Interpolations
using NearestNeighbors
using Unitful
using ..Types

include("_utils.jl")
include("argo.jl")
include("duacs.jl")
include("swot.jl")
include("ndbc.jl")
include("rafos.jl")
include("ies.jl")
include("mooring.jl")
include("thermistor.jl")
include("gem.jl")
include("ghrsst.jl")
include("goflow.jl")
include("canek_section.jl")
include("clouddrift.jl")
include("glider.jl")

export ArgoLoader, argo_temperature, argo_salinity, argo_pressure, argo_to_dataframe
export DUACSLoader, duacs_ssh, duacs_sla, duacs_geostrophic_velocity
export SWOTLoader, swot_ssh, swot_sla, swot_to_points
export NDBCLoader, ndbc_station, ndbc_waves, ndbc_winds, ndbc_to_dataframe, ndbc_winds_ts
export RAFOSLoader, rafos_positions, rafos_velocity_estimates, rafos_to_dataframe
export IESLoader, ies_travel_time, ies_ssh_anomaly, ies_to_dataframe, ies_travel_time_ts
export MooringCurrentLoader, mooring_current_profiles, mooring_speed, mooring_direction,
       mooring_variance_ellipse, mooring_progressive_vector, mooring_current_ts
export ThermistorLoader, thermistor_temperature, thermistor_isotherm_depth, thermistor_heat_content
export GEMBuilder, gem_from_argo, gem_fit!, gem_tau_to_profiles, gem_model_tau,
       gem_save, gem_load, sound_speed_chen_millero
export GHRSSTLoader, ghrsst_sst
export GOFLOWLoader, goflow_u, goflow_v, goflow_speed, goflow_vorticity
export CANEKSectionLoader, canek_to_dataset, canek_transport
export CloudDriftLoader, clouddrift_trajectory, clouddrift_n_trajectories, clouddrift_to_dataframe
export GliderLoader, glider_profiles, glider_section, glider_to_dataframe

end # module
