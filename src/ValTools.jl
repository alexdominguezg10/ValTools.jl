module ValTools

using LinearAlgebra
using Statistics
using Random
using Dates
using Unitful

# ── Types submodule (Core type definitions) ────────────────────────
include("Types/Types.jl")
using .Types
export TimeSeriesVector, TimeSeriesMatrix, TimeSeriesCollection, SpectralEstimate,
       RotarySpectralEstimate, RotaryCoherenceEstimate, CrossSpectralEstimate,
       EllipsePolarizationEstimate, ColocatedObservation, ObsMetadata,
       WaveletTransform

# ── Dispatch submodule (Polymorphic methods) ──────────────────────
include("Dispatch/Dispatch.jl")
using .Dispatch

# ── Readers submodule ──────────────────────────────────────────────
include("Readers/Readers.jl")
using .Readers
export CROCOReader, ROMSReader, NEMOReader
export ssh, sst, sss, temperature, salinity, velocities, z_levels
export build_stretching, sigma_to_z, interp_z

# ── Metrics submodule ──────────────────────────────────────────────
include("Metrics/Metrics.jl")
using .Metrics
export compute_metrics, taylor_stats, bootstrap_metrics,
       metrics_by_group, current_ellipse_metrics
export rotary_spectrum, rotary_coherence, cross_coherence, ellipse_polarization,
       alongtrack_wavenumber_spectrum,
       isotropic_2d_spectrum, cross_spectrum_kx_ky, detrend_2d_linear,
       msvd, resample_uniform
export HelmholtzSpectra, helmholtz_decomposition, wave_vortex_decomposition
export velocity_structure_functions, helmholtz_structure_function

# ── Colocation submodule ──────────────────────────────────────────
include("Colocation/Colocation.jl")
using .Colocation
export colocate_model_obs, colocate_model_grid,
       colocate_model_trajectory, colocate_model_mooring,
       colocate_model_section
export pressure_to_depth, depth_to_pressure
export MooringKnockdownModel, fit_knockdown,
       delta_z, horizontal_excursion
export virtual_mooring, project_to_fixed_depths

# ── Observations submodule ────────────────────────────────────────
include("Observations/Observations.jl")
using .Observations
export ArgoLoader, argo_temperature, argo_salinity, argo_pressure, argo_to_dataframe
export DUACSLoader, duacs_ssh, duacs_sla, duacs_geostrophic_velocity
export SWOTLoader, swot_ssh, swot_sla, swot_to_points
export NDBCLoader, ndbc_station, ndbc_waves, ndbc_winds, ndbc_to_dataframe, ndbc_winds_ts
export RAFOSLoader, rafos_positions, rafos_velocity_estimates, rafos_to_dataframe
export IESLoader, ies_travel_time, ies_ssh_anomaly, ies_to_dataframe, ies_travel_time_ts
export MooringCurrentLoader, mooring_current_profiles, mooring_speed, mooring_direction,
       mooring_variance_ellipse, mooring_progressive_vector, mooring_current_ts
export ThermistorLoader, thermistor_temperature, thermistor_isotherm_depth, thermistor_heat_content
export GEMBuilder, gem_from_argo, gem_fit!, gem_tau_to_profiles,
       gem_save, gem_load, sound_speed_chen_millero
export GHRSSTLoader, ghrsst_sst
export GOFLOWLoader, goflow_u, goflow_v, goflow_speed, goflow_vorticity
export CANEKSectionLoader, canek_to_dataset, canek_transport
export CloudDriftLoader, clouddrift_trajectory, clouddrift_n_trajectories, clouddrift_to_dataframe
export GliderLoader, glider_profiles, glider_section, glider_to_dataframe

# ── JLab submodule (Wavelet & Spectral Analysis) ─────────────────────
include("JLab/JLab.jl")
using .JLab
export wavetrans, wavetrans_batch, tiredecode, transmax,
       morsewave, morsewave_freq,
       ridgemap, wavelet_significance, ridge_significant,
       mspec, sleptap,
       ellipsefit, rotary, rotary_wavetrans, rotary_ridge,
       bandpass, highpass, lowpass, detrend, fillgaps, hilbert,
       validate_model_spectra, kinetic_energy_budget, fit_spectral_slope

# ── GPU function stubs (implemented by ValToolsCUDAExt) ───────────
function sigma_to_z_gpu end
function interp_z_gpu end
function lic_texture_gpu end

export sigma_to_z_gpu, interp_z_gpu, lic_texture_gpu

# ── Plot function stubs (implemented by ValToolsCairoMakieExt) ────
function taylor_diagram end
function plot_comparison_map end
function plot_timeseries_comparison end
function plot_wind_rose end
function plot_wind_rose_comparison end
function lic_texture end
function plot_lic end
function plot_streamlines end
function plot_flow end
function plot_field_panel end
function animate_field_realtime end
function reference_slope! end

# Typed-struct plotting (Stage 4a): one function name per plot kind,
# dispatching via multiple dispatch across the Types.jl structs it applies to.
function plot_timeseries end
function plot_spectrum end
function plot_rotary_spectrum end
function plot_rotary_coherence end
function plot_cross_spectrum end
function plot_ellipse_polarization end
function plot_colocation end

export taylor_diagram, plot_comparison_map, plot_timeseries_comparison,
       plot_wind_rose, plot_wind_rose_comparison, lic_texture, plot_lic,
       plot_streamlines, plot_flow, plot_field_panel, animate_field_realtime,
       reference_slope!,
       plot_timeseries, plot_spectrum, plot_rotary_spectrum, plot_rotary_coherence,
       plot_cross_spectrum, plot_ellipse_polarization, plot_colocation

end # module
