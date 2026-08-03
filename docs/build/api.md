# API Reference

## Core Types

### Time Series Objects
```@docs
ValTools.TimeSeriesVector
ValTools.TimeSeriesMatrix
ValTools.TimeSeriesCollection
```

### Spectral Estimates
```@docs
ValTools.SpectralEstimate
ValTools.RotarySpectralEstimate
ValTools.RotaryCoherenceEstimate
ValTools.CrossSpectralEstimate
ValTools.EllipsePolarizationEstimate
```

### Wavelet Analysis
```@docs
ValTools.WaveletTransform
```

## Multitaper Spectral Analysis

```@docs
ValTools.spectral_multitaper
ValTools.rotary_spectrum
ValTools.rotary_coherence
ValTools.cross_coherence
ValTools.cross_spectrum
ValTools.ellipse_polarization
```

## Wavelet Analysis

### Continuous Wavelet Transform
```@docs
ValTools.wavetrans
ValTools.wavetrans_batch
```

### Ridge Analysis
```@docs
ValTools.ridgemap
ValTools.wavelet_significance
ValTools.ridge_significant
```

### Instantaneous Moments
```@docs
ValTools.tiredecode
ValTools.instmom
ValTools.jointmom
```

### Rotary Wavelet Transform
```@docs
ValTools.rotary_wavetrans
ValTools.rotary_ridge
```

## Spectral Utilities

```@docs
ValTools.morsewave
ValTools.morsewave_freq
ValTools.sleptap
ValTools.mspec
```

## Time Series Filtering & Manipulation

```@docs
ValTools.bandpass
ValTools.highpass
ValTools.lowpass
ValTools.detrend
ValTools.fillgaps
ValTools.hilbert
```

## Polarization Analysis

```@docs
ValTools.ellipsefit
ValTools.rotary
ValTools.msvd
```

## Model Readers

### CROCO/ROMS/NEMO
```@docs
ValTools.CROCOReader
ValTools.ROMSReader
ValTools.NEMOReader
ValTools.ssh
ValTools.sst
ValTools.sss
ValTools.temperature
ValTools.salinity
ValTools.velocities
ValTools.z_levels
```

### Coordinate Transforms
```@docs
ValTools.sigma_to_z
ValTools.interp_z
ValTools.build_stretching
```

## Observation Loaders

### Profile Data (Argo, etc.)
```@docs
ValTools.ArgoLoader
ValTools.argo_temperature
ValTools.argo_salinity
ValTools.argo_pressure
ValTools.argo_to_dataframe
```

### Satellite SSH/SLA
```@docs
ValTools.DUACSLoader
ValTools.duacs_ssh
ValTools.duacs_sla
ValTools.duacs_geostrophic_velocity
ValTools.SWOTLoader
ValTools.swot_ssh
ValTools.swot_sla
```

### Surface Winds & Waves
```@docs
ValTools.NDBCLoader
ValTools.ndbc_station
ValTools.ndbc_waves
ValTools.ndbc_winds
ValTools.ndbc_winds_ts
ValTools.ndbc_to_dataframe
```

### Lagrangian (Drifters)
```@docs
ValTools.RAFOSLoader
ValTools.rafos_positions
ValTools.rafos_velocity_estimates
ValTools.rafos_to_dataframe
ValTools.CloudDriftLoader
ValTools.clouddrift_trajectory
ValTools.clouddrift_n_trajectories
ValTools.clouddrift_to_dataframe
```

### Acoustic Tomography
```@docs
ValTools.IESLoader
ValTools.ies_travel_time
ValTools.ies_ssh_anomaly
ValTools.ies_to_dataframe
ValTools.ies_travel_time_ts
```

### Moorings
```@docs
ValTools.MooringCurrentLoader
ValTools.mooring_current_profiles
ValTools.mooring_speed
ValTools.mooring_direction
ValTools.mooring_variance_ellipse
ValTools.mooring_progressive_vector
ValTools.mooring_current_ts
ValTools.ThermistorLoader
ValTools.thermistor_temperature
ValTools.thermistor_isotherm_depth
ValTools.thermistor_heat_content
```

### Specialized
```@docs
ValTools.GEMBuilder
ValTools.gem_from_argo
ValTools.gem_fit!
ValTools.gem_tau_to_profiles
ValTools.gem_save
ValTools.gem_load
ValTools.sound_speed_chen_millero
ValTools.GHRSSTLoader
ValTools.ghrsst_sst
ValTools.GOFLOWLoader
ValTools.goflow_u
ValTools.goflow_v
ValTools.goflow_speed
ValTools.goflow_vorticity
ValTools.CANEKSectionLoader
ValTools.canek_to_dataset
ValTools.canek_transport
ValTools.GliderLoader
ValTools.glider_profiles
ValTools.glider_section
ValTools.glider_to_dataframe
```

## Colocation & Matching

```@docs
ValTools.colocate_model_obs
ValTools.colocate_model_grid
ValTools.colocate_model_trajectory
ValTools.colocate_model_mooring
ValTools.colocate_model_section
```

## Mooring Dynamics

```@docs
ValTools.MooringKnockdownModel
ValTools.fit_knockdown
ValTools.delta_z
ValTools.horizontal_excursion
ValTools.virtual_mooring
ValTools.project_to_fixed_depths
```

## Validation Metrics

```@docs
ValTools.compute_metrics
ValTools.taylor_stats
ValTools.bootstrap_metrics
ValTools.metrics_by_group
ValTools.current_ellipse_metrics
```

## 2-D Spectral Analysis

```@docs
ValTools.isotropic_2d_spectrum
ValTools.cross_spectrum_kx_ky
ValTools.detrend_2d_linear
ValTools.alongtrack_wavenumber_spectrum
```

## Plotting (Dispatch-Based)

### Time Series
```@docs
ValTools.plot_timeseries
ValTools.plot_timeseries_comparison
```

### Spectral
```@docs
ValTools.plot_spectrum
ValTools.plot_rotary_spectrum
ValTools.plot_rotary_coherence
ValTools.plot_cross_spectrum
```

### Polarization
```@docs
ValTools.plot_ellipse_polarization
```

### Colocation
```@docs
ValTools.plot_colocation
ValTools.taylor_diagram
```

### Maps & Comparisons
```@docs
ValTools.plot_comparison_map
```

### Wind & Flow
```@docs
ValTools.plot_wind_rose
ValTools.plot_wind_rose_comparison
ValTools.plot_streamlines
ValTools.plot_flow
ValTools.plot_field_panel
```

### Animation
```@docs
ValTools.animate_field_realtime
```

### Visualization
```@docs
ValTools.lic_texture
ValTools.plot_lic
```

## Validation & Conversion

```@docs
ValTools.pressure_to_depth
ValTools.depth_to_pressure
ValTools.validate_model_spectra
ValTools.kinetic_energy_budget
```

## GPU Functions

```@docs
ValTools.sigma_to_z_gpu
ValTools.interp_z_gpu
ValTools.lic_texture_gpu
```
