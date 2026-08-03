# ValTools.jl — Reference Manual

**Version 0.1.0** | Julia port of the Python `valtools` validation package

ValTools.jl provides a unified toolkit for validating ocean model output
(CROCO, ROMS, NEMO) against observations. It handles model I/O, observation
loading, spatial/temporal colocation, statistical metrics, spectral analysis,
and publication-quality plotting.

```julia
using ValTools            # core functionality
using CairoMakie          # activates the plot extension (optional)
```

---

## Table of Contents

1. [Model Readers](#1-model-readers)
2. [Sigma Coordinates](#2-sigma-coordinates)
3. [Metrics — Statistics](#3-metrics--statistics)
4. [Metrics — Spectral](#4-metrics--spectral)
5. [Colocation Engine](#5-colocation-engine)
6. [Mooring Dynamics](#6-mooring-dynamics)
7. [Observation Loaders](#7-observation-loaders)
8. [Plots (CairoMakie Extension)](#8-plots-cairomakie-extension)
9. [Dependencies](#9-dependencies)
12. [Type System & Polymorphic Dispatch (Types/Dispatch)](#12-type-system--polymorphic-dispatch-typesdispatch)
13. [Examples Gallery](#13-examples-gallery)

---

## 1. Model Readers

All readers use NCDatasets.jl for lazy NetCDF access. Glob patterns are
supported for multi-file datasets.

### CROCOReader

```julia
r = CROCOReader("croco_avg_*.nc"; grid_file="croco_grd.nc", vtransform=2)
```

| Field | Type | Description |
|-------|------|-------------|
| `r.h` | `Matrix{Float64}` | Bathymetry (ny, nx) [m, positive] |
| `r.lon` | `Array{Float64}` | Longitude at rho-points |
| `r.lat` | `Array{Float64}` | Latitude at rho-points |
| `r.mask` | `Matrix{Float64}` or `nothing` | Land-sea mask |
| `r.sc_r` | `Vector{Float64}` | Sigma at rho-points (N) |
| `r.Cs_r` | `Vector{Float64}` | Stretching function (N) |
| `r.hc` | `Float64` | Critical depth [m] |
| `r.vtransform` | `Int` | 1 or 2 |

**Functions** — all return `NamedTuple` with `(data, lon, lat, time)`:

```julia
ssh(r; apply_mask=true)                    # → ζ [m], (ny, nx, nt)
sst(r; apply_mask=true)                    # → SST [°C]
sss(r; apply_mask=true)                    # → SSS [PSU]
temperature(r; depths=nothing)             # → T [°C], full or at target depths
salinity(r; depths=nothing)                # → S [PSU]
velocities(r; depths=nothing)              # → (u, v, lon, lat, time)
z_levels(r; time_idx=nothing)              # → z [m, negative down]
```

When `depths` is given (e.g. `[-50.0, -200.0]`, negative-down), the field is
interpolated from sigma levels to those z-depths.

### ROMSReader

```julia
r = ROMSReader("roms_his_*.nc")
```

Identical to `CROCOReader`; defaults `vtransform` to 1 when the file attribute
is absent.

### NEMOReader

```julia
r = NEMOReader("NEMO_GoM_*.nc")
```

Reads z-coordinate NEMO output. Same function API (`ssh`, `sst`, `temperature`,
`velocities`, etc.). Auto-detects variable names (CF-compliant + legacy NEMO).

**Depth convention**: NEMO depths are positive-down in files; NEMOReader negates
them internally so `depths` arguments use negative-down consistently.

**Close readers** when done: `close(r)`.

---

## 2. Sigma Coordinates

```julia
sc_r, Cs_r = build_stretching(N, theta_s, theta_b; vstretching=4)
```

| Arg | Description |
|-----|-------------|
| `N` | Number of sigma layers |
| `theta_s` | Surface control (0–10) |
| `theta_b` | Bottom control (0–4) |
| `vstretching` | 1 = Song & Haidvogel 1994; 4 = Shchepetkin 2010 |

```julia
z = sigma_to_z(h, zeta, sc_r, Cs_r, hc; vtransform=2)
```

- `h`: bathymetry `(ny, nx)` [m, positive]
- `zeta`: SSH `(ny, nx)` or `(ny, nx, nt)` [m]
- Returns `z` `(ny, nx, N)` or `(ny, nx, N, nt)` [m, negative down]

```julia
out = interp_z(var, z, depths)
```

- `var`, `z`: `(ny, nx, N, nt)` or `(ny, nx, N)`
- `depths`: vector of target depths [m, negative down]
- Returns `(ny, nx, nz, nt)` or `(ny, nx, nz)`

---

## 3. Metrics — Statistics

### compute_metrics

```julia
m = compute_metrics(obs, model; weights=nothing, prefix="")
```

Returns `Dict{String,Float64}` with keys:

| Key | Definition |
|-----|------------|
| `n` | Number of valid pairs |
| `bias` | mean(model) − mean(obs) |
| `mae` | mean(\|model − obs\|) |
| `rmse` | √mean((model − obs)²) |
| `rmse_unbiased` | √(rmse² − bias²) |
| `std_obs` | σ(obs) |
| `std_model` | σ(model) |
| `correlation` | Pearson r |
| `scatter_index` | rmse_unbiased / σ(obs) |
| `skill_score` | 1 − rmse² / σ²(obs) (Murphy 1988) |
| `nse` | Nash–Sutcliffe efficiency |

NaN pairs are dropped. Weighted metrics supported via `weights`.

### taylor_stats

```julia
ts = taylor_stats(obs, model)
```

Returns `Dict` with `std_ref`, `std_test`, `correlation`, `rms_diff`.

### bootstrap_metrics

```julia
bm = bootstrap_metrics(obs, model; n_boot=1000, ci=0.95, seed=42)
```

Returns `Dict{String, NamedTuple{(:mean,:lo,:hi)}}` — percentile-method
confidence intervals.

### metrics_by_group

```julia
df = metrics_by_group(df, :obs_col, :model_col, :group_col)
```

Returns a `DataFrame` with metrics per group (e.g. by season, station, depth).

### current_ellipse_metrics

```julia
cem = current_ellipse_metrics(obs_u, obs_v, model_u, model_v)
```

Returns `Dict` with variance ellipse parameters (`semi_major`, `semi_minor`,
`inclination`), EKE, MKE, per-component RMSE/bias/correlation, and speed
metrics.

---

## 4. Metrics — Spectral

### rotary_spectrum

```julia
spec = rotary_spectrum(u, v; dt_hours=1.0, detrend="linear", nw=4.0, ntapers=0, ci=true, confidence=0.95)
```

Multitaper (DPSS) rotary CW/CCW decomposition, requires `using Multitaper`
(implemented in `ValToolsMultitaperExt`). Returns a
[`Types.RotarySpectralEstimate`](#12-type-system--polymorphic-dispatch-typesdispatch)
with `freq`, `S_ccw` (counter-clockwise PSD, positive rotation), `S_cw`
(clockwise PSD, near-inertial in NH), jackknife confidence intervals
(`ci_ccw`/`ci_cw`), a Thomson (1982) harmonic F-test for a line component
in each branch (`ftest_ccw`/`ftest_cw`, p-values — small means "genuine
peak, not noise"; pass `ftest=false` to skip), and `rotary_coefficient`.
Also supports tuple destructuring `freqs, S_ccw, S_cw = rotary_spectrum(u, v)`
for backward compatibility. See [examples/inertial_oscillation.jl](examples/inertial_oscillation.jl).

### rotary_coherence

```julia
rc = rotary_coherence(u1, v1, u2, v2; dt_hours=1.0, nw=4.0, confidence=0.95)
```

Multitaper CW/CCW cross-coherence between two velocity series (e.g. model
vs. observed currents), applying the CW/CCW split to each series
separately. Returns a `Types.RotaryCoherenceEstimate` with `coh_ccw`,
`coh_cw`, `phase_ccw`, `phase_cw`, and `significance_level` (critical
coherence value above which the null hypothesis of no true coherence is
rejected). Requires `using Multitaper`.

### cross_coherence

```julia
cc = cross_coherence(x, y; dt=1.0, nw=4.0, confidence=0.95)
```

The non-rotary counterpart of `rotary_coherence`, for two ordinary real
time series (e.g. two SST records, a wind stress and a current
component) — no CW/CCW split. Returns a `Types.CrossSpectralEstimate`
with `freq`, `cross_power`, `coherence`, `phase`, `significance_level`.
Requires `using Multitaper`.

### alongtrack_wavenumber_spectrum

```julia
k, psd = alongtrack_wavenumber_spectrum(ssh_track, dx_km; detrend="linear", window="hann")
```

Welch's method (segment ≤ 256, 50% overlap). Returns wavenumbers [cycles/km].

### isotropic_2d_spectrum

```julia
k_iso, psd_iso = isotropic_2d_spectrum(field, dx_km, dy_km; detrend="linear", window="hann", n_bins=nothing)
```

2-D FFT → radially averaged. Input: `(ny, nx)` field on regular grid.

### cross_spectrum_kx_ky

```julia
k_iso, coherence, phase, cross_psd = cross_spectrum_kx_ky(field1, field2, dx_km, dy_km)
```

Isotropic coherence and phase of two co-located 2-D fields.

### detrend_2d_linear

```julia
f_detrended = detrend_2d_linear(field)
```

Removes a least-squares 2-D plane fit.

---

## 5. Colocation Engine

All colocation functions accept both **regular grids** (1-D lon/lat) and
**curvilinear grids** (2-D lon/lat). Curvilinear grids use KDTree-based
inverse-distance weighting.

### colocate_model_obs

```julia
result = colocate_model_obs(field, model_lon, model_lat, model_times, obs_df;
                            lon_col=:lon, lat_col=:lat, time_col=:time,
                            max_dt_hours=12.0, max_dist_km=50.0,
                            method=:bilinear, out_col=:model_value)
```

- `field`: `(ny, nx, nt)` model data
- `obs_df`: DataFrame with lon, lat, time columns
- Returns a copy of `obs_df` with interpolated model values appended

### colocate_model_grid

```julia
result = colocate_model_grid(field, model_lon, model_lat, target_lon, target_lat; method=:bilinear)
```

Regrids a 2-D field to a target regular grid. Returns `(data, lon, lat)`.

### colocate_model_trajectory

```julia
result = colocate_model_trajectory(field, model_lon, model_lat, model_times, traj_df; ...)
```

Same API as `colocate_model_obs`, designed for RAFOS/GDP/Argo trajectories.

### colocate_model_mooring

```julia
result = colocate_model_mooring(field, model_lon, model_lat, model_times, lon, lat;
                                 max_dist_km=25.0, method=:bilinear)
```

Extracts a time series at a fixed location. Returns `(data, time)`.

### colocate_model_section

```julia
result = colocate_model_section(u_field, v_field, model_lon, model_lat, model_times,
                                 lon_section, lat_section; angle_deg=0.0, ...)
```

Velocity cross-section along a transect. Computes along-channel component:
`v_along = u·sin(θ) + v·cos(θ)`. Returns `(u, v, v_along, lon, lat, time)`.

---

## 6. Mooring Dynamics

### Pressure–Depth Conversion

```julia
depth = pressure_to_depth(pressure; latitude=nothing)   # [dbar] → [m, positive down]
press = depth_to_pressure(depth; latitude=nothing)       # [m] → [dbar]
```

UNESCO (Fofonoff & Millard 1983) when latitude given; constant factor
1.019716 otherwise. Inverse via Newton–Raphson (10 iterations).

### MooringKnockdownModel

Empirical knockdown: `Δz(U) = L(1 − cos(arctan(k·U²)))`.

```julia
m = MooringKnockdownModel(k=0.05, line_length=400.0)
dz = delta_z(m, speed)                  # vertical excursion [m]
dx = horizontal_excursion(m, speed)      # horizontal displacement [m]
θ  = theta(m, speed)                     # line angle [rad]
```

### fit_knockdown

```julia
m = fit_knockdown(speed, depth_actual, depth_nominal; line_length=nothing)
```

Fits `k` (and optionally `line_length`) from pressure-sensor records via
gradient descent. Needs ≥ 4 finite (speed, depth) pairs.

### virtual_mooring

```julia
result = virtual_mooring(reader, lon, lat, nominal_depths;
                          knockdown=nothing, depth_grid=nothing,
                          variables=(:temperature, :salinity, :u, :v),
                          max_dist_km=25.0)
```

Samples a model reader at (lon, lat) with realistic knockdown dynamics.
Returns NamedTuple with variables + `actual_depth`, `knockdown`,
`current_speed`, `nominal_depth`.

### project_to_fixed_depths

```julia
result = project_to_fixed_depths(time, depths, values, target_depths; extrapolate=false)
```

- `depths`: `(n_time, n_instruments)` actual depths [m, positive down]
- `values`: `Dict{Symbol, Matrix}` of variables to project
- Returns NamedTuple with projected variables + `n_instruments`, `time`, `depth`

---

## 7. Observation Loaders

All loaders use NCDatasets.jl for NetCDF, DataFrames.jl for tabular output,
and Glob.jl for file pattern matching.

### ArgoLoader

```julia
r = ArgoLoader("argo_profiles_*.nc"; bbox=(-98,-80,18,31), date_range=("2024-01-01","2024-12-31"))
T   = argo_temperature(r; good_qc=true)     # (N_PROF, N_LEVELS) [°C]
S   = argo_salinity(r; good_qc=true)        # [PSU]
P   = argo_pressure(r)                      # [dbar]
df  = argo_to_dataframe(r; good_qc=true)    # tidy format
```

QC: retains flags 1 (good) and 2 (probably good) by default.

### DUACSLoader

```julia
r = DUACSLoader("cmems_duacs_*.nc")
ssh_r = duacs_ssh(r)                         # ADT [m]
sla_r = duacs_sla(r)                         # SLA [m]
vel_r = duacs_geostrophic_velocity(r)        # (u, v) [m/s]
```

### SWOTLoader

```julia
r = SWOTLoader("swot_l3_*.nc")
ssh_r = swot_ssh(r)                          # SSH [m]
sla_r = swot_sla(r)                          # SLA [m]
df    = swot_to_points(r)                    # flattened DataFrame
```

### NDBCLoader

```julia
r = NDBCLoader("42001h2024.txt")
sub = ndbc_station(r, "42001")               # single station DataFrame
w   = ndbc_waves(r)                          # Dict{station => DataFrame(time, Hs)}
wnd = ndbc_winds(r)                          # Dict{station => DataFrame(time, wspd, wdir)}
df  = ndbc_to_dataframe(r; add_position=true)
```

Column mapping: WVHT→Hs, DPD→Tp, WSPD→wspd, WDIR→wdir, WTMP→wtmp, etc.
Built-in GoM station metadata for 42001, 42002, 42019, 42020, 42055, 42056.

### RAFOSLoader

```julia
r = RAFOSLoader("rafos_*.nc"; bbox=(-98,-80,18,31), pressure_range=(400,600))
pos = rafos_positions(r)                     # float_id, time, lon, lat, pressure, depth
vel = rafos_velocity_estimates(r; min_dt_hours=12, max_dt_hours=96)  # Lagrangian u, v [m/s]
df  = rafos_to_dataframe(r)
```

Depth conversion: pressure × 1.019716. Velocities via finite differences on
the sphere (R = 6371 km).

### IESLoader

```julia
r = IESLoader("ies_data.nc"; site="IES-A", depth=1000.0, lowpass_days=30.0)
tt  = ies_travel_time(r)                     # τ [s]
ssh = ies_ssh_anomaly(r; method="linear")    # SSH' ≈ −c̄²·τ'/(2Z) [m]
df  = ies_to_dataframe(r)
```

### MooringCurrentLoader

```julia
r = MooringCurrentLoader("adcp_mooring.nc"; site="M1", lon=-93.9, lat=22.0)
p   = mooring_current_profiles(r)            # (u, v, time, depths)
spd = mooring_speed(r)                       # |U| [m/s]
dir = mooring_direction(r)                   # [°T, 0=N clockwise]
ell = mooring_variance_ellipse(r)            # Dict: semi_major/minor, inclination, EKE, MKE
pvd = mooring_progressive_vector(r)          # DataFrame: time, dx_km, dy_km
```

### ThermistorLoader

```julia
r = ThermistorLoader("thermistors.nc"; site="T1", lon=-93.5, lat=22.0)
T   = thermistor_temperature(r)              # (data, time, depths)
iso = thermistor_isotherm_depth(r, 20.0)     # depth of 20°C isotherm vs time
Q   = thermistor_heat_content(r; z1=50, z2=500)  # ρ·c_p·∫T dz [J/m²]
```

### GEMBuilder

```julia
g = gem_from_argo(argo_loader; depth_grid=0:10:2000, site="GoM")
gem_fit!(g)                                  # fit T(τ,z), S(τ,z) splines
result = gem_tau_to_profiles(g, tau_series)  # τ(t) → T(z,t), S(z,t)
gem_save(g, "gem_table.nc")
g2 = gem_load("gem_table.nc")
```

Sound speed (Chen & Millero 1977):
```julia
c = sound_speed_chen_millero(T, S, P)        # [m/s]
```

### GHRSSTLoader

```julia
r = GHRSSTLoader("mur_sst_*.nc")
sst_r = ghrsst_sst(r; celsius=true)         # auto K→°C conversion
```

### GOFLOWLoader

```julia
r = GOFLOWLoader("goflow_predictions_*.nc")
u_r = goflow_u(r)                            # eastward [m/s]
v_r = goflow_v(r)                            # northward [m/s]
spd = goflow_speed(r)                        # |U| [m/s]
vor = goflow_vorticity(r)                    # ∂v/∂x − ∂u/∂y [1/s]
```

### CANEKSectionLoader

```julia
r = CANEKSectionLoader("canek_vel.mat"; errfile="canek_err.mat")
ds  = canek_to_dataset(r)                    # (v, lon, depth, time)
tr  = canek_transport(r; cell_area=nothing)  # volume transport [Sv]
```

Reads MATLAB v7.3 (HDF5) files. Requires HDF5.jl.

---

## 8. Plots (CairoMakie Extension)

Activated by `using CairoMakie` alongside `using ValTools`. All functions
return a `Makie.Figure`.

### taylor_diagram

```julia
fig = taylor_diagram(std_ref;
                      samples=[(std=0.9, corr=0.95, label="CROCO"),
                               (std=1.1, corr=0.88, label="NEMO")],
                      normalise=true, smin=0.0, smax=1.6, title="SSH")
save("taylor.png", fig)
```

Polar diagram: azimuth = arccos(correlation), radius = normalized σ.
CRMSE contour arcs drawn automatically.

### plot_comparison_map

```julia
fig = plot_comparison_map(obs_data, model_data, lon, lat;
                           title="SSH 2024-03-15", units="m",
                           vmin=-0.3, vmax=0.3)
```

Three panels: observations | model | difference. Auto color limits from data
percentiles.

### plot_timeseries_comparison

```julia
fig = plot_timeseries_comparison(
    Dict("DUACS" => (t_obs, v_obs)),
    Dict("CROCO" => (t_mod, v_mod));
    variable="SSH [m]", show_metrics=true)
```

Observations plotted solid, models dashed. RMSE/r annotated when
`show_metrics=true`.

### plot_wind_rose

```julia
fig = plot_wind_rose(u, v;
                      n_sectors=16, calm_threshold=0.5, title="ERA5 winds")
```

Meteorological convention: direction FROM which wind blows, 0° = North,
clockwise. Polar histogram with stacked speed bins.

### lic_texture / plot_lic

```julia
tex = lic_texture(u, v; length=30, seed=42, step=0.5)   # → Matrix{Float64} ∈ [0,1]
fig = plot_lic(u, v; field=sst_field, title="Surface currents")
```

Line Integral Convolution (Cabral & Leedom 1993). Optional scalar field
overlay with transparency.

---

## 9. GPU Acceleration (CUDA Extension)

Activated by `using CUDA` alongside `using ValTools`. Provides GPU-accelerated
versions of the most compute-intensive functions. Requires an NVIDIA GPU
(tested on H200).

```julia
using ValTools
using CUDA    # triggers the CUDA extension
```

### sigma_to_z_gpu

```julia
z = sigma_to_z_gpu(h, zeta, sc_r, Cs_r, hc; vtransform=2)
```

Same interface as `sigma_to_z`. Each (i, j, k, t) point runs as an
independent GPU thread. For a 400×500×32×100 grid (~640M points), expect
20-50× speedup over CPU.

### interp_z_gpu

```julia
out = interp_z_gpu(var, z, depths)
```

Same interface as `interp_z`. Each output point searches its sigma column
independently on the GPU. Accepts 3-D `(ny, nx, N)` or 4-D `(ny, nx, N, nt)`.

### lic_texture_gpu

```julia
tex = lic_texture_gpu(u, v; length=30, seed=42, step=0.5)
```

Same interface as `lic_texture`. Each pixel traces its own streamline on
the GPU. For a 1000×1000 velocity field, expect 50-100× speedup.

### Typical workflow

```julia
using ValTools, CUDA

r = CROCOReader("croco_avg_*.nc"; grid_file="croco_grd.nc")
zeta = Float64.(Array(r.ds["zeta"]))

# GPU sigma-to-z (fast on large grids)
z = sigma_to_z_gpu(r.h, zeta, r.sc_r, r.Cs_r, r.hc; vtransform=r.vtransform)

# GPU vertical interpolation
T = Float64.(Array(r.ds["temp"]))
T_at_depths = interp_z_gpu(T, z, [-50.0, -200.0, -500.0])

# GPU LIC for flow visualization
vel = velocities(r)
tex = lic_texture_gpu(vel.u[:,:,1], vel.v[:,:,1]; length=40)
```

Data is automatically transferred CPU→GPU→CPU. The GPU functions accept
regular Julia arrays — no manual `CuArray` wrapping needed.

### GPU wavelet transforms (JLab)

The CUDA extension also accelerates JLab wavelet transforms:

```julia
wt = wavetrans(signal; gpu=true)              # single signal
W  = wavetrans_batch(signals; gpu=true)       # batch of signals (biggest speedup)
```

---

## 10. Dependencies

| Package | Purpose |
|---------|---------|
| NCDatasets.jl | NetCDF I/O |
| DataFrames.jl | Tabular data |
| FFTW.jl | FFT for spectral analysis |
| Interpolations.jl | Regular-grid bilinear interpolation |
| NearestNeighbors.jl | KDTree for curvilinear grid colocation |
| Glob.jl | File pattern matching |
| CairoMakie.jl | Plotting (weak dependency, optional) |
| CUDA.jl | GPU acceleration (weak dependency, optional) |
| HDF5.jl | MATLAB .mat files (CANEK only, optional) |

### Array Convention

Julia is column-major. ValTools.jl uses `(ny, nx, ...)` ordering:
- 2-D fields: `(ny, nx)`
- 3-D with time: `(ny, nx, nt)`
- 3-D with depth: `(ny, nx, N)`
- 4-D: `(ny, nx, N, nt)`

This differs from the Python version which uses `(nt, N, ny, nx)`.
Depth convention: **negative downward** (same as Python).

---

## 11. Testing

### Run the full test suite

```bash
cd ~/ADominguez/CLAUDE_SPACE/ValTools.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

Or run the test file directly (slightly faster, skips creating a temp environment):

```bash
julia --project=. test/runtests.jl
```

### Test individual modules

**Readers + sigma coordinates:**

```julia
using ValTools
sc_r, Cs_r = build_stretching(32, 7.0, 0.0; vstretching=4)
h = 500.0 .* ones(3, 4); zeta = zeros(3, 4)
z = sigma_to_z(h, zeta, sc_r, Cs_r, 200.0)
println("sigma_to_z: ", size(z))   # (3, 4, 32)
```

**Metrics:**

```julia
using ValTools
obs = [1.0, 2.0, 3.0, 4.0, 5.0]
mod = [1.1, 2.2, 2.8, 4.1, 5.3]
m = compute_metrics(obs, mod)
println("RMSE = $(m["rmse"]), r = $(m["correlation"])")
```

**Colocation:**

```julia
using ValTools, DataFrames, Dates
field = randn(15, 20, 3)
lon = collect(range(-95, -90; length=20))
lat = collect(range(20, 25; length=15))
times = [DateTime(2025, 1, d) for d in 1:3]
obs = DataFrame(lon=[-92.5], lat=[22.5], time=[DateTime(2025, 1, 1, 6)])
result = colocate_model_obs(field, lon, lat, times, obs)
println(result)
```

**Plots (requires CairoMakie):**

```julia
using ValTools, CairoMakie
fig = taylor_diagram(1.0; samples=[(std=0.9, corr=0.95, label="CROCO")])
save(joinpath(pkgdir(ValTools), "test/figures/taylor_test.png"), fig)

u = [sin(π*i/20)*cos(π*j/25) for i in 1:40, j in 1:50]
v = [-cos(π*i/20)*sin(π*j/25) for i in 1:40, j in 1:50]
fig2 = plot_streamlines(Float64.(u), Float64.(v); density=1.0, title="Test")
save(joinpath(pkgdir(ValTools), "test/figures/streamlines_test.png"), fig2)
```

**GPU (requires CUDA + NVIDIA GPU):**

```julia
using ValTools, CUDA
h = 500.0 .* ones(400, 500)
zeta = randn(400, 500, 10)
sc_r, Cs_r = build_stretching(32, 7.0, 0.0)
@time z_cpu = sigma_to_z(h, zeta, sc_r, Cs_r, 200.0)
@time z_gpu = sigma_to_z_gpu(h, zeta, sc_r, Cs_r, 200.0)
println("Max diff: ", maximum(abs.(z_cpu .- z_gpu)))
```

### Test coverage summary

| Module | Tests | What is verified |
|--------|-------|-----------------|
| Readers / sigma | 17 | build_stretching, sigma_to_z, interp_z ranges and values |
| Metrics | 30 | compute_metrics, taylor_stats, bootstrap CI, ellipse, rotary/wavenumber/2D spectra |
| Colocation | 22 | Regular + curvilinear grids, mooring, section, pressure/depth, knockdown fit |
| Observations | 20 | Synthetic NetCDF round-trips for GHRSST, DUACS, thermistor, mooring, IES, GEM |
| Plots | 9 | Each plot function returns a `Figure` (taylor, maps, timeseries, LIC, streamlines, flow) |
| Types & Dispatch | 75 | Construction, unit-preserving statistics/validation, unit-aware arithmetic, `SpectralEstimate` integration (`test/types/`) |
| **Total** | **~195** | |

---

## 12. Type System & Polymorphic Dispatch (Types/Dispatch)

Concrete, Unitful-aware types replacing plain `NamedTuple` returns, with
multiple dispatch for statistics, validation, and arithmetic. `Q` (the
type parameter on `TimeSeriesVector`, `TimeSeriesMatrix`, and
`SpectralEstimate`) is inferred automatically from the data you pass in
— a `Unitful.Quantity` for unit-tagged series, or bare `Float64` for
dimensionless ones.

```julia
using ValTools, ValTools.Dispatch, Unitful, Dates

t = Dates.now() .+ Dates.Second.(0:99)
model = TimeSeriesVector(t, randn(100) * u"m/s" .+ 0.3u"m/s", "model", (;))
obs   = TimeSeriesVector(t, randn(100) * u"m/s", "obs", (;))

Dispatch.mean(model)           # preserves m/s
model + obs                    # unit-checked elementwise sum (errors on mismatched time axis)
Dispatch.rmse(model, obs)      # m/s
Dispatch.validate(model, obs)  # (rmse=..., correlation=..., skill=..., bias=...)
Dispatch.convert_units(model, u"cm/s")
```

### Types

- `TimeSeriesVector{Q<:Number}` / `TimeSeriesMatrix{Q<:Number}` — time
  series carrying units in the element type
- `SpectralEstimate{Q<:Number}` — returned by `spectral_multitaper`
  (JLab); pass `unit=u"m/s"` to get `power` in physical units (`unit^2`)
  instead of dimensionless
- `ObsMetadata`, `ColocatedObservation` — structured provenance and
  model-obs colocation results

### Dispatch — statistics & validation

Requires `using ValTools.Dispatch`. `mean`, `std`, `var`, `cov` dispatch
on `TimeSeriesVector`/`TimeSeriesMatrix` and preserve units; `rmse`,
`correlation`, `skill_score`, `validate` compare a model series against
an observed one (`correlation`/`skill_score` are unitless by
construction, `rmse` preserves the shared unit).

### Unit-aware arithmetic

`+`, `-`, `*`, `/` extend `Base` and work with a plain `using ValTools`
(no explicit `Dispatch` import needed). `convert_units`, `scale_to_unit`,
`strip_units`, and `unit_of` require `using ValTools.Dispatch`.
Dimensionally-incompatible operations raise `Unitful.DimensionError`;
mismatched time axes raise a plain `ErrorException`.

See `test/types/` for the full test suite (construction, unit-preserving
statistics, arithmetic error paths, and `SpectralEstimate` integration).

---

## 13. Examples Gallery

Runnable, self-contained scripts in [`examples/`](examples/) — each one
tells a small oceanographic story end to end: simulate something physical,
ask ValTools.jl a real question about it, and look at the answer. Every
script below was actually run to produce the figure shown; nothing here is
hand-drawn.

### [Detecting an inertial oscillation](examples/inertial_oscillation.jl)

Positive frequencies of a complex velocity `w = u + iv` are
counter-clockwise rotation; negative are clockwise. In the Northern
Hemisphere, a free inertial oscillation is *always* CCW (Gonella 1972) — so
a rotary spectrum can pull a 26.7-hour inertial signal out of a noisy
current-meter record and tell you, quantitatively, which way it's
spinning.

![Inertial oscillation hodograph and rotary spectrum](examples/inertial_oscillation.png)

```julia
spec = rotary_spectrum(u, v; dt_hours=1.0)
# rotary_coefficient ≈ +1.0 at the 26.7 h peak → purely CCW → inertial
```

### [Tracking an eddy's spin-down](examples/eddy_spindown.jl)

A single rotary spectrum smears a *decaying* eddy's energy across many
frequencies. `rotary_ridge` instead follows the instantaneous CCW
amplitude and period through time using a wavelet transform — watching the
eddy actually spin down, rather than reading off an average.

![Eddy spin-down amplitude and period tracked by wavelet ridge](examples/eddy_spindown.png)

```julia
result = rotary_ridge(u, v; dt=1.0, nv=8)
# result.amp_ccw tracks the true exponential decay envelope almost exactly
```

A ridge alone doesn't say whether it's a real feature or the strongest
fluctuation pure noise could produce over a record this long.
`wavelet_significance` answers that with a Monte Carlo noise threshold
(white or AR(1)/red background), and `ridge_significant` applies it
directly to `ridgemap`/`rotary_ridge` output:

```julia
sig_level, fs = wavelet_significance(x; dt=1.0, background=:red)
flags = ridge_significant(ridge_freq, ridge_amp, sig_level, fs)
```

### [Unit-safe model-vs-observation validation](examples/unit_safe_validation.jl)

The classic silent bug: comparing a model in m/s against a mooring in cm/s
and getting a validation score that's wrong by 100×, with no error and no
warning. `TimeSeriesVector` carries its unit in the data itself, so
`Dispatch.validate` converts automatically for compatible units and raises
`Unitful.DimensionError` immediately for incompatible ones — it cannot
make that mistake.

```julia
result = Dispatch.validate(model, obs)   # model in cm/s, obs in m/s — no problem
# RMSE: 0.036 m/s   Correlation: 0.998   Skill: 0.985
```

*This example is how we caught a real bug while writing it* — an earlier
version of `skill_score`/`bias` stripped each series' units independently
instead of converting to a common scale first, silently turning a skill
score of `0.985` into `-10069`. Fixed in `Dispatch._stripped_common_unit`;
see the script for the full story.

### [Multitaper line detection: F-test vs. eyeballing the periodogram](examples/multitaper_line_detection.jl)

A single periodogram is noisy enough that its highest bin isn't always the
true signal frequency — sampling noise can nudge the visible peak one bin
over. Thomson's (1982) multitaper F-test isn't fooled: it tests each
frequency for a genuine coherent line component, not just "which bin looks
tallest."

![Multitaper spectrum with F-test-identified line](examples/multitaper_spectrum.png)

```julia
spec = spectral_multitaper(x, 1.0; nw=4.0, unit=u"m/s")
line_peak = argmin(spec.ftest_pval)   # smallest p-value = the real line
# lands exactly on the true frequency, even when the raw power peak doesn't
```

### [A common oscillation across a mooring: multivariate ridge analysis](examples/multivariate_common_oscillation.jl)

Three current meters at different depths on one mooring all feel the same
near-inertial wave, just with different (depth-decaying) amplitude.
Analyzing each depth separately throws away the fact that they share one
signal. `multivariate_ridges` (Lilly & Olhede 2012) instead finds ONE joint
ridge across all `N` channels at once — jLab's genuinely `N`-general
wavelet ridge analysis, not a per-channel loop.

![Three depths sharing one oscillation, and the recovered amplitude-vs-depth profile](examples/multivariate_common_oscillation.png)

```julia
ridges = multivariate_ridges(X; dt=1.0, nv=8)   # X: (N, 3) — three depths, one call
r = only(ridges)
# r.wt_ridge reconstructs each depth's amplitude, matching the true decay profile
```

### [Batching a mooring array through one wavelet transform](examples/mooring_array_batch.jl)

`wavetrans` accepts any array with time along dimension 1 and independent
signals along the trailing dimensions (jLab's own column-signal
convention), so an entire 8-depth mooring array goes through a single
batched FFT call instead of a loop — one transform, one amplitude-vs-depth
profile.

![8-depth mooring array and its batched amplitude-vs-depth profile](examples/mooring_array_batch.png)

```julia
wt, fs = wavetrans(X; dt=1.0, nv=8)     # X: (N, 8) — one batched call, not 8
amp = tiredecode(wt, fs; kind="amp")    # (N, n_freq, 8), still fully batched
```

### [Detecting a coherent signal with SVD-based polarization (msvd)](examples/svd_polarization_detection.jl)

`msvd` (Park, Vernon & Lindberg 1987) singular-value-decomposes each
frequency band's raw `channels × looks` matrix directly, before any
pooling — giving the fraction of a band's power explained by a single
rank-1 structure, `d₁²/trace(S)`. A genuinely coherent bivariate
oscillation sits near 1; independent noise sits near its asymptotic 0.5 —
a clean signal-detection statistic the pooled `ellipse_polarization` route
doesn't give.

![MSVD rank-1 power fraction spiking sharply at the true signal frequency](examples/svd_polarization_detection.png)

```julia
r = msvd(W)                                # W: (bands, channels, looks)
explained = r.d[:, 1].^2 ./ r.trS          # sharp peak at the true signal frequency
```

### [Synthesizing and characterizing a modulated ellipse (ellpol/ellsig)](examples/ridge_ellipse_polarization.jl)

`ellipse_polarization` characterizes polarization in the frequency domain.
`ellsig`/`ellpol` take the complementary time-domain route: given a wavelet
ridge's time-varying ellipse parameters (RMS amplitude, linearity,
orientation, orbital phase), `ellsig` reconstructs the actual signal and
`ellpol` summarizes it with one polarization state, satisfying the
identity `P² = alpha² + beta²` for any consistent decomposition.

![A precessing, breathing ellipse and its true time-varying parameters](examples/ridge_ellipse_polarization.png)

```julia
x, y = ellsig(kappa, lambda, theta, phi)   # synthesize the signal from known ellipse params
r = ellpol(kappa, lambda, theta, phi)
# r.P^2 ≈ r.alpha^2 + r.beta^2 exactly, as it must for any consistent decomposition
```

*A strongly modulated ellipse (orientation sweeping a half-turn) gives a
low time-averaged `P`* — not a bug, but `ellpol` correctly reporting that
one fixed polarization state is a poor summary of a signal whose character
changes a lot over the record; see the script for the full explanation.

### [Case study: detecting significant eddies in real Lagrangian drifter data](examples/gomed_eddy_census.jl)

Every other example above uses synthetic data. This one is production
output: `local_tangent_plane` + `rotary_ridge_properties` +
`density_ratio_significance` (Lilly & Pérez-Brunius 2021, *NPG* 28,
181–212 — jLab's `eddyridges.m`/`spheretrans.m`, which wavelet-transforms
*position*, never velocity) run on the real, public GulfDriftersOpen
dataset (2684 Gulf of Mexico drifters). The full census is a multi-hour HPC
job, so the example loads the pipeline's own ~36 KB output CSV and
reproduces its summary statistics and figures from that.

![Ridge length and rotation-sense distributions of 497 significant eddy events](examples/gomed_eddy_census.png)

```julia
events = read_events_csv("gomed_eddy_census_data.csv")   # pipeline's own output, bundled
# 497 significant events, 86.5% cyclonic — consistent with the paper's published asymmetry
```

*Note on GOMED:* the published GOMED eddy-census NetCDF is
non-commercial/no-redistribution licensed and is not used, shipped, or
plotted anywhere in this example. What's compared above is our own
detections against the paper's own **published** headline figures
(Sect. 4.7) — not against GOMED's underlying per-ridge data.

---

*ValTools.jl v0.1.0 — A. Dominguez, CICESE, 2026*
