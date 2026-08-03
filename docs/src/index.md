# ValTools.jl

**ValTools.jl** is a Julia package for oceanographic model validation, observation synthesis, and signal analysis. It brings together spectral methods (multitaper, wavelets, rotary decomposition), GPU-accelerated transforms, type-safe data handling (Unitful.jl), and a comprehensive suite of loaders for oceanographic observations (Argo, DUACS, SWOT, NDBC, RAFOS, CloudDrift, gliders, moorings, thermistors, and more).

## What is ValTools for?

You use ValTools.jl when you have:
- **Ocean model output** (CROCO, ROMS, NEMO) and want to **validate it** against observations
- **Time series data** (currents, temperature, salinity, SSH) and want to **decompose it into oscillations** (inertial, tidal, mesoscale, topographic)
- **Velocity pairs** (u, v) and want to **separate rotational modes** (counter-clockwise vs. clockwise, e.g. inertial vs. tidal)
- **Noisy signals** where **wavelet analysis** can **track time-varying frequency** (e.g., eddy spin-up/decay)
- **Moored instruments** affected by **knockdown dynamics** (where water depth ≠ instrument depth under strong currents)
- **Observation networks** (gliders, drifters, Argo profiles) that need **colocation** to model grid points or trajectories

## Key features

### Signal Analysis (from Flare project)
- **Multitaper spectral estimation** (Thomson's method, adaptive CIs via jackknife, harmonic F-tests)
- **Wavelet analysis** (continuous wavelet transform, generalized Morse wavelets, ridge chains, multivariate tracking, significance tests)
- **Rotary decomposition** (split velocity into inertial/tidal components via positive/negative frequency branches)
- **Polarization metrics** (ellipse parameters, SVD-based analysis)
- **Parametric spectral modeling** (AR processes, autoregressive spectral estimation)

### Oceanographic Methods
- **Model readers** (CROCO/ROMS/NEMO: SSH, SST, temperature, salinity, velocities, vertical coordinate transforms with GPU support)
- **Observation loaders** (Argo, DUACS, SWOT, NDBC, RAFOS, IES, thermistor chains, mooring currents, gliders, CloudDrift, CANEK sections, GOFLOW imagery)
- **Colocation & matching** (model-to-observation, model-to-trajectory, virtual moorings, knockdown correction)
- **Validation metrics** (Taylor diagrams, bias/RMSE/skill scores, bootstrap confidence intervals, current ellipse statistics)
- **Specialized models** (gravitational-equivalent model builder from Argo profiles, mooring knockdown dynamics)

### Type Safety & GPU Acceleration
- **Unitful.jl integration**: track physical units (m/s, °C, dbar) throughout your pipeline
- **Typed structs** for time series and spectral estimates (dispatch-based plotting)
- **GPU kernels** (CUDA via weakdep): sigma-to-z vertical coordinate transforms, LIC flow visualization
- **Batched wavelet transforms** on GPU (up to 15× speedup on H200 verified)

## Gallery

Browse the **[Examples Gallery](gallery.md)** for self-contained demos of each capability:
- Detecting inertial oscillations
- Tracking eddy decay with wavelets
- Validating model currents against mooring data
- Colocation & Taylor diagrams
- GPU-accelerated spectral analysis
- (And many more — plus roadmap stubs for planned features)

## Quick Start

```julia
using ValTools, CairoMakie, Random

# Simulate 40 days of inertial oscillations at 25°N
dt = 1.0  # hourly samples
t = 0:dt:(24*40)
f_inertial = 1 / 26.7  # cycles/hour

u = 0.15 .* cos.(2π * f_inertial .* t) .+ 0.03 .* randn(length(t))
v = 0.15 .* sin.(2π * f_inertial .* t) .+ 0.03 .* randn(length(t))

# Rotary spectral decomposition
spec = rotary_spectrum(u, v; dt_hours=dt)

# Inspect the result
peak_idx = argmax(spec.S_ccw)
println("Inertial period: ", round(1/spec.freq[peak_idx], digits=1), " h")
println("Rotary coeff: ", round(spec.rotary_coefficient[peak_idx], digits=2))
# Output: Inertial period: 26.7 h, Rotary coeff: 0.96  ✓
```

## API Reference

See [API Reference](api.md) for the full function listing.

## Contributing

ValTools.jl is actively developed. Contributions — new loaders, methods, examples, documentation — are welcome. See [Contributing Guide](contributing.md).

## Citation

If you use ValTools.jl in research, please cite:
```bibtex
@software{dominguez2024valtools,
  title={ValTools.jl: Oceanographic model validation and signal analysis in Julia},
  author={Dominguez, Alex},
  year={2024},
  url={https://github.com/alexdominguezg10/ValTools.jl}
}
```

## License

ValTools.jl is licensed under the MIT License.
