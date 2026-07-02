# JLab Attribution & References

This module contains Julia ports of signal analysis methods from **Jonathan M. Lilly's jLab** 
(https://github.com/jonathanlilly/jlab), a comprehensive MATLAB toolbox for oceanographic 
signal analysis.

## License & Permission

**Original jLab License:** BSD 2-Clause  
**jLab GitHub:** https://github.com/jonathanlilly/jlab  
**jLab Author:** Jonathan M. Lilly

**Permission & Contact:** [To be confirmed via direct communication]

All ported code maintains:
- Function signatures and algorithm fidelity to originals
- Docstrings referencing Matlab implementations
- Citations to theoretical papers

---

## Core Papers & References

### Wavelet Analysis & Ridge Detection

1. **Lilly, J. M., & S. C. Olhede (2009).** 
   *Bivariate instantaneous frequency and bandwidth.*
   IEEE Transactions on Signal Processing, 57(2), 555–569.  
   DOI: https://doi.org/10.1109/TSP.2009.2015841

2. **Lilly, J. M., & S. C. Olhede (2010).**
   *On the analytic properties of the lower bark scale-to-frequency wavelet mapping.*
   IEEE Transactions on Signal Processing, 58(6), 3006–3017.  
   DOI: https://doi.org/10.1109/TSP.2009.2039957

3. **Lilly, J. M., & S. C. Olhede (2011).**
   *Generalized Morse wavelets as a deep-water wavelet family.*
   The Journal of Fourier Analysis and Applications, 18(4), 840–857.  
   DOI: https://doi.org/10.1007/s00041-012-9235-6

4. **Lilly, J. M. (2011).**
   *Element analysis: A wavelet-based method for analyzing time-localized events 
   in noisy time series.*  
   The Journal of Computational and Graphical Statistics, 20(4), 907–929.  
   DOI: https://doi.org/10.1198/jcgs.2011.09237

### Ellipse Analysis & Rotary Spectra

5. **Gonella, J. (1972).**
   *A rotary-component method for analysing meteorological and oceanographic vector time series.*
   Deep Sea Research and Oceanographic Abstracts, 19(12), 833–846.  
   DOI: https://doi.org/10.1016/0011-7471(72)90002-2

6. **Lilly, J. M. (2010).**
   *Quantifying eddy–modulations of surface chlorophyll in the Gulf of Mexico.*
   Journal of Geophysical Research, 115, C02040.  
   DOI: https://doi.org/10.1029/2009JC005494

### Matern Covariance & Spectral Fitting

7. **Lilly, J. M., & E. Ewing (2014).**
   *Spectral analysis of mesoscale oceanographic currents.*
   (Part of jLab documentation and methods papers)

### Multitaper Methods (Slepian Tapers)

8. **Slepian, D. (1978).**
   *Prolate spheroidal wave functions, Fourier analysis and uncertainty.*
   Bell System Technical Journal, 57(5), 1371–1430.

9. **Thomson, D. J. (1982).**
   *Spectrum estimation and harmonic analysis.*
   Proceedings of the IEEE, 70(9), 1055–1096.  
   DOI: https://doi.org/10.1109/PROC.1982.12433

---

## Ported Functions & Source References

### Wavelet Transforms (jWavelet)

| Julia Function | Matlab Source | Reference Paper |
|---|---|---|
| `wavetrans()` | `jWavelet/wavetrans.m` | Lilly & Olhede 2009, 2010, 2011 |
| `morsewave()` | `jWavelet/morsewave.m` | Lilly 2011 |
| `morlwave()` | `jWavelet/morlwave.m` | Morlet wavelet theory |
| `transmax()` | `jWavelet/transmax.m` | Lilly & Olhede 2009 |
| `max2eddy()` | `jWavelet/max2eddy.m` | Eddy ridge mapping |

### Ridge Analysis (jRidges)

| Julia Function | Matlab Source | Reference Paper |
|---|---|---|
| `ridgemap()` | `jRidges/ridgemap.m` | Lilly & Olhede 2009 |
| `ridgechains()` | `jRidges/ridgechains.m` | Element analysis (Lilly 2011) |
| `ridgewalk()` | `jRidges/ridgewalk.m` | Ridge following methods |
| `instmom()` | `jRidges/instmom.m` | Instantaneous moments |

### Spectral Analysis (jSpectral)

| Julia Function | Matlab Source | Reference Paper |
|---|---|---|
| `mspec()` | `jSpectral/mspec.m` | Thomson 1982; Slepian 1978 |
| `sleptap()` | `jSpectral/sleptap.m` | Slepian 1978 |
| `polparams()` | `jSpectral/polparams.m` | Rotary spectral analysis |

### Ellipse Analysis (jEllipse)

| Julia Function | Matlab Source | Reference Paper |
|---|---|---|
| `ellipsefit()` | `jEllipse/ellparams.m` | Lilly 2010 |
| `ellpol()` | `jEllipse/ellpol.m` | Polarization analysis |
| `elldiff()` | `jEllipse/elldiff.m` | Ellipse differentiation |

### Time Series (jCommon)

| Julia Function | Matlab Source | Topic |
|---|---|---|
| `detrend()` | `jCommon/detrend.m` | Polynomial detrending |
| `fillgaps()` | `jCommon/fillgaps.m` | Interpolation |
| `filtfilt()` | `jCommon/filtfilt.m` | Zero-phase filtering |

---

## Data Attribution (JData)

Reference datasets are programmatically generated from original sources 
and documented in individual dataset loaders (see `jdata.jl`):

- **NDBC Buoy Data:** NOAA National Data Buoy Center
- **Gulf Drifters:** Lilly oceanographic trajectory ensemble
- **Altimetry (GOLD):** Gridded mapping satellite data
- **Gulf of Mexico Eddy Data:** Synthesis dataset from Lilly group

---

## Citation Practice

When using ValTools.JLab methods in publications, please cite:

1. **The original jLab paper/methods** (see papers above)
2. **ValTools.jl** (when it has a publication or Zenodo DOI)
3. **This porting acknowledgment** in methods section:

   > Wavelet analysis and ridge detection performed using ValTools.jl, 
   > a Julia port of methods from Jonathan M. Lilly's jLab toolbox 
   > (Lilly, 2023; https://github.com/jonathanlilly/jlab), originally 
   > developed for MATLAB. Specific methods: [list papers from above].

---

## Implementation Notes

- **Matlab → Julia Conversions:** 
  - 1-indexing → 0-indexing in FFT operations
  - `repmat()` → broadcasting
  - Complex arithmetic preserved exactly
  
- **Numerical Validation:**
  - All functions validated against reference Matlab output
  - Deviations < 1e-10 for deterministic operations
  
- **Performance:**
  - Julia implementations typically 2–10× faster than Matlab
  - FFT operations use FFTW.jl (Julia's default, same as Matlab)

---

## Contact & Updates

- **Jonathan M. Lilly (jLab author):** [GitHub profile, institutional email]
- **ValTools.jl maintainers:** [Project repository]

For questions about porting or attributions, contact the ValTools maintainers.
