"""
ValToolsMultitaperExt — Multitaper.jl-backed spectral estimation for ValTools

Loaded automatically when both ValTools and Multitaper are in the environment
(`using ValTools, Multitaper`). Provides the real implementations of:
- `JLab.spectral_multitaper`, `JLab.mspec`, `JLab.sleptap`
- `Metrics.rotary_spectrum`, `Metrics.rotary_coherence`
- `Metrics.cross_coherence`
- `Metrics.ellipse_polarization`

Multitaper.jl is GPL-2.0-licensed. Isolating it behind this extension keeps
the rest of ValTools.jl (including plotting-only environments that don't
need multitaper spectral estimation) free of that dependency and its
transitive version constraints.
"""
module ValToolsMultitaperExt

using ValTools
using Multitaper: multispec, dpss_tapers, MTSpectrum
using Unitful
using Dates
using Statistics: mean
using FFTW: fft, ifft
using SpecialFunctions: erfinv
using Random

const JL = ValTools.JLab
const Met = ValTools.Metrics

import ValTools.JLab: spectral_multitaper, mspec, sleptap
import ValTools.Metrics: rotary_spectrum, rotary_coherence, cross_coherence, ellipse_polarization,
                         rotary_noise_spectrum, rotary_noise_surrogate

include("spectral_multitaper.jl")
include("rotary_spectrum.jl")
include("rotary_noise.jl")
include("cross_coherence.jl")
include("ellipse_polarization.jl")

end # module
