module Metrics

using Statistics
using LinearAlgebra
using FFTW
using Random
using DataFrames

include("stats.jl")
include("spectral.jl")

export compute_metrics, taylor_stats, bootstrap_metrics,
       metrics_by_group, current_ellipse_metrics
export rotary_spectrum, alongtrack_wavenumber_spectrum,
       isotropic_2d_spectrum, cross_spectrum_kx_ky, detrend_2d_linear

end # module
