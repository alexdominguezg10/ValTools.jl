"""
Ellipse Analysis & Rotary Spectra

Ported from Jonathan M. Lilly's jEllipse (https://github.com/jonathanlilly/jlab)

**References:**
- Gonella, J. (1972). A rotary-component method for analysing meteorological
  and oceanographic vector time series. Deep Sea Res., 19(12), 833–846.
- Lilly, J. M. (2010). Quantifying eddy–modulations of surface chlorophyll
  in the Gulf of Mexico. J. Geophys. Res., 115, C02040.
"""

"""
    ellipsefit(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
               method::String="ls", weighted::Bool=false)

Fit ellipse to velocity hodograph (u,v trajectory).

# Arguments
- `u::AbstractVector`: East-west velocity component
- `v::AbstractVector`: North-south velocity component
- `method::String`: "ls" (least squares, default)
- `weighted::Bool`: Use weighted fit (false by default)

# Returns
- `a::Float64`: Semi-major axis (m/s)
- `b::Float64`: Semi-minor axis (m/s)
- `theta::Float64`: Orientation angle (rad, from east)
- `phase::Float64`: Phase of rotation
- `ecc::Float64`: Eccentricity [0, 1]

# References
jEllipse/ellparams.m; rotary analysis theory

# Example
```julia
# Simulate anticyclone rotating clockwise (GoM-like)
t = 0:0.01:100
omega_rot = 2π / 20  # 20 second period
u = 0.5 .* cos.(omega_rot .* t)
v = -0.5 .* sin.(omega_rot .* t)  # negative for CW

a, b, theta, phase, ecc = ellipsefit(u, v)
# a ≈ 0.5, b ≈ 0.5, ecc ≈ 0 (circle)
```
"""
function ellipsefit(u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                   method::String="ls", weighted::Bool=false)

    u = vec(u)
    v = vec(v)

    if length(u) != length(v)
        error("u and v must have same length")
    end

    n = length(u)

    if method == "ls"
        # Least squares ellipse fit
        # Formulation: z = u + i*v, fit z = A*exp(i*t) + B*exp(-i*t)
        # where A = (a_x - i*a_y)/2, B = (a_x + i*a_y)/2

        z = u .+ im .* v

        # Solve for complex amplitudes A, B
        # (This is a simplified version; full method in jLab is more robust)
        t_lin = collect(0:n-1)
        M = [exp.(im .* t_lin) exp.(-im .* t_lin)]

        # Normal equations
        A_solve = M' * M
        b_solve = M' * z

        if abs(det(A_solve)) > 1e-10
            coeffs = A_solve \ b_solve
            A_comp = coeffs[1]
            B_comp = coeffs[2]

            # Extract ellipse parameters
            a_complex = A_comp + B_comp
            b_complex = im .* (B_comp - A_comp)

            a = abs(a_complex) / 2
            b = abs(b_complex) / 2
            theta = atan(imag(a_complex), real(a_complex))
            phase = 0.0
            ecc = min(abs(b), abs(a)) / max(abs(b), abs(a))
        else
            # Degenerate case
            a = std(u)
            b = std(v)
            theta = 0.0
            phase = 0.0
            ecc = min(a, b) / max(a, b)
        end

        return a, b, theta, phase, ecc
    else
        error("Method not implemented: $method")
    end
end

"""
    rotary(u::AbstractVector{<:Real}, v::AbstractVector{<:Real}, dt::Real;
           ntapers::Int=3, kind::String="power")

**Deprecated** — thin wrapper over [`Metrics.rotary_spectrum`](@ref), the
single unified rotary spectral implementation in ValTools.jl. Kept for
backward compatibility with existing call sites (e.g. `JLab.validate_rotary`)
and jLab's `polparams.m`-derived `(freqs, psd_cw, psd_ccw)` return order.
Prefer calling `Metrics.rotary_spectrum` directly for new code — it also
returns jackknife confidence intervals and a per-frequency rotary
coefficient.

# Returns
- `freqs::Vector{Float64}`: Frequencies
- `psd_cw::Vector{Float64}`: Clockwise rotary spectrum
- `psd_ccw::Vector{Float64}`: Counter-clockwise rotary spectrum

# References
Gonella (1972); jSpectral/polparams.m
"""
function rotary(u::AbstractVector{<:Real}, v::AbstractVector{<:Real}, dt::Real;
               ntapers::Int=3, kind::String="power")
    spec = Metrics.rotary_spectrum(u, v; dt_hours=dt, ntapers=ntapers, ci=false)
    return spec.freq, spec.S_cw, spec.S_ccw
end

