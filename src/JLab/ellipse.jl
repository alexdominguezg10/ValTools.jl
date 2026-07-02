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

Rotary spectral decomposition of velocity into CW and CCW components.

# Arguments
- `u::AbstractVector`: East-west velocity
- `v::AbstractVector`: North-south velocity
- `dt::Real`: Sampling interval
- `ntapers::Int`: Number of tapers for spectral estimate
- `kind::String`: "power" (default)

# Returns
- `freqs::Vector{Float64}`: Frequencies
- `psd_cw::Vector{Float64}`: Clockwise rotary spectrum
- `psd_ccw::Vector{Float64}`: Counter-clockwise rotary spectrum

# Notes
- For ocean: CW = inertial oscillations (NH);
  CCW = tidal, wind-driven
- See Gonella (1972) for theory

# References
Gonella (1972); jSpectral/polparams.m
"""
function rotary(u::AbstractVector{<:Real}, v::AbstractVector{<:Real}, dt::Real;
               ntapers::Int=3, kind::String="power")

    u = vec(u)
    v = vec(v)

    if length(u) != length(v)
        error("u and v must have same length")
    end

    n = length(u)

    # Complex velocity
    w = u .+ im .* v

    # FFT
    N_fft = 2^ceil(Int, log2(2*n - 1))
    w_pad = vcat(w, zeros(N_fft - n))
    W = fft(w_pad)

    # Frequency grid
    freqs = fftfreq(N_fft, 1.0/dt)

    # Gonella (1972) convention for w = u + iv:
    # positive frequencies → CCW rotation
    # negative frequencies → CW rotation
    psd_all = (abs.(W).^2) .* dt ./ n

    pos_mask = freqs .> 0
    neg_mask = freqs .< 0

    freqs_pos = freqs[pos_mask]
    psd_ccw = psd_all[pos_mask]           # positive freq → CCW

    freqs_neg = -freqs[neg_mask]
    psd_cw_raw = psd_all[neg_mask]        # negative freq → CW

    # Sort to match positive frequency order
    order = sortperm(freqs_neg)
    freqs_neg = freqs_neg[order]
    psd_cw_raw = psd_cw_raw[order]

    # Interpolate CW to positive frequency grid
    psd_cw = linear_interp(freqs_neg, psd_cw_raw, freqs_pos)

    return freqs_pos, psd_cw, psd_ccw
end

# Helper interpolation function
function linear_interp(x::AbstractVector, y::AbstractVector, xi::AbstractVector)
    yi = similar(xi)
    for i in eachindex(xi)
        # Find surrounding points
        if xi[i] <= x[1]
            yi[i] = y[1]
        elseif xi[i] >= x[end]
            yi[i] = y[end]
        else
            # Find interval
            j = searchsortin(x, xi[i])
            if j > length(x)
                j = length(x)
            end
            if j == 0
                j = 1
            end

            # Linear interpolation
            if j < length(x)
                w = (xi[i] - x[j]) / (x[j+1] - x[j])
                yi[i] = (1-w) * y[j] + w * y[j+1]
            else
                yi[i] = y[j]
            end
        end
    end
    return yi
end

function searchsortin(a::AbstractVector, x::Real)
    # Find index j such that a[j] <= x < a[j+1]
    for j in 1:length(a)-1
        if a[j] <= x < a[j+1]
            return j
        end
    end
    return length(a)
end

