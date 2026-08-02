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
    ellsig(kappa, lambda, theta, phi)

Reconstruct the analytic (positive-frequency) `x`, `y` signal components of
a modulated elliptical signal from its time-varying **RMS amplitude**
`kappa`, **linearity** `lambda` (∈ `[-1,1]`; 0 = circular, ±1 = rectilinear,
sign gives rotation sense), **orientation** `theta`, and **orbital phase**
`phi` — all in radians except `kappa` and `lambda`, which are unitless.

Ported from `jEllipse/ellsig.m` (2D case only; jLab's 3D/trivariate case is
not implemented). `ellsig` is inverted by `ellparams` (not yet ported — see
[`ellipsefit`](@ref) for the current, simpler single-fit alternative).

# Returns
`(x, y)::Tuple{Vector{ComplexF64}, Vector{ComplexF64}}`, the analytic
signal along each axis (`z = x + iy` in jLab's complex-signal convention).

# References
Lilly & Gascard (2006); Lilly & Olhede (2010a).
"""
function ellsig(kappa::AbstractVector{<:Real}, lambda::AbstractVector{<:Real},
                theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    n = length(kappa)
    (length(lambda) == n && length(theta) == n && length(phi) == n) ||
        error("kappa, lambda, theta, phi must have the same length")

    # kl2ab.m: semi-major/minor axes from RMS amplitude + linearity.
    a = kappa .* sqrt.(1 .+ abs.(lambda))
    b = sign.(lambda) .* kappa .* sqrt.(1 .- abs.(lambda))
    # kl2ab.m defines b positive for lambda exactly zero (sign(0)=0 would
    # otherwise zero it out).
    b = [lambda[i] == 0 ? kappa[i] : b[i] for i in eachindex(b)]

    r = cis.(phi)   # rot.m: exp(i*phi)
    x = r .* (a .* cos.(theta) .+ im .* b .* sin.(theta))
    y = r .* (a .* sin.(theta) .- im .* b .* cos.(theta))
    return x, y
end

# Eigenvalues of the 2x2 Hermitian "spectral matrix" [[a, c+id],[c-id, b]]
# (specdiag.m's closed-form d1/d2, ported directly -- ellpol only ever uses
# the eigenvalues, never specdiag's th/nu orientation angles, so those
# aren't ported). `disc` is mathematically >=0 for any matrix built from
# real second moments (a,b>=0 by construction, c^2+d^2<=a*b by
# Cauchy-Schwarz) -- clamped at 0 as a numerical-noise guard only; jLab's
# own specdiag.m considered this exact guard (a commented-out
# `dets(dets<0)=0`) and left it disabled, so this is a deliberate,
# harmless divergence from bit-for-bit fidelity, not a physical difference.
function _specmat_eigenvalues(a::Real, b::Real, c::Real, d::Real)
    trs = a + b
    dets = a * b - c^2 - d^2
    disc = sqrt(max(trs^2 - 4 * dets, 0.0))
    # Both eigenvalues are mathematically >=0 for a genuine PSD Hermitian
    # matrix; clamp d2 too (not just the discriminant above) since it can
    # still drift a few ULPs negative in the degenerate case (disc≈trs,
    # e.g. perfectly rectilinear motion) -- caught by ellpol's own rbar
    # (sqrt(sqrt(d1*d2))) erroring on a tiny negative product, unlike
    # MATLAB which silently promotes to a tiny complex value here.
    return max((trs + disc) / 2, 0.0), max((trs - disc) / 2, 0.0)
end

"""
    ellpol(kappa, lambda, theta, phi)

Time-averaged polarization parameters of an elliptical signal, given its
time-varying ellipse parameters (as from a bivariate wavelet ridge — see
[`ellsig`](@ref)'s docstring for the parameter definitions).

`P` gives the **total polarization**, `alpha` the **excess of positive to
negative rotational energy** (the frequency-integrated rotary coefficient),
and `beta` the **polarization of the real part** of the spectral matrix
(linear-motion polarization). These satisfy `P² = alpha² + beta²`. `kbar`
and `rbar` are the average RMS axis length and average geometric-mean
radius, computed from the real part of the spectral matrix's eigenvalues.

Ported from `jEllipse/ellpol.m` (single-record vector input only — jLab's
generic array/cell-array batching over multiple records is not ported; call
once per ridge/record). Works by reconstructing the analytic signal via
[`ellsig`](@ref), forming its time-averaged second-moment "spectral matrix"
(`sxx`, `syy`, `sxy`), and diagonalizing it (jLab's `specdiag.m`) — this
naturally gives the correct κ²-weighted-in-effect average, since `ellsig`'s
instantaneous amplitude directly enters the second moments.

# Returns
`NamedTuple` with fields `P`, `alpha`, `beta`, `kbar`, `rbar`.

# References
Lilly (2018), jEllipse/ellpol.m. See also `Metrics.rotary_coherence` for
the analogous frequency-domain (non-ridge) polarization decomposition.
"""
function ellpol(kappa::AbstractVector{<:Real}, lambda::AbstractVector{<:Real},
                theta::AbstractVector{<:Real}, phi::AbstractVector{<:Real})
    xr, yr = ellsig(kappa, lambda, theta, phi)
    sxx = mean(abs2.(xr))
    syy = mean(abs2.(yr))
    sxy = mean(xr .* conj.(yr))

    d1, d2 = _specmat_eigenvalues(sxx, syy, real(sxy), imag(sxy))
    P = (d1 - d2) / (d1 + d2)
    alpha = 2 * imag(sxy) / (sxx + syy)

    # beta/kbar/rbar use ONLY the real part of sxy (specdiag called with
    # real(sxy), per ellpol.m) -- describes the real/linear-motion ellipse
    # geometry, separate from the rotary (imaginary-part) polarization alpha.
    d1r, d2r = _specmat_eigenvalues(sxx, syy, real(sxy), 0.0)
    beta = (d1r - d2r) / (d1r + d2r)
    kbar = sqrt(d1r / 2 + d2r / 2)
    rbar = sqrt(sqrt(d1r * d2r))

    return (P=P, alpha=alpha, beta=beta, kbar=kbar, rbar=rbar)
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

