# ============================================================================
# HELMHOLTZ DECOMPOSITION OF WAVENUMBER SPECTRA (Bühler, Callies & Ferrari 2014)
# ============================================================================

"""
    HelmholtzSpectra

Wavenumber-domain Helmholtz decomposition of a 1-D (along-track) velocity
spectrum, following Bühler, Callies & Ferrari (2014, JFM 756), eqs. 2.27–2.31.

# Fields
- `k`: wavenumber grid (ascending, same values as the `k` passed in)
- `Dpsi`, `Dphi`: the auxiliary functions Dψ, Dφ (their eqs. 2.30–2.31)
- `KErot`, `KEdiv`: rotational and divergent kinetic energy spectra (eq. 2.27).
  These satisfy `KErot .+ KEdiv ≈ 0.5 .* (Cu .+ Cv)` exactly (energy
  conservation — see [`helmholtz_decomposition`](@ref)).
"""
struct HelmholtzSpectra
    k::Vector{Float64}
    Dpsi::Vector{Float64}
    Dphi::Vector{Float64}
    KErot::Vector{Float64}
    KEdiv::Vector{Float64}
end

"""
    helmholtz_decomposition(k, Cu, Cv)

Split a 1-D along-track velocity wavenumber spectrum into rotational and
divergent kinetic energy, following Bühler, Callies & Ferrari (2014, JFM
756, "Wavenumber spectra of rotational and divergent kinetic energy from
spatial data"), eqs. 2.27–2.31.

`k` must be strictly increasing and positive. `Cu` is the longitudinal
(along-track-direction) and `Cv` the transverse velocity spectrum, both
sampled on `k`.

# Method
Integration is performed in log-wavenumber `s = ln(k)`, backward from
`k[end]` (`Dpsi = Dphi = 0` there, per BCF14's own convention). The nested
kernel integral in their eqs. 2.30–2.31 is turned into two plain cumulative
integrals via the hyperbolic addition identities
`sinh(s-s̄) = sinh(s)cosh(s̄) - cosh(s)sinh(s̄)` and the equivalent for
`cosh(s-s̄)`:

```
A(s̄) = Cu(s̄)cosh(s̄) - Cv(s̄)sinh(s̄)
B(s̄) = Cv(s̄)cosh(s̄) - Cu(s̄)sinh(s̄)
IA(s) = ∫_s^smax A(s̄) ds̄,   IB(s) = ∫_s^smax B(s̄) ds̄
Dψ(s) = sinh(s)·IA(s) + cosh(s)·IB(s)
Dφ(s) = cosh(s)·IA(s) + sinh(s)·IB(s)
```

`IA`, `IB` are accumulated with the trapezoidal rule (O(Δs²)), exact in the
continuum limit — this avoids ever evaluating `sinh`/`cosh` of a *difference*
of two wavenumber-grid points directly, which is the naive (and less
accurate) way to discretize the nested kernel.

Since `k d/dk = d/ds`, `KErot = 0.5*(Dψ - dDψ/ds)` and
`KEdiv = 0.5*(Dφ - dDφ/ds)` (eq. 2.27), with `dD/ds` from central
differences (one-sided at the endpoints).

A warning is emitted if `Dψ` or `Dφ` goes negative anywhere — BCF14 note
this is a genuine pathology near the noise floor of real (as opposed to
analytic) spectra, not necessarily a bug, so it is surfaced rather than
silently clipped.

# References
Bühler, O., Callies, J., & Ferrari, R. (2014). Wavenumber spectra of
rotational and divergent kinetic energy from spatial data. Journal of
Fluid Mechanics, 756, 1007–1026. doi:10.1017/jfm.2014.488
"""
function helmholtz_decomposition(k::AbstractVector{<:Real}, Cu::AbstractVector{<:Real},
                                  Cv::AbstractVector{<:Real})
    n = length(k)
    (length(Cu) == n && length(Cv) == n) || error("k, Cu, Cv must have the same length")
    n >= 3 || error("need at least 3 wavenumber samples")
    all(diff(k) .> 0) || error("k must be strictly increasing")
    all(k .> 0) || error("k must be positive")

    kk = Float64.(k)
    Cu64 = Float64.(Cu)
    Cv64 = Float64.(Cv)
    s = log.(kk)

    ch = cosh.(s)
    sh = sinh.(s)
    A = Cu64 .* ch .- Cv64 .* sh
    B = Cv64 .* ch .- Cu64 .* sh

    IA = zeros(n)
    IB = zeros(n)
    for i in (n - 1):-1:1
        ds = s[i + 1] - s[i]
        IA[i] = IA[i + 1] + 0.5 * (A[i] + A[i + 1]) * ds
        IB[i] = IB[i + 1] + 0.5 * (B[i] + B[i + 1]) * ds
    end

    Dpsi = sh .* IA .+ ch .* IB
    Dphi = ch .* IA .+ sh .* IB

    n_neg_psi = count(<(0), Dpsi)
    n_neg_phi = count(<(0), Dphi)
    if n_neg_psi > 0 || n_neg_phi > 0
        @warn "helmholtz_decomposition: Dpsi or Dphi went negative — BCF14 document " *
              "this as a genuine pathology near the noise floor, not necessarily a bug" n_neg_psi n_neg_phi
    end

    dDpsi = _central_diff(Dpsi, s)
    dDphi = _central_diff(Dphi, s)

    KErot = 0.5 .* (Dpsi .- dDpsi)
    KEdiv = 0.5 .* (Dphi .- dDphi)

    return HelmholtzSpectra(kk, Dpsi, Dphi, KErot, KEdiv)
end

function _central_diff(y::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    n = length(y)
    dy = zeros(Float64, n)
    dy[1] = (y[2] - y[1]) / (x[2] - x[1])
    dy[n] = (y[n] - y[n - 1]) / (x[n] - x[n - 1])
    for i in 2:(n - 1)
        dy[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1])
    end
    return dy
end

"""
    wave_vortex_decomposition(hs::HelmholtzSpectra)

Split total kinetic energy into internal-gravity-wave and vortical
(balanced) parts, following Bühler, Callies & Ferrari (2014), eq. 2.37.

For linear hydrostatic internal gravity waves, ψ and φ are in quadrature,
and the vortex flow is exactly non-divergent so it contributes no φ at
all. The total wave energy therefore follows from the divergent part of
the velocities alone — **no buoyancy measurement is required**:

```
Ewave = Dφ - k dDφ/dk = 2·KEdiv
Evortex = Etotal - Ewave
```

Because `KErot + KEdiv ≡ 0.5*(Cu+Cv)` exactly (energy conservation, see
[`helmholtz_decomposition`](@ref)), `Etotal = 2*(KErot+KEdiv)` and so
`Evortex` reduces algebraically to `2·KErot`.

Returns a named tuple `(k, Ewave, Evortex)`.
"""
function wave_vortex_decomposition(hs::HelmholtzSpectra)
    Ewave = 2.0 .* hs.KEdiv
    Evortex = 2.0 .* hs.KErot
    return (; k=hs.k, Ewave, Evortex)
end
