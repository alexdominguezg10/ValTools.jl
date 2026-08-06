# ============================================================================
# VELOCITY STRUCTURE FUNCTIONS (Lindborg 2015) — the drifter-pair estimator
# ============================================================================

"""
    velocity_structure_functions(x, y, u, v; rbins)

Second-order longitudinal (`Dll`) and transverse (`Dtt`) velocity structure
functions from scattered position/velocity pairs (e.g. simultaneous
drifters), the separation-space equivalent of a wavenumber spectrum.

For each pair `(i,j)`, the velocity difference `(u[j]-u[i], v[j]-v[i])` is
decomposed into components parallel (`ul`, longitudinal) and perpendicular
(`ut`, transverse) to the separation vector `(x[j]-x[i], y[j]-y[i])`. Then
`ul^2`/`ut^2` are binned by separation distance `r` into `rbins` (bin
edges, ascending) and averaged within each bin.

`x`, `y` should be in a common length unit (e.g. km — see
[`JLab.local_tangent_plane`](@ref) for lat/lon → km projection); `u`, `v` in
a common velocity unit. Pairs with zero or out-of-range separation are
skipped.

This is `O(n^2)` in the number of points. It is the estimator this project
uses for Gulf of Mexico drifters specifically because Qian, LaCasce & Peng
(2025, JPO) document that direct wavenumber-spectral estimates from GoM
drifter velocities have been unsuccessful — [`helmholtz_decomposition`](@ref)
needs a spectrum, this function needs only pair separations.

Returns a named tuple `(r, Dll, Dtt, npairs)` with `r` the bin centers. Bins
with zero pairs are `NaN` in `Dll`/`Dtt` (and `0` in `npairs`).

# References
Lindborg, E. (2015). A Helmholtz decomposition of structure functions and
spectra calculated from aircraft data. Journal of Fluid Mechanics, 762, R4.
doi:10.1017/jfm.2014.685
"""
function velocity_structure_functions(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                                       u::AbstractVector{<:Real}, v::AbstractVector{<:Real};
                                       rbins::AbstractVector{<:Real})
    n = length(x)
    (length(y) == n && length(u) == n && length(v) == n) ||
        error("x, y, u, v must have the same length")
    issorted(rbins) || error("rbins must be sorted ascending")
    nb = length(rbins) - 1
    nb >= 1 || error("rbins must have at least 2 edges")

    xs = Float64.(x)
    ys = Float64.(y)
    us = Float64.(u)
    vs = Float64.(v)

    sum_ll = zeros(nb)
    sum_tt = zeros(nb)
    npairs = zeros(Int, nb)

    @inbounds for i in 1:(n - 1), j in (i + 1):n
        dx = xs[j] - xs[i]
        dy = ys[j] - ys[i]
        r = hypot(dx, dy)
        (r == 0 || r < rbins[1] || r >= rbins[end]) && continue
        b = searchsortedlast(rbins, r)
        (b < 1 || b > nb) && continue

        rx, ry = dx / r, dy / r
        tx, ty = -ry, rx
        du = us[j] - us[i]
        dv = vs[j] - vs[i]
        ul = du * rx + dv * ry
        ut = du * tx + dv * ty

        sum_ll[b] += ul^2
        sum_tt[b] += ut^2
        npairs[b] += 1
    end

    Dll = fill(NaN, nb)
    Dtt = fill(NaN, nb)
    for b in 1:nb
        if npairs[b] > 0
            Dll[b] = sum_ll[b] / npairs[b]
            Dtt[b] = sum_tt[b] / npairs[b]
        end
    end

    r_centers = 0.5 .* (rbins[1:(end - 1)] .+ rbins[2:end])
    return (r=r_centers, Dll=Dll, Dtt=Dtt, npairs=npairs)
end

"""
    helmholtz_structure_function(r, D_ll, D_tt)

Split longitudinal/transverse structure functions into rotational (`Drr`)
and divergent (`Ddd`) parts, following Lindborg (2015) — the
structure-function equivalent of [`helmholtz_decomposition`](@ref):

```
Drr(r) = Dtt(r) + ∫_0^r τ^-1 [Dtt(τ) - Dll(τ)] dτ
Ddd(r) = Dll(r) - ∫_0^r τ^-1 [Dtt(τ) - Dll(τ)] dτ
```

`r` must be strictly increasing and positive (e.g. the bin centers from
[`velocity_structure_functions`](@ref), with any `NaN` bins removed first).
The integral is accumulated by the trapezoidal rule from `r[1]` outward,
assuming the integrand `(Dtt-Dll)/τ → 0` as `τ → 0` — the unresolved
`[0, r[1])` contribution is approximated as a single trapezoid down to
zero rather than dropped. This holds for any real, differentiable velocity
field, where `Dtt - Dll = O(τ^2)` at the origin (equivalently: a power-law
`Dtt-Dll ~ τ^p` needs `p>1` for the integrand to vanish at `τ=0` rather
than diverge there — real structure functions satisfy this because they
must flatten to `O(τ^2)` at small enough separation regardless of their
inertial-range slope further out, but a *synthetic* test that extends a
sub-linear power law, `p<1`, all the way to `τ=0` violates it and will see
spurious error concentrated near `r[1]`).

By construction `Drr + Ddd ≡ Dll + Dtt` exactly, since the integral term
cancels algebraically in the two formulas above — this is *not* an
independent correctness check of this function's arithmetic, only a
guarantee that comes for free from the way `Drr`/`Ddd` are defined.

⚠️ For a power-law pair `Dll(r) = r^p`, `Dtt(r) = c·r^p`, the split is
`Ddd ≡ 0` for all `r` only at `c = 1+p` (not the naive guess `c=1`) —
mirroring [`helmholtz_decomposition`](@ref)'s wavenumber-space
`Cv = n·Cu` relation exactly. **Solid-body rotation (`Dll≡0`) and uniform
radial strain (`Dtt≡0`) are a *boundary* case of this family, not
counterexamples** — they do *not* give `Drr=Dtt,Ddd=0` /
`Ddd=Dll,Drr=0` respectively; `Drr`/`Ddd` can come out negative for these
inputs (`1.5×`/`-0.5×` the nonzero component). This was verified
numerically (not just derived) and initially looked like a bug before the
algebra above confirmed it's the exact, self-consistent behavior of this
formula at that boundary — a real-space analog of the negative-`Dpsi`/
`Dphi` pathology BCF14 document for the wavenumber case. If you're about
to test or use this function with a flow where one of `Dll`/`Dtt` is
identically zero, expect this.

Returns a named tuple `(r, Drr, Ddd)`.

# References
Lindborg, E. (2015). Journal of Fluid Mechanics, 762, R4.
doi:10.1017/jfm.2014.685
"""
function helmholtz_structure_function(r::AbstractVector{<:Real}, D_ll::AbstractVector{<:Real},
                                       D_tt::AbstractVector{<:Real})
    n = length(r)
    (length(D_ll) == n && length(D_tt) == n) || error("r, D_ll, D_tt must have the same length")
    n >= 1 || error("need at least one point")
    all(diff(r) .> 0) || error("r must be strictly increasing")
    all(r .> 0) || error("r must be positive")

    rr = Float64.(r)
    Dll = Float64.(D_ll)
    Dtt = Float64.(D_tt)
    integrand = (Dtt .- Dll) ./ rr

    I = zeros(Float64, n)
    I[1] = 0.5 * integrand[1] * rr[1]   # trapezoid over [0, r[1]], integrand(0) := 0
    for i in 2:n
        I[i] = I[i - 1] + 0.5 * (integrand[i - 1] + integrand[i]) * (rr[i] - rr[i - 1])
    end

    Drr = Dtt .+ I
    Ddd = Dll .- I
    return (r=rr, Drr=Drr, Ddd=Ddd)
end
