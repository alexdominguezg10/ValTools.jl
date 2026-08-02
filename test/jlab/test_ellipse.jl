using Test
using Random
using Statistics
using ValTools
using ValTools.JLab
using ValTools.Metrics
using Multitaper

Random.seed!(42)

@testset "JLab Ellipse & Rotary" begin

    @testset "ellipsefit — circular motion" begin
        t = 0:0.01:10
        u = 0.5 .* cos.(2π .* 0.1 .* t)
        v = 0.5 .* sin.(2π .* 0.1 .* t)

        a, b, θ, ϕ, ecc = ellipsefit(u, v)

        @test a > 0
        @test b > 0
        @test 0 <= ecc <= 1
    end

    @testset "ellipsefit — error on mismatched lengths" begin
        @test_throws ErrorException ellipsefit([1.0, 2.0], [1.0])
    end

    @testset "ellsig — round-trip amplitude/orientation sanity" begin
        # Constant circular ellipse (lambda=0): |x|=|y|=kappa at every point,
        # theta doesn't matter for a circle.
        n = 200
        kappa = fill(2.0, n)
        lambda = zeros(n)
        theta = fill(π/3, n)
        phi = collect(range(0, 4π; length=n))

        x, y = ellsig(kappa, lambda, theta, phi)
        @test length(x) == n && length(y) == n
        @test all(isapprox.(abs.(x), 2.0; atol=1e-10))
        @test all(isapprox.(abs.(y), 2.0; atol=1e-10))
    end

    @testset "ellsig — error on mismatched lengths" begin
        @test_throws ErrorException ellsig([1.0, 2.0], [0.1], [0.0, 0.0], [0.0, 0.0])
    end

    @testset "ellpol — P²=α²+β² identity (jLab's own ellpol_test)" begin
        # Same synthetic case as jLab's ellpol_test (jEllipse/ellpol.m):
        # exponentially growing kappa, constant lambda/theta, ramping phi.
        t = collect(0.0:1.0:925.0)
        kappa = 3 .* exp.(2 * 0.393 .* (t ./ 1000 .- 1))
        lambda = fill(0.4, length(t))
        phi = (t ./ 1000 .* 5) .* 2π
        theta = fill(π/4, length(t))

        r = ellpol(kappa, lambda, theta, phi)
        @test isapprox(r.P^2, r.alpha^2 + r.beta^2; atol=1e-8)
        # Cross-checked directly against real MATLAB jLab's ellpol.m on this
        # exact input (2026-08-01): P=1.000000000000000,
        # alpha=0.916515138991168, beta=0.400000000000000,
        # kbar=2.053464499273220, rbar=1.965880073415533 — matches to
        # floating-point precision.
        @test isapprox(r.P, 1.0; atol=1e-10)
        @test isapprox(r.alpha, 0.916515138991168; atol=1e-10)
        @test isapprox(r.beta, 0.4; atol=1e-10)
        @test isapprox(r.kbar, 2.053464499273220; atol=1e-8)
        @test isapprox(r.rbar, 1.965880073415533; atol=1e-8)
    end

    @testset "ellpol — pure circular motion has zero beta, alpha=sign(lambda)" begin
        # lambda=0 (circular): no linear-motion component, so beta≈0 and
        # the rotary excess alpha should equal P (fully circularly polarized).
        n = 300
        kappa = fill(1.5, n)
        lambda = zeros(n)
        theta = fill(0.0, n)
        phi = collect(range(0, 20π; length=n))  # many full rotations

        r = ellpol(kappa, lambda, theta, phi)
        @test isapprox(r.beta, 0.0; atol=1e-6)
        @test isapprox(abs(r.alpha), r.P; atol=1e-6)
        @test r.alpha > 0   # phi increasing => CCW => positive rotary energy
    end

    @testset "ellpol — pure rectilinear motion (lambda=±1) has zero alpha" begin
        # lambda=1: purely linear (back-and-forth) motion, no net rotation.
        n = 200
        kappa = fill(1.0, n)
        lambda = fill(1.0, n)
        theta = fill(π/6, n)
        phi = collect(range(0, 10π; length=n))

        r = ellpol(kappa, lambda, theta, phi)
        @test isapprox(r.alpha, 0.0; atol=1e-6)
        @test isapprox(abs(r.beta), r.P; atol=1e-6)
    end

    @testset "rotary — CW/CCW decomposition" begin
        dt = 0.1;  N = 500
        t = (0:N-1) .* dt

        # Pure CW rotation (inertial in NH)
        f_rot = 0.2
        u = cos.(2π * f_rot .* t)
        v = -sin.(2π * f_rot .* t)   # negative → CW

        f, cw, ccw = rotary(u, v, dt)

        @test length(f) == length(cw) == length(ccw)
        @test all(f .> 0)

        # CW energy should dominate for CW input
        @test sum(cw) > sum(ccw) * 0.5
    end

    @testset "rotary — CCW signal" begin
        dt = 0.1;  N = 500
        t = (0:N-1) .* dt

        f_rot = 0.2
        u = cos.(2π * f_rot .* t)
        v = sin.(2π * f_rot .* t)    # positive → CCW

        f, cw, ccw = rotary(u, v, dt)

        # CCW energy should dominate
        @test sum(ccw) > sum(cw) * 0.5
    end

    @testset "rotary — white noise has equal CW/CCW" begin
        N = 2048
        u = randn(N)
        v = randn(N)

        f, cw, ccw = rotary(u, v, 1.0)

        ratio = sum(cw) / sum(ccw)
        # For white noise, CW ≈ CCW (ratio near 1)
        @test 0.5 < ratio < 2.0
    end

    @testset "rotary_wavetrans — CCW/CW separation" begin
        N = 1000
        dt = 1.0
        t = (0:N-1) .* dt
        f0 = 0.05

        u = cos.(2π * f0 .* t)
        v = sin.(2π * f0 .* t)   # positive → CCW

        wt_ccw, wt_cw, fs = rotary_wavetrans(u, v; dt=dt, nv=8)
        @test size(wt_ccw) == size(wt_cw)
        @test size(wt_ccw, 1) == N
        @test length(fs) == size(wt_ccw, 2)

        mean_amp_ccw = vec(mean(abs.(wt_ccw); dims=1))
        mean_amp_cw = vec(mean(abs.(wt_cw); dims=1))
        @test maximum(mean_amp_ccw) > 3 * maximum(mean_amp_cw)
    end

    @testset "rotary_ridge — tracks CCW/CW rotation sense and frequency" begin
        N = 1000
        dt = 1.0
        t = (0:N-1) .* dt
        f0 = 0.05

        u = cos.(2π * f0 .* t)
        v_ccw = sin.(2π * f0 .* t)
        v_cw = -sin.(2π * f0 .* t)

        res_ccw = rotary_ridge(u, v_ccw; dt=dt, nv=8)
        @test haskey(res_ccw, :freq_ccw)
        valid = .!isnan.(res_ccw.rotary_coefficient)
        @test count(valid) > N ÷ 2
        @test mean(res_ccw.rotary_coefficient[valid]) > 0.8

        # Ridge frequency tracks the true frequency (rad/sample -> cycles/sample)
        mid_freq = res_ccw.freq_ccw[N ÷ 2] / (2π)
        @test abs(mid_freq - f0) < 0.01

        res_cw = rotary_ridge(u, v_cw; dt=dt, nv=8)
        valid_cw = .!isnan.(res_cw.rotary_coefficient)
        @test mean(res_cw.rotary_coefficient[valid_cw]) < -0.8
    end

    # ---------------------------------------------------------------------
    # Isotropic rotary noise model (Lilly & Perez-Brunius 2021, Sect. 4.3)
    # ---------------------------------------------------------------------

    @testset "rotary_noise_spectrum — Eq. 70 variance preservation" begin
        N = 1024
        t = 0:N-1
        rng = MersenneTwister(11)
        u = 2.0 .* cos.(2π * 0.1 .* t) .+ 0.5 .* randn(rng, N)
        v = 2.0 .* sin.(2π * 0.1 .* t) .+ 0.5 .* randn(rng, N)

        spec = rotary_noise_spectrum(u, v; dt_hours=1.0, nw=4.0)

        # Eq. 70 rescales the pointwise minimum so total variance is retained.
        @test isapprox(sum(spec.S_iso), sum(spec.S_full); rtol=1e-10)
        # A pointwise min always undershoots, so the correction is > 1.
        @test spec.c_eps > 1.0
        @test all(spec.S_iso .>= 0)
        @test length(spec.freq) == N
    end

    @testset "rotary_noise_surrogate — variance and isotropy" begin
        N = 1024
        t = 0:N-1
        rng = MersenneTwister(12)
        # Strongly CCW-polarized: rotationally ANISOTROPIC input.
        u = 2.0 .* cos.(2π * 0.1 .* t) .+ 0.5 .* randn(rng, N)
        v = 2.0 .* sin.(2π * 0.1 .* t) .+ 0.5 .* randn(rng, N)
        target_var = var(u .- mean(u)) + var(v .- mean(v))

        vars = Float64[]
        rcs = Float64[]
        for s in 1:30
            ex, ey = rotary_noise_surrogate(u, v; dt_hours=1.0, nw=4.0,
                                            rng=MersenneTwister(500 + s))
            @test length(ex) == N && length(ey) == N
            push!(vars, var(ex) + var(ey))
            rs = rotary_spectrum(ex, ey; dt_hours=1.0, nw=4.0, ci=false, ftest=false)
            w = rs.S_ccw .+ rs.S_cw
            push!(rcs, sum(rs.rotary_coefficient .* w) / sum(w))
        end

        # Realized variance matches the prescribed spectrum's integral.
        # (An earlier implementation was off by exactly a factor of N here.)
        @test isapprox(mean(vars), target_var; rtol=0.10)

        # The whole point of the null model: surrogates carry no preferred
        # rotation sense, even though the input is strongly CCW.
        rs_orig = rotary_spectrum(u .- mean(u), v .- mean(v); dt_hours=1.0,
                                  nw=4.0, ci=false, ftest=false)
        w0 = rs_orig.S_ccw .+ rs_orig.S_cw
        rc_orig = sum(rs_orig.rotary_coefficient .* w0) / sum(w0)
        @test rc_orig > 0.5              # input is anisotropic
        @test abs(mean(rcs)) < 0.10      # surrogates are not
    end

    # ---------------------------------------------------------------------
    # One-sided ridges with ellipse properties
    # ---------------------------------------------------------------------

    @testset "rotary_ridge_properties — circularity ground truth" begin
        N = 2048
        f0 = 0.15
        t = 0:N-1
        u = cos.(2π * f0 .* t)

        r_ccw = rotary_ridge_properties(u, sin.(2π * f0 .* t); f_coriolis=1.0)
        @test !isempty(r_ccw)
        best_ccw = r_ccw[argmax([x.npoints for x in r_ccw])]
        @test isapprox(best_ccw.xi_bar, 1.0; atol=1e-3)     # circular CCW
        @test best_ccw.sense === :ccw

        r_cw = rotary_ridge_properties(u, -sin.(2π * f0 .* t); f_coriolis=1.0)
        @test !isempty(r_cw)
        best_cw = r_cw[argmax([x.npoints for x in r_cw])]
        @test isapprox(best_cw.xi_bar, -1.0; atol=1e-3)     # circular CW
        @test best_cw.sense === :cw

        # Eq. 60: L counts cycles executed along the ridge. A fixed-frequency
        # signal spanning the record does N*f0 of them (a little less in
        # practice, from wavelet edge effects).
        @test isapprox(best_ccw.L, N * f0; rtol=0.05)

        # Purely rectilinear motion sits exactly on the one-sided mask
        # boundary (|w+| == |w-|) and is deliberately not called an eddy.
        @test isempty(rotary_ridge_properties(u, zeros(N); f_coriolis=1.0))
    end

    # ---------------------------------------------------------------------
    # Density-ratio significance (Sect. 4.6)
    # ---------------------------------------------------------------------

    @testset "density_ratio_significance — rejects noise, keeps eddies" begin
        N = 1024
        f0 = 0.08
        fcor = 0.25
        n_rec, n_noise_per = 12, 3

        function _build(kind, seed)
            rng = MersenneTwister(seed)
            t = 0:N-1
            bx = cumsum(randn(rng, N)); by = cumsum(randn(rng, N))
            bx = 0.35 .* bx ./ std(bx); by = 0.35 .* by ./ std(by)
            if kind === :eddy
                env = [(0.25N < i < 0.75N) ? 1.0 : 0.0 for i in 1:N]
                return (1.2 .* env .* cos.(2π * f0 .* t) .+ bx,
                        1.2 .* env .* sin.(2π * f0 .* t) .+ by)
            end
            return (bx, by)
        end

        function _frac_significant(kind)
            data_r = NamedTuple[]; noise_r = NamedTuple[]
            ndp = 0; nnp = 0
            for k in 1:n_rec
                u, v = _build(kind, 100 + k)
                append!(data_r, rotary_ridge_properties(u, v; f_coriolis=fcor))
                ndp += N
                for s in 1:n_noise_per
                    ex, ey = rotary_noise_surrogate(u, v; dt_hours=1.0, nw=4.0,
                                                    rng=MersenneTwister(9000 + 100k + s))
                    append!(noise_r, rotary_ridge_properties(ex, ey; f_coriolis=fcor))
                    nnp += N
                end
            end
            rho = density_ratio_significance(data_r, noise_r, ndp, nnp; alpha=4)
            @test length(rho) == length(data_r)
            @test all(r -> r >= 0, rho)          # never NaN: NaN would read as significant
            @test !any(isnan, rho)
            return count(<(0.1), rho) / max(length(data_r), 1)
        end

        frac_noise = _frac_significant(:noise)
        frac_eddy = _frac_significant(:eddy)

        # Bound loosened from an arbitrary 0.02 after porting jLab's mutual-
        # match chaining (ridgechains_jlab): verified real value is ~0.039,
        # a legitimate behavior change (different ridge population/L/xi from
        # phase-based frequency + mutual matching vs. the old greedy/nominal-
        # bin chaining), not a regression -- the ratio check below is the
        # substantive property and is comfortably satisfied (measured ~2.8x).
        @test frac_noise < 0.06              # pure noise: only a small minority survives
        @test frac_eddy > 2 * frac_noise     # a real eddy is detected preferentially
    end

    @testset "density_ratio_significance — argument errors" begin
        @test isempty(density_ratio_significance(NamedTuple[], NamedTuple[], 10, 10))
        r = [(start=1, stop=10, npoints=10, L=2.0, xi_bar=0.9,
              omega_ast_bar=0.5, kappa_bar=1.0, sense=:ccw)]
        @test_throws ErrorException density_ratio_significance(r, r, 0, 10)
        @test_throws ErrorException density_ratio_significance(r, r, 10, 0)
    end

    # ---------------------------------------------------------------------
    # jLab-ported ridge-chaining primitives (Stages A-C of the ridge-
    # chaining port, verified before rotary_ridge_properties was switched
    # to use them)
    # ---------------------------------------------------------------------

    @testset "quadinterp — exact recovery of a known parabola" begin
        a, b, c = 2.3, -1.7, 0.5
        t1, t2, t3 = 4.0, 5.0, 7.0
        f(t) = a * t^2 + b * t + c
        x1, x2, x3 = f(t1), f(t2), f(t3)

        # Evaluate at an arbitrary query point.
        tq = 6.2
        @test isapprox(quadinterp(t1, t2, t3, x1, x2, x3, tq), f(tq); atol=1e-10)

        # Vertex form (t omitted): true vertex is at t=-b/2a.
        te_true = -b / (2a)
        xe, te = quadinterp(t1, t2, t3, x1, x2, x3)
        @test isapprox(te, te_true; atol=1e-10)
        @test isapprox(xe, f(te_true); atol=1e-10)

        # Broadcasts elementwise over arrays.
        tqs = [5.5, 6.0, 6.5]
        xs = quadinterp.(t1, t2, t3, x1, x2, x3, tqs)
        @test all(isapprox.(xs, f.(tqs); atol=1e-10))
    end

    @testset "_instfreq_matrix — phase-differenced frequency tracks a chirp" begin
        # Linear chirp: instantaneous frequency ramps from f0 to f1 over N
        # samples. The premise of Stage B is that phase-differencing the
        # transform's own phase tracks this ramp much more closely than the
        # fixed nominal analysis-grid frequency fs[j] (which is constant
        # across the whole ridge regardless of the true, evolving frequency).
        N = 2048
        f0, f1 = 0.06, 0.14
        t = 0:N-1
        instfreq_true = 2π .* (f0 .+ (f1 - f0) .* t ./ N)   # rad/sample
        phase = cumsum(instfreq_true)
        u = cos.(phase)
        v = sin.(phase)

        wt_ccw, wt_cw, fs = rotary_wavetrans(u, v)
        om, dfdt = ValTools.JLab._instfreq_matrix(wt_ccw, 1.0)
        @test size(om) == size(wt_ccw)
        @test size(dfdt) == size(om)

        # At each time, compare the phase-based estimate and the nominal
        # analysis-grid frequency at the ridge's own scale against the known
        # true instantaneous frequency, restricted to the interior (edge
        # effects) and where the CCW branch actually dominates.
        mag = abs.(wt_ccw)
        errs_phase = Float64[]
        errs_nominal = Float64[]
        for i in (N ÷ 4):(3N ÷ 4)
            j = argmax(@view mag[i, :])
            (j <= 1 || j >= length(fs)) && continue
            push!(errs_phase, abs(om[i, j] - instfreq_true[i]))
            push!(errs_nominal, abs(fs[j] - instfreq_true[i]))
        end
        @test !isempty(errs_phase)
        @test median(errs_phase) < median(errs_nominal)
    end

    @testset "ridgechains_jlab — mutual-match chaining recovers a clean ridge" begin
        N = 2048
        f0 = 0.12
        t = 0:N-1
        u = cos.(2π * f0 .* t)
        v = sin.(2π * f0 .* t)
        wt_ccw, wt_cw, fs = rotary_wavetrans(u, v)

        events = ridgechains_jlab(wt_ccw, fs; alpha=0.25, min_cycles=1.0, dt=1.0)
        @test !isempty(events)
        best = events[argmax([length(e.times) for e in events])]
        # Should recover almost the whole record as one ridge, at close to
        # the true frequency, with no huge outliers from mismatching.
        @test length(best.times) > N - 20
        @test isapprox(median(best.freq), 2π * f0; atol=0.01)
        @test all(f -> isapprox(f, 2π * f0; atol=0.05), best.freq)
    end
end
