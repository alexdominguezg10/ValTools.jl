# Multivariate instantaneous moments (instmom) + multivariate wavelet ridge
# analysis (Lilly & Olhede 2012). Dependency-free (no Multitaper needed),
# matching test_wavelets.jl's convention.
using Test
using Random
using Statistics
using ValTools.JLab
using ValTools.JLab: _instfreq_joint, _instfreq_joint_nd

@testset "JLab Multivariate Moments & Ridges" begin

    @testset "instmom (univariate) — matches tiredecode kinds" begin
        rng = MersenneTwister(21)
        x = cos.(2π * 0.15 .* (0:249)) .+ 0.05 .* randn(rng, 250)
        wt, fs = wavetrans(x)

        a, om, upsilon, xi = instmom(wt)
        @test a ≈ tiredecode(wt, fs; kind="amp")
        @test om ≈ tiredecode(wt, fs; kind="freq")
        @test upsilon ≈ tiredecode(wt, fs; kind="bandwidth")
        @test xi ≈ tiredecode(wt, fs; kind="curvature")
        @test eltype(xi) <: Complex
    end

    @testset "instmom (univariate) — curvature vanishes for a pure complex exponential" begin
        # x(t) = exp(i*om0*t): constant amplitude, constant frequency, so
        # upsilon=0 and om=om0 everywhere (away from endpoints), hence
        # xi = upsilon^2 + d/dt(upsilon) + i*d/dt(om) = 0 exactly in the interior.
        N = 200
        om0 = 0.3
        w = reshape(exp.(im .* om0 .* (0:N-1)), N, 1)
        a, om, upsilon, xi = instmom(w)
        interior = 3:N-2
        @test all(isapprox.(a[interior, 1], 1.0; atol=1e-10))
        @test all(isapprox.(om[interior, 1], om0; atol=1e-8))
        @test all(isapprox.(upsilon[interior, 1], 0.0; atol=1e-8))
        @test all(isapprox.(abs.(xi[interior, 1]), 0.0; atol=1e-6))
    end

    @testset "instmom (joint) — single-channel (N=1) reduces to univariate" begin
        rng = MersenneTwister(22)
        x = cos.(2π * 0.2 .* (0:199)) .+ 0.05 .* randn(rng, 200)
        wt, fs = wavetrans(x)
        a, om, upsilon, xi = instmom(wt)

        W = reshape(wt, size(wt)..., 1)
        ja, jomega, jupsilon, jxi = instmom(W)
        @test ja ≈ a rtol=1e-10
        @test jomega ≈ om rtol=1e-10
        @test jupsilon ≈ abs.(upsilon) rtol=1e-10  # jupsilon is a magnitude; sign of upsilon alone doesn't survive |.|
        @test jxi ≈ abs.(xi) rtol=1e-10
    end

    @testset "instmom (joint) — output shape and finiteness on a 3-channel signal" begin
        rng = MersenneTwister(23)
        N = 300
        t = collect(0.0:N-1)
        phase = cumsum(0.5 .+ 0.0005 .* t)
        X = hcat(2.0 .* cos.(phase), 1.0 .* cos.(phase .+ 0.3), 0.5 .* cos.(phase .+ 1.1)) .+
            0.02 .* randn(rng, N, 3)
        wt3, fs = wavetrans(X)
        ja, jomega, jupsilon, jxi = instmom(wt3)

        @test size(ja) == (N, length(fs)) == size(jomega) == size(jupsilon) == size(jxi)
        @test all(isfinite, ja) && all(ja .>= 0)
        @test all(isfinite, jomega)
        @test all(isfinite, jupsilon) && all(jupsilon .>= 0)
        @test all(isfinite, jxi) && all(jxi .>= 0)   # jxi is a magnitude (sqrt of a sum of squares)
    end

    @testset "_instfreq_joint_nd — N=2 consistency with the validated rotary _instfreq_joint" begin
        rng = MersenneTwister(24)
        N = 300
        u = randn(rng, N)
        v = 0.6 .* circshift(u, 5) .+ 0.3 .* randn(rng, N)

        wt_ccw, wt_cw, fs = rotary_wavetrans(u, v)
        om1, dfdt1 = _instfreq_joint(wt_ccw, wt_cw, 1.0)

        wx = (wt_ccw .+ wt_cw) ./ sqrt(2)
        wy = .-im .* (wt_ccw .- wt_cw) ./ sqrt(2)
        wt3 = cat(wx, wy; dims=3)
        om2, dfdt2 = _instfreq_joint_nd(wt3, 1.0)

        @test om1 ≈ om2 rtol=1e-12
        @test dfdt1 ≈ dfdt2 rtol=1e-12
    end

    @testset "multivariate_ridges — synthetic common chirp, N=3" begin
        N = 300
        t = collect(0.0:N-1)
        phase = cumsum(0.5 .+ 0.0005 .* t)  # frequency well within the default grid's resolvable band
        c1 = 2.0 .* cos.(phase)
        c2 = 1.0 .* cos.(phase .+ 0.3)
        c3 = 0.5 .* cos.(phase .+ 1.1)
        X = hcat(c1, c2, c3)

        ridges = multivariate_ridges(X)
        @test length(ridges) == 1
        r = only(ridges)
        @test r.npoints == N
        @test r.start == 1 && r.stop == N

        true_freq_mean = sum(0.5 .+ 0.0005 .* t) / N
        @test r.omega_bar ≈ true_freq_mean atol=0.02

        true_kappa = sqrt(2.0^2 + 1.0^2 + 0.5^2)  # Euclidean-norm convention
        @test r.kappa_bar ≈ true_kappa rtol=0.05

        @test size(r.wt_ridge) == (N, 3)
        # wt_ridge should recover each channel's relative amplitude (2:1:0.5)
        amp_ridge = vec(mean(abs.(r.wt_ridge); dims=1))
        @test amp_ridge[1] / amp_ridge[2] ≈ 2.0 rtol=0.1
        @test amp_ridge[1] / amp_ridge[3] ≈ 4.0 rtol=0.15
    end

    @testset "multivariate_ridges — pure noise yields no comparably long ridge" begin
        rng = MersenneTwister(25)
        N = 300
        Xn = randn(rng, N, 3)
        ridges_signal = multivariate_ridges(
            hcat(2.0 .* cos.(cumsum(fill(0.55, N))),
                 1.0 .* cos.(cumsum(fill(0.55, N)) .+ 0.3),
                 0.5 .* cos.(cumsum(fill(0.55, N)) .+ 1.1)))
        ridges_noise = multivariate_ridges(Xn)

        @test maximum(r -> r.npoints, ridges_signal) == N
        @test isempty(ridges_noise) || maximum(r -> r.npoints, ridges_noise) < N ÷ 2
    end

    @testset "multivariate_ridges — varargs form matches matrix form" begin
        N = 200
        phase = cumsum(fill(0.6, N))
        x1, x2, x3 = cos.(phase), sin.(phase), cos.(phase .+ 0.5)
        r_vec = multivariate_ridges(x1, x2, x3)
        r_mat = multivariate_ridges(hcat(x1, x2, x3))
        @test length(r_vec) == length(r_mat)
        @test all(r_vec[i].wt_ridge == r_mat[i].wt_ridge for i in eachindex(r_vec))
    end

    @testset "multivariate_ridges — argument errors" begin
        @test_throws ErrorException multivariate_ridges(randn(100, 1))
    end
end
