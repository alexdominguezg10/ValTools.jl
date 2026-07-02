using Test
using Random
using Statistics
using ValTools.JLab

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
end
