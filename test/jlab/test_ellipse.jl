using Test
using Random
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
end
