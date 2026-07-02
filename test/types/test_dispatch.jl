using Test
using ValTools
using ValTools.Types
using ValTools.Dispatch
using Statistics
using Unitful
using Dates

@testset "Dispatch — unit-preserving statistics" begin
    t = Dates.now() .+ Dates.Second.(0:99)
    v = randn(100) .* u"m/s" .+ 5.0u"m/s"
    ts = TimeSeriesVector(t, v, "u", (;))

    @testset "mean/std/var preserve units" begin
        m = Dispatch.mean(ts)
        s = Dispatch.std(ts)
        va = Dispatch.var(ts)

        @test Unitful.unit(m) == u"m/s"
        @test Unitful.unit(s) == u"m/s"
        @test Unitful.unit(va) == u"m^2/s^2"
        @test abs(Unitful.ustrip(m) - 5.0) < 1.0
    end

    @testset "TimeSeriesMatrix mean/std/var/cov" begin
        tm = TimeSeriesMatrix(t, randn(100, 3) .* u"m/s", ["a", "b", "c"], "mat", (;))
        m = Dispatch.mean(tm)
        @test size(m) == (1, 3)
        @test Unitful.unit(m[1]) == u"m/s"

        c = Dispatch.cov(tm)
        @test size(c) == (3, 3)
    end
end

@testset "Dispatch — validation metrics" begin
    t = Dates.now() .+ Dates.Second.(0:199)
    obs_val = randn(200) .* u"m/s"
    model = TimeSeriesVector(t, obs_val .+ 0.1u"m/s", "model", (;))
    obs = TimeSeriesVector(t, obs_val, "obs", (;))

    @testset "rmse" begin
        r = Dispatch.rmse(model, obs)
        @test Unitful.unit(r) == u"m/s"
        @test Unitful.ustrip(r) ≈ 0.1 atol=1e-8
    end

    @testset "rmse — perfect match is zero" begin
        r0 = Dispatch.rmse(obs, obs)
        @test Unitful.ustrip(r0) ≈ 0.0 atol=1e-10
    end

    @testset "correlation — unitless, invariant to constant offset" begin
        c = Dispatch.correlation(model, obs)
        @test c isa Real
        @test c ≈ 1.0 atol=1e-8
    end

    @testset "skill_score" begin
        sk = Dispatch.skill_score(model, obs)
        @test sk isa Real
        @test sk > 0.9
    end

    @testset "validate — bundled metrics" begin
        result = Dispatch.validate(model, obs)
        @test haskey(result, :rmse)
        @test haskey(result, :correlation)
        @test haskey(result, :skill)
        @test haskey(result, :bias)
        @test result.bias ≈ 0.1 atol=1e-8
    end
end
