using Test
using ValTools
using ValTools.Types
using ValTools.JLab
using Unitful
using Random

Random.seed!(1)

@testset "spectral_multitaper -> SpectralEstimate" begin

    @testset "single signal — dimensionless" begin
        x = randn(500)
        spec = spectral_multitaper(x, 0.1; nw=4.0)

        @test spec isa Types.SpectralEstimate
        @test length(spec.freq) == length(spec.power)
        @test all(spec.power .>= 0)
        @test spec.params.dt == 0.1
        @test spec.params.detrend == "linear"
    end

    @testset "single signal — with physical unit" begin
        x = randn(500)
        spec = spectral_multitaper(x, 0.1; nw=4.0, unit=u"m/s")

        @test eltype(spec.power) <: Unitful.Quantity
        @test Unitful.dimension(spec.power[1]) == Unitful.dimension(u"m^2/s^2")
    end

    @testset "batch (matrix) signals" begin
        X = randn(500, 4)
        specs = spectral_multitaper(X, 0.1; nw=4.0)

        @test specs isa Vector{<:Types.SpectralEstimate}
        @test length(specs) == 4
        @test all(length(s.freq) == length(specs[1].freq) for s in specs)
    end

    @testset "ftest_pval / jkvar populated" begin
        x = randn(500)
        spec = spectral_multitaper(x, 0.1; nw=4.0)

        @test spec.ftest_pval !== nothing
        @test spec.jkvar !== nothing
        @test length(spec.ftest_pval) == length(spec.freq)
        @test length(spec.jkvar) == length(spec.freq)
    end
end
