using Test
using ValTools
using ValTools.Types
using ValTools.Dispatch
using Unitful
using Dates

@testset "Unitful ops — arithmetic" begin
    t = Dates.now() .+ Dates.Second.(0:9)
    ts1 = TimeSeriesVector(t, fill(2.0, 10) * u"m/s", "a", (;))
    ts2 = TimeSeriesVector(t, fill(3.0, 10) * u"m/s", "b", (;))

    @testset "addition" begin
        s = ts1 + ts2
        @test all(Unitful.ustrip.(s.value) .≈ 5.0)
        @test Unitful.unit(s.value[1]) == u"m/s"
        @test s.name == "a + b"
    end

    @testset "subtraction" begin
        d = ts1 - ts2
        @test all(Unitful.ustrip.(d.value) .≈ -1.0)
        @test d.name == "a - b"
    end

    @testset "addition — mismatched time axis errors" begin
        t2 = t .+ Dates.Second(1)
        ts3 = TimeSeriesVector(t2, fill(1.0, 10) * u"m/s", "c", (;))
        @test_throws ErrorException ts1 + ts3
    end

    @testset "addition — mismatched length errors" begin
        ts_short = TimeSeriesVector(t[1:5], fill(1.0, 5) * u"m/s", "short", (;))
        @test_throws ErrorException ts1 + ts_short
    end

    @testset "addition — incompatible dimensions errors" begin
        ts_mass = TimeSeriesVector(t, fill(1.0, 10) * u"kg", "mass", (;))
        @test_throws Unitful.DimensionError ts1 + ts_mass
    end

    @testset "scalar multiplication" begin
        scaled = ts1 * 2.5
        @test all(Unitful.ustrip.(scaled.value) .≈ 5.0)
        scaled2 = 2.5 * ts1
        @test all(Unitful.ustrip.(scaled2.value) .≈ 5.0)
        @test Unitful.unit(scaled.value[1]) == u"m/s"
    end

    @testset "scalar division" begin
        divided = ts1 / 2.0
        @test all(Unitful.ustrip.(divided.value) .≈ 1.0)
        @test Unitful.unit(divided.value[1]) == u"m/s"
    end

    @testset "unit conversion" begin
        converted = Dispatch.convert_units(ts1, u"cm/s")
        @test Unitful.unit(converted.value[1]) == u"cm/s"
        @test all(Unitful.ustrip.(converted.value) .≈ 200.0)
    end

    @testset "unit_of" begin
        @test Dispatch.unit_of(ts1) == u"m/s"
    end

    @testset "unit_of — empty series returns nothing" begin
        ts_empty = TimeSeriesVector(DateTime[], Float64[] * u"m/s", "empty", (;))
        @test Dispatch.unit_of(ts_empty) === nothing
    end

    @testset "unit stripping" begin
        bare = Dispatch.strip_units(ts1)
        @test eltype(bare.value) == Float64
        @test all(bare.value .≈ 2.0)
    end

    @testset "scale_to_unit" begin
        scaled = Dispatch.scale_to_unit(ts1, 1.0u"cm/s")
        @test Unitful.unit(scaled.value[1]) == u"cm/s"
        @test all(Unitful.ustrip.(scaled.value) .≈ 200.0)
    end
end
