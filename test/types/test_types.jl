using Test
using ValTools
using ValTools.Types
using Unitful
using Dates

@testset "Types — construction" begin

    @testset "TimeSeriesVector — basic construction" begin
        t = Dates.now() .+ Dates.Second.(0:9)
        v = randn(10) * u"m/s"
        ts = TimeSeriesVector(t, v, "velocity", (;))

        @test ts isa TimeSeriesVector
        @test length(ts.time) == 10
        @test length(ts.value) == 10
        @test ts.name == "velocity"
        @test eltype(ts.value) <: Unitful.Quantity
        @test Unitful.unit(ts.value[1]) == u"m/s"
    end

    @testset "TimeSeriesVector — type parameter tracks the unit" begin
        t = Dates.now() .+ Dates.Second.(0:4)
        ts_ms = TimeSeriesVector(t, randn(5) * u"m/s", "a", (;))
        ts_cms = TimeSeriesVector(t, randn(5) * u"cm/s", "b", (;))

        @test typeof(ts_ms) != typeof(ts_cms)
        @test ts_ms isa TimeSeriesVector{<:Unitful.Quantity}
    end

    @testset "TimeSeriesVector — dimensionless (stripped) values" begin
        t = Dates.now() .+ Dates.Second.(0:4)
        ts = TimeSeriesVector(t, randn(5), "bare", (;))
        @test eltype(ts.value) == Float64
        @test !(eltype(ts.value) <: Unitful.Quantity)
    end

    @testset "TimeSeriesMatrix — basic construction" begin
        t = Dates.now() .+ Dates.Second.(0:9)
        v = randn(10, 3) * u"m/s"
        tm = TimeSeriesMatrix(t, v, ["ch1", "ch2", "ch3"], "currents", (;))

        @test size(tm.value) == (10, 3)
        @test length(tm.channels) == 3
        @test Unitful.unit(tm.value[1, 1]) == u"m/s"
    end

    @testset "SpectralEstimate — basic construction" begin
        freq = collect(0.0:0.1:1.0)
        power = rand(length(freq)) * u"m^2/s^2"
        se = SpectralEstimate(freq, power, nothing, nothing, (nw=4.0,))

        @test length(se.freq) == length(se.power)
        @test se.ftest_pval === nothing
        @test se.jkvar === nothing
        @test se.params.nw == 4.0
    end

    @testset "ObsMetadata — basic construction" begin
        m = ObsMetadata("NDBC", "m/s", [true, true, false], Dates.now(), "ADCP",
                         (lat=27.5, lon=-90.0))
        @test m.source == "NDBC"
        @test length(m.qc_flags) == 3
    end

    @testset "ColocatedObservation — basic construction" begin
        t = Dates.now() .+ Dates.Second.(0:4)
        model = TimeSeriesVector(t, randn(5) * u"m/s", "model", (;))
        obs = TimeSeriesVector(t, randn(5) * u"m/s", "obs", (;))
        co = ColocatedObservation(model, obs, 1.5, (rmse=0.2,))

        @test co.distance == 1.5
        @test co.metrics.rmse == 0.2
    end
end
