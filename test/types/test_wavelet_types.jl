# Typed wavelet API: WaveletTransform + wavetrans/tiredecode methods on
# the TimeSeries* containers (self-contained in src — no extension needed,
# unlike the Multitaper-gated typed spectral methods).
using Test
using Dates
using Random
using Unitful
using ValTools
using ValTools.Types

@testset "WaveletTransform typed API" begin
    t0 = DateTime(2026, 1, 1)
    N = 200
    time = [t0 + Hour(i) for i in 0:N-1]
    raw = sin.(2π .* (0:N-1) ./ 20) .+ 0.1 .* randn(MersenneTwister(7), N)

    @testset "TimeSeriesVector -> WaveletTransform" begin
        ts = TimeSeriesVector(time, raw .* u"m/s", "u_east", (;))
        W = wavetrans(ts)

        @test W isa WaveletTransform
        @test size(W.wt) == (N, length(W.fs))
        @test W.time == time
        @test isempty(W.channels)
        @test W.params.dt_hours == 1.0
        @test W.params.unit == u"m/s"
        @test W.params.name == "u_east"

        # Matches the plain-array call with the derived dt
        wt_plain, fs_plain = wavetrans(raw; dt=1.0)
        @test W.fs == fs_plain
        @test W.wt ≈ wt_plain rtol=1e-12

        # kwargs forward and are recorded in params (beta=2.0 changes the
        # auto-generated grid density, so shapes legitimately differ too)
        W2 = wavetrans(ts; boundary=:mirror, beta=2.0)
        @test W2.params.boundary == :mirror
        @test W2.params.beta == 2.0
        @test size(W2.wt) != size(W.wt)

        # dt is derived from the time axis — passing it is an error
        @test_throws ErrorException wavetrans(ts; dt=1.0)
    end

    @testset "irregular time axis errors" begin
        bad_time = copy(time)
        bad_time[100] += Minute(30)
        ts_bad = TimeSeriesVector(bad_time, raw .* u"m/s", "u", (;))
        @test_throws ErrorException wavetrans(ts_bad)
    end

    @testset "dimensionless values stay plain" begin
        ts = TimeSeriesVector(time, copy(raw), "eta", (;))
        W = wavetrans(ts)
        @test W.params.unit == Unitful.NoUnits
        amp = tiredecode(W; kind="amp")
        @test eltype(amp) == Float64
    end

    @testset "tiredecode on WaveletTransform" begin
        ts = TimeSeriesVector(time, raw .* u"m/s", "u", (;))
        W = wavetrans(ts)

        amp = tiredecode(W; kind="amp")
        @test eltype(amp) <: Unitful.Quantity
        @test Unitful.unit(first(amp)) == u"m/s"
        @test Unitful.ustrip.(amp) ≈ tiredecode(W.wt, W.fs; kind="amp") rtol=1e-12

        fr = tiredecode(W; kind="freq")
        @test eltype(fr) == Float64
        @test size(fr) == size(W.wt)
    end

    @testset "TimeSeriesMatrix -> 3-D WaveletTransform with channels" begin
        vals = hcat(raw, reverse(raw), circshift(raw, 13)) .* u"cm/s"
        tsm = TimeSeriesMatrix(time, vals, ["a", "b", "c"], "moor", (;))
        W = wavetrans(tsm)

        @test W isa WaveletTransform
        @test size(W.wt) == (N, length(W.fs), 3)
        @test W.channels == ["a", "b", "c"]
        @test W.params.unit == u"cm/s"

        # Each channel slice matches the corresponding single-series call
        for k in 1:3
            tsk = TimeSeriesVector(time, vals[:, k], "ch$k", (;))
            Wk = wavetrans(tsk)
            @test W.wt[:, :, k] ≈ Wk.wt rtol=1e-12
        end

        amp = tiredecode(W; kind="amp")
        @test size(amp) == size(W.wt)
        @test Unitful.unit(first(amp)) == u"cm/s"
    end

    @testset "TimeSeriesCollection -> Vector{WaveletTransform} (ragged)" begin
        time2 = [t0 + Hour(i) for i in 0:149]  # different length
        s1 = TimeSeriesVector(time, raw .* u"m/s", "d1", (;))
        s2 = TimeSeriesVector(time2, raw[1:150] .* u"m/s", "d2", (;))
        tc = TimeSeriesCollection([s1, s2], "drifters", (;))

        Ws = wavetrans(tc)
        @test Ws isa Vector{<:WaveletTransform}
        @test length(Ws) == 2
        @test size(Ws[1].wt, 1) == N
        @test size(Ws[2].wt, 1) == 150
        # Ragged records resolve their own default grids (different N)
        @test length(Ws[1].fs) != length(Ws[2].fs)
        @test Ws[2].params.name == "d2"
    end
end
