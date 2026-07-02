using Test
using Random
using Statistics
using ValTools.JLab

Random.seed!(42)

@testset "JLab TimeSeries" begin

    @testset "detrend — linear" begin
        x = collect(1.0:100.0) .+ 0.1 .* randn(100)   # strong linear trend
        xd, trend, coeffs = detrend(x; order=1)

        # Detrended should be near zero mean
        @test abs(mean(xd)) < 0.5

        # Trend should capture the linear component
        @test coeffs[1] ≈ 1.0 atol=0.1   # slope ≈ 1

        # Dimensions preserved
        @test length(xd) == 100
        @test length(trend) == 100
    end

    @testset "detrend — quadratic" begin
        t = collect(1.0:200.0)
        x = 0.001 .* t .^ 2 .+ randn(200)

        xd, trend, coeffs = detrend(x; order=2)
        @test abs(mean(xd)) < 2.0
        @test length(coeffs) == 3
    end

    @testset "detrend — constant (demean)" begin
        x = 5.0 .+ randn(50)
        xd, trend, coeffs = detrend(x; kind="constant")

        @test abs(mean(xd)) < 1e-10
        # Coefficient should be near 5.0 (mean of 5 + noise)
        @test coeffs[1] ≈ 5.0 atol=1.0
    end

    @testset "bandpass — isolates frequency band" begin
        dt = 0.1;  N = 1000
        t = (0:N-1) .* dt
        # Two components: 0.2 Hz and 2.0 Hz
        x = cos.(2π * 0.2 .* t) .+ cos.(2π * 2.0 .* t)

        x_bp = bandpass(x, dt, 0.1, 0.5)

        # After bandpass [0.1, 0.5] Hz, the 2.0 Hz component should be gone
        # Check: filtered signal should be closer to the 0.2 Hz component
        ref = cos.(2π * 0.2 .* t)
        corr_filtered = abs(sum(x_bp .* ref)) / (sqrt(sum(x_bp.^2)) * sqrt(sum(ref.^2)))
        @test corr_filtered > 0.7
    end

    @testset "highpass — removes low frequencies" begin
        dt = 0.1;  N = 500
        t = (0:N-1) .* dt
        x = cos.(2π * 0.05 .* t) .+ cos.(2π * 2.0 .* t)

        x_hp = highpass(x, dt, 1.0)

        # Low-freq component should be attenuated
        @test std(x_hp) < std(x)
    end

    @testset "lowpass — removes high frequencies" begin
        dt = 0.1;  N = 500
        t = (0:N-1) .* dt
        x = cos.(2π * 0.05 .* t) .+ cos.(2π * 2.0 .* t)

        x_lp = lowpass(x, dt, 0.5)

        # High-freq component should be attenuated
        @test std(x_lp) < std(x)
    end

    @testset "fillgaps — linear interpolation" begin
        x = collect(1.0:20.0)
        x[5] = NaN;  x[6] = NaN;  x[10] = NaN

        xf = fillgaps(x; method="linear", max_gap=10)

        @test xf[5] ≈ 5.0
        @test xf[6] ≈ 6.0
        @test xf[10] ≈ 10.0
        @test !any(isnan, xf)
    end

    @testset "fillgaps — respects max_gap" begin
        x = collect(1.0:20.0)
        x[5:15] .= NaN   # gap of 11

        xf = fillgaps(x; max_gap=5)

        # Gap too large → should remain NaN
        @test any(isnan, xf[5:15])
    end

    @testset "fillgaps — no valid data" begin
        x = fill(NaN, 10)
        xf = fillgaps(x)
        @test all(isnan, xf)
    end

    @testset "hilbert — analytic signal" begin
        N = 256
        t = collect(0:N-1) ./ N
        x = cos.(2π * 5.0 .* t)

        z = hilbert(x)

        @test length(z) == N
        @test eltype(z) <: Complex

        # Envelope of pure cosine should be ~1
        env = abs.(z)
        @test mean(env) ≈ 1.0 atol=0.15

        # Phase should increase linearly for pure tone
        ph = angle.(z)
        @test !any(isnan, ph)
    end

    @testset "hilbert — real part preserved" begin
        x = randn(128)
        z = hilbert(x)

        # Real part of analytic signal should match input
        @test maximum(abs.(real.(z) .- x)) < 1e-10
    end
end
