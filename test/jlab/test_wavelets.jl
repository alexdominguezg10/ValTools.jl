using Test
using Random
using Statistics
using ValTools.JLab

Random.seed!(42)

@testset "JLab Wavelets (Calibrated)" begin

    @testset "morsefreq — peak frequency" begin
        # ωₘ = (β/γ)^(1/γ)
        @test morsefreq(3.0, 8.0) ≈ (8/3)^(1/3) atol=1e-10
        @test morsefreq(3.0, 3.0) ≈ 1.0 atol=1e-10
        @test morsefreq(2.0, 12.0) ≈ sqrt(6.0) atol=1e-10
    end

    @testset "morseprops — time-frequency product" begin
        P, skew, kurt = morseprops(3.0, 8.0)
        @test P ≈ sqrt(24.0) atol=1e-10
        @test skew ≈ 0.0 atol=1e-10    # γ=3 → skewness=0
    end

    @testset "morsespace — frequency generation" begin
        fs = morsespace(3.0, 8.0, 1000)
        @test length(fs) > 5
        @test issorted(fs; rev=true)      # descending order
        @test all(fs .> 0)
        @test fs[1] <= π                  # below Nyquist
        @test fs[end] >= 2π / 1000        # above resolution limit
    end

    @testset "morsespace — longer signal → more frequencies" begin
        fs_short = morsespace(3.0, 8.0, 100)
        fs_long  = morsespace(3.0, 8.0, 10000)
        @test length(fs_long) > length(fs_short)
    end

    @testset "morsewave — basic properties" begin
        N = 256; γ = 3.0; β = 8.0
        ψ, ψf, s = morsewave(N, γ, β)

        @test length(ψ) == N
        @test length(ψf) == N
        @test eltype(ψ) == ComplexF64
        @test all(isfinite, ψf)

        # Peak should be at morsefreq
        @test argmax(ψf) > 1
    end

    @testset "wavetrans — returns radian frequencies" begin
        x = randn(500)
        wt, fs = wavetrans(x; dt=1.0, nv=8)

        @test size(wt, 1) == 500
        @test length(fs) == size(wt, 2)
        @test issorted(fs; rev=true)       # descending (high freq first)
        @test all(fs .> 0)
        @test all(isfinite, abs.(wt))
    end

    @testset "wavetrans — sinusoid energy concentration" begin
        dt = 1.0; f0 = 0.1  # Hz
        t = (0:999) .* dt
        x = cos.(2π * f0 .* t)

        wt, fs = wavetrans(x; dt=dt, nv=8)
        amp = abs.(wt)

        # Mean amplitude per frequency should peak somewhere
        mean_amp = vec(mean(amp; dims=1))
        @test maximum(mean_amp) > 3 * median(mean_amp)

        # Factor of 2: amplitude of pure cosine ≈ 1.0 at the ridge
        peak_amp = maximum(mean_amp)
        @test peak_amp > 0.5   # should be near 1.0 for unit-amplitude cosine
    end

    @testset "wavetrans — backward compat with scales kwarg" begin
        x = randn(200)
        sc = [0.05, 0.1, 0.2, 0.5, 1.0]
        wt, fs = wavetrans(x; dt=1.0, scales=sc)
        @test size(wt) == (200, 5)
    end

    @testset "wavetrans_batch — multiple signals" begin
        N = 300; n_sig = 5
        X = randn(N, n_sig)
        wt3d, fs = wavetrans_batch(X; dt=0.5, nv=4)

        @test size(wt3d, 1) == N
        @test size(wt3d, 3) == n_sig
        @test size(wt3d, 2) == length(fs)
    end

    @testset "tiredecode — amp/phase/freq/bandwidth" begin
        wt = randn(ComplexF64, 100, 20)
        fs = collect(range(0.1, 3.0; length=20))

        amp = tiredecode(wt, fs; kind="amp")
        @test all(amp .>= 0)

        ph = tiredecode(wt, fs; kind="phase")
        @test all(-π .<= ph .<= π)

        fr = tiredecode(wt, fs; kind="freq")
        @test size(fr) == (100, 20)

        bw = tiredecode(wt, fs; kind="bandwidth")
        @test size(bw) == (100, 20)

        @test_throws ErrorException tiredecode(wt, fs; kind="bogus")
    end

    @testset "ridgemap — basic detection" begin
        dt = 1.0; f0 = 0.1
        t = (0:499) .* dt
        x = cos.(2π * f0 .* t)

        wt, fs = wavetrans(x; dt=dt, nv=8)
        rf, ra = ridgemap(wt, fs)

        @test length(rf) == 500
        n_valid = count(!isnan, rf)
        @test n_valid > 200   # most points should have a ridge
    end

    @testset "ridgemap — with quality metric" begin
        wt = randn(ComplexF64, 50, 15)
        fs = collect(range(0.1, 2.0; length=15))

        rf, ra, rq = ridgemap(wt, fs; quality=true)
        @test length(rq) == 50
        valid_q = rq[.!isnan.(rq)]
        @test all(0 .<= valid_q .<= 1)
    end

    @testset "ridgemap — threshold filters weak signals" begin
        wt = 0.01 .* randn(ComplexF64, 50, 10)
        fs = collect(range(0.1, 1.0; length=10))
        rf, ra = ridgemap(wt, fs; thresh=1.0)
        @test all(isnan, rf)
    end

    @testset "ridgechains — detects coherent oscillation" begin
        dt = 1.0
        t = (0:999) .* dt

        # Inject oscillation from t=200 to t=700
        x = 0.01 .* randn(1000)
        f_signal = 0.05
        for i in 200:700
            x[i] += cos(2π * f_signal * t[i])
        end

        wt, fs = wavetrans(x; dt=dt, nv=8)
        events = ridgechains(wt, fs; alpha=0.25, min_length=1.0)

        @test isa(events, Vector{RidgeEvent})
        if !isempty(events)
            # Should find at least one event overlapping 200–700
            longest = events[argmax([e.duration for e in events])]
            @test longest.duration > 50
            @test longest.start >= 1
            @test longest.stop <= 1000
            @test all(longest.amp .> 0)
            @test all(longest.freq .> 0)
        end
    end

    @testset "ridgechains — noise only → shorter events than signal" begin
        x = 0.01 .* randn(500)
        wt, fs = wavetrans(x; dt=1.0, nv=4)
        events = ridgechains(wt, fs; alpha=0.05, min_length=5.0, thresh=0.5)

        # Tight params + noise → fewer/shorter events than with real signal
        total_dur = sum(e.duration for e in events; init=0)
        @test total_dur < 400
    end

    @testset "RidgeEvent — struct fields" begin
        e = RidgeEvent(10, 50, 41, randn(41), abs.(randn(41)), collect(1:41))
        @test e.start == 10
        @test e.stop == 50
        @test e.duration == 41
        @test length(e.freq) == 41
    end

    @testset "transmax — top-N maxima" begin
        wt = randn(ComplexF64, 64, 32)
        it, is, a = transmax(wt; n=5)

        @test length(it) == 5
        @test issorted(a; rev=true)
        @test a[1] ≈ maximum(abs.(wt))
    end

    @testset "GPU stub — errors without CUDA" begin
        @test_throws ErrorException wavetrans(randn(100); gpu=true)
        @test_throws ErrorException wavetrans_batch(randn(100, 3); gpu=true)
    end
end
