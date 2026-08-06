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

    @testset "wavelet_significance — signal exceeds noise threshold" begin
        N = 300
        dt = 1.0
        t = (0:N-1) .* dt
        f0 = 0.3  # rad/sample

        x = 2.0 .* cos.(f0 .* t) .+ 0.3 .* randn(MersenneTwister(1), N)

        wt, fs = wavetrans(x; dt=dt, nv=8)
        rf, ra = ridgemap(wt, fs)

        sig_level, fs_sig = wavelet_significance(x; dt=dt, fs=fs, n_surrogates=100,
                                                  rng=MersenneTwister(99))
        @test fs_sig == fs
        @test length(sig_level) == length(fs)
        @test all(sig_level .> 0)

        sig_flags = ridge_significant(rf, ra, sig_level, fs)
        valid = .!isnan.(rf)
        @test mean(sig_flags[valid]) > 0.7
    end

    @testset "wavelet_significance — pure noise mostly insignificant" begin
        N = 300
        xn = randn(MersenneTwister(2), N)

        wt, fs = wavetrans(xn; dt=1.0, nv=8)
        rf, ra = ridgemap(wt, fs)

        sig_level, _ = wavelet_significance(xn; dt=1.0, fs=fs, n_surrogates=100,
                                             rng=MersenneTwister(77))
        sig_flags = ridge_significant(rf, ra, sig_level, fs)
        valid = .!isnan.(rf)
        @test mean(sig_flags[valid]) < 0.5
    end

    @testset "wavelet_significance — AR(1) red-noise background" begin
        function _make_ar1(N, alpha, rng)
            z = zeros(N)
            z[1] = randn(rng)
            for i in 2:N
                z[i] = alpha * z[i-1] + randn(rng)
            end
            return z
        end

        xr = _make_ar1(300, 0.6, MersenneTwister(3))
        sig_level, fs = wavelet_significance(xr; background=:red, n_surrogates=80,
                                              rng=MersenneTwister(55))
        @test length(sig_level) == length(fs)
        @test all(sig_level .> 0)
    end

    @testset "wavelet_significance — argument errors" begin
        @test_throws ErrorException wavelet_significance(randn(3))
        @test_throws ErrorException wavelet_significance(randn(50); background=:pink)
        @test_throws ErrorException wavelet_significance(randn(50); n_surrogates=1)
    end

    @testset "ridge_significant — length checks and NaN handling" begin
        fs = [1.0, 0.5, 0.25]
        sig_level = [1.0, 1.0, 1.0]
        rf = [0.5, NaN, 1.0]
        ra = [2.0, 5.0, 0.1]

        flags = ridge_significant(rf, ra, sig_level, fs)
        @test flags == [true, false, false]

        @test_throws ErrorException ridge_significant([1.0], [1.0, 2.0], sig_level, fs)
        @test_throws ErrorException ridge_significant(rf, ra, [1.0], fs)
    end

    # ========================================================================
    # N-D SUPPORT (jLab trailing-dims semantics: dim 1 = time, trailing
    # dims = independent signals). The per-column equivalence tests below
    # are the regression guard for the unified transform engine.
    # ========================================================================

    @testset "wavetrans N-D — per-column equivalence" begin
        rng = MersenneTwister(11)
        N = 200
        X = randn(rng, N, 3)

        for boundary in (:zeros, :mirror)
            wt3, fs3 = wavetrans(X; boundary=boundary)
            @test size(wt3) == (N, length(fs3), 3)
            for k in 1:3
                wtk, fsk = wavetrans(X[:, k]; boundary=boundary)
                @test fsk == fs3
                @test wt3[:, :, k] ≈ wtk rtol=1e-12
            end
        end

        # Rank-3 input (N, 2, 2): trailing dims preserved, slices match
        X4 = reshape(randn(rng, N * 4), N, 2, 2)
        wt4, fs4 = wavetrans(X4)
        @test size(wt4) == (N, length(fs4), 2, 2)
        for i in 1:2, j in 1:2
            wtk, _ = wavetrans(X4[:, i, j])
            @test wt4[:, :, i, j] ≈ wtk rtol=1e-12
        end
    end

    @testset "wavetrans — multi-input jLab form" begin
        rng = MersenneTwister(12)
        N = 150
        x, y, z = randn(rng, N), randn(rng, N), randn(rng, N)

        wx, wy, wz, fs = wavetrans(x, y, z)
        wx1, fs1 = wavetrans(x)
        @test fs == fs1
        @test wx ≈ wx1 rtol=1e-12
        @test wy ≈ wavetrans(y)[1] rtol=1e-12
        @test wz ≈ wavetrans(z)[1] rtol=1e-12

        # Matrix inputs stack along a new trailing dim
        A, B = randn(rng, N, 2), randn(rng, N, 2)
        wA, wB, fsm = wavetrans(A, B)
        @test size(wA) == (N, length(fsm), 2)
        @test wA ≈ wavetrans(A)[1] rtol=1e-12

        @test_throws ErrorException wavetrans(x, randn(rng, N + 1))
        @test_throws ErrorException wavetrans(x, complex.(y))  # mixed real/complex
    end

    @testset "wavetrans — complex input vs rotary_wavetrans (1/sqrt(2) factor)" begin
        rng = MersenneTwister(13)
        N = 200
        u = randn(rng, N)
        v = randn(rng, N)
        w = complex.(u, v)

        wtc, fsc = wavetrans(w)
        @test size(wtc) == (N, length(fsc))
        # rotary_wavetrans's CCW branch is jLab's WP=(1/sqrt(2))*(WX+iWY)
        # convention (wavetrans.m); plain wavetrans(complex(u,v)) has no
        # such convention (it's ValTools' own general complex-signal path,
        # with no jLab equivalent to match), so the two differ by exactly
        # that factor -- not equal, as an earlier version of this test
        # (written before the missing-1/sqrt(2) bug was found and fixed
        # 2026-08-05) incorrectly asserted.
        ccw, _, fsr = rotary_wavetrans(u, v)
        @test fsc == fsr
        @test wtc ./ sqrt(2) ≈ ccw rtol=1e-12
    end

    @testset "rotary_wavetrans — N-D per-column equivalence" begin
        rng = MersenneTwister(14)
        N = 180
        U, V = randn(rng, N, 3), randn(rng, N, 3)

        ccw3, cw3, fs3 = rotary_wavetrans(U, V)
        @test size(ccw3) == (N, length(fs3), 3)
        for k in 1:3
            ccwk, cwk, fsk = rotary_wavetrans(U[:, k], V[:, k])
            @test fsk == fs3
            @test ccw3[:, :, k] ≈ ccwk rtol=1e-12
            @test cw3[:, :, k] ≈ cwk rtol=1e-12
        end

        @test_throws ErrorException rotary_wavetrans(U, randn(rng, N, 2))
    end

    @testset "tiredecode — N-D trailing dims" begin
        rng = MersenneTwister(15)
        X = randn(rng, 120, 3)
        wt3, fs = wavetrans(X)

        for kind in ("amp", "phase", "freq", "bandwidth")
            out3 = tiredecode(wt3, fs; kind=kind)
            @test size(out3) == size(wt3)
            for k in 1:3
                @test out3[:, :, k] ≈ tiredecode(wt3[:, :, k], fs; kind=kind) rtol=1e-12
            end
        end
        @test_throws ErrorException tiredecode(wt3, fs; kind="bogus")
        @test_throws ErrorException tiredecode(wt3, fs[1:3]; kind="amp")
    end

    @testset "ridge functions — N-D input guards" begin
        wt3 = randn(ComplexF64, 50, 10, 2)
        fs = collect(range(1.0, 0.1; length=10))
        @test_throws ErrorException ridgemap(wt3, fs)
        @test_throws ErrorException ridgechains(wt3, fs)
        @test_throws ErrorException ridgechains_jlab(wt3, fs)
        @test_throws ErrorException transmax(wt3)
    end

    @testset "wavetrans_batch — forwards to wavetrans, honors boundary" begin
        rng = MersenneTwister(16)
        X = randn(rng, 128, 4)
        wt_b, fs_b = wavetrans_batch(X)
        wt_n, fs_n = wavetrans(X)
        @test fs_b == fs_n
        @test wt_b == wt_n
        # boundary was silently unsupported by the old standalone batch path
        wt_bm, _ = wavetrans_batch(X; boundary=:mirror)
        @test wt_bm ≈ wavetrans(X; boundary=:mirror)[1] rtol=1e-12
        @test !(wt_bm ≈ wt_b)
    end

    @testset "wavelet_significance — gpu kwarg stub errors without CUDA" begin
        @test_throws ErrorException wavelet_significance(randn(64); n_surrogates=4, gpu=true)
    end
end
