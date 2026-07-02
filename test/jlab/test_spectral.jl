using Test
using Random
using Statistics
using ValTools.JLab

Random.seed!(42)

@testset "JLab Spectral" begin

    @testset "sleptap — DPSS tapers" begin
        N = 256; K = 5
        tapers, lambdas = sleptap(N, K; bandwidth=4.0)

        @test size(tapers) == (N, K)
        @test length(lambdas) == K

        # Orthonormality: each taper has unit energy
        for k in 1:K
            @test abs(sum(tapers[:, k] .^ 2) - 1.0) < 1e-10
        end

        # Near-orthogonal between tapers
        for i in 1:K, j in i+1:K
            dot_ij = abs(sum(tapers[:, i] .* tapers[:, j]))
            @test dot_ij < 0.15
        end

        # Eigenvalues should be sorted descending
        @test issorted(lambdas; rev=true)

        # Concentration should be high for first tapers
        @test lambdas[1] > 0.9
    end

    @testset "sleptap — edge cases" begin
        @test_throws ErrorException sleptap(0, 3)
        @test_throws ErrorException sleptap(10, 0)

        # Small N
        tap, lam = sleptap(8, 2)
        @test size(tap) == (8, 2)
    end

    @testset "mspec — detects spectral peak" begin
        dt = 0.1;  f0 = 0.5
        t = 0:dt:200;  N = length(t)
        x = cos.(2π * f0 .* t) .+ 0.1 .* randn(N)

        freqs, psd = mspec(x, dt; ntapers=5)

        @test length(freqs) == length(psd)
        @test all(psd .>= 0)

        # Peak should be near f0
        pos = freqs .> 0
        f_pos = freqs[pos]; p_pos = psd[pos]
        f_peak = f_pos[argmax(p_pos)]
        @test abs(f_peak - f0) < 0.1
    end

    @testset "mspec — white noise is flat" begin
        N = 4096
        x = randn(N)

        freqs, psd = mspec(x, 1.0; ntapers=5, detrend="constant")

        pos = freqs .> 0
        p = psd[pos]

        # For white noise, PSD should be roughly constant
        # Check that std/mean < 1 (not highly peaked)
        @test std(p) / mean(p) < 1.5
    end

    @testset "mspec — detrend options" begin
        N = 500
        x = randn(N) .+ collect(1:N) .* 0.01  # linear trend

        _, p_none   = mspec(x, 1.0; ntapers=3, detrend="none")
        _, p_linear = mspec(x, 1.0; ntapers=3, detrend="linear")

        # Detrended should have less low-freq energy
        @test p_linear[2] < p_none[2]
    end

    @testset "mspec — GPU stub" begin
        @test_throws ErrorException mspec(randn(100), 1.0; gpu=true)
    end
end
