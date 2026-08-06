using Test
using ValTools
using DataFrames
using Dates
using Statistics
using Random
using NCDatasets
using CairoMakie
using Multitaper
using Unitful
using LinearAlgebra

@testset "ValTools.jl" begin

    # Previously written but never actually run: test/types/runtests.jl
    # (test_types/test_dispatch/test_unitful_ops/test_spectral_estimate)
    # wasn't included anywhere, so `Pkg.test("ValTools")` silently skipped
    # all of it. Each sub-file declares its own `using` statements and is
    # self-contained.
    include("types/runtests.jl")

    # Same bug, same fix, found 2026-08-01 while adding ellpol/ellsig:
    # test/jlab/runtests.jl (test_wavelets/test_spectral/test_timeseries/
    # test_ellipse/test_validation/test_jdata -- everything covering the
    # wavelet ridge, rotary, and ellipse code most of ValTools 7's session
    # actually touched) was ALSO never included here. Every "full test
    # suite" run this session before this fix only ever exercised
    # types/runtests.jl + the tests written directly below -- the JLab
    # module's own tests were being run manually, one file at a time, not
    # as part of `Pkg.test("ValTools")`. See project_gomed_validation_results
    # memory for the fuller story.
    include("jlab/runtests.jl")

    @testset "build_stretching" begin
        sc_r, Cs_r = build_stretching(32, 7.0, 0.0; vstretching=4)
        @test length(sc_r) == 32
        @test length(Cs_r) == 32
        @test sc_r[1] < sc_r[end]
        @test sc_r[1] ≈ -1.0 + 0.5/32
        @test sc_r[end] ≈ -1.0 + 31.5/32
        @test all(-1.0 .<= sc_r .<= 0.0)
        @test all(-1.0 .<= Cs_r .<= 0.0)

        sc1, Cs1 = build_stretching(10, 5.0, 2.0; vstretching=1)
        @test length(sc1) == 10
        @test length(Cs1) == 10
    end

    @testset "sigma_to_z" begin
        ny, nx, N = 3, 4, 5
        h = 500.0 .* ones(ny, nx)
        zeta = zeros(ny, nx)
        sc_r, Cs_r = build_stretching(N, 7.0, 0.0; vstretching=4)
        hc = 200.0

        z = sigma_to_z(h, zeta, sc_r, Cs_r, hc; vtransform=2)
        @test size(z) == (ny, nx, N)
        @test all(z .< 0)
        @test z[1,1,end] > z[1,1,1]  # surface > bottom

        zeta_3d = zeros(ny, nx, 2)
        z4 = sigma_to_z(h, zeta_3d, sc_r, Cs_r, hc; vtransform=2)
        @test size(z4) == (ny, nx, N, 2)
    end

    @testset "interp_z" begin
        ny, nx, N, nt = 2, 2, 10, 1
        z = Array{Float64}(undef, ny, nx, N, nt)
        var = Array{Float64}(undef, ny, nx, N, nt)
        for k in 1:N
            z[:,:,k,:] .= -100.0 + (k - 1) * 10.0  # -100 to -10
            var[:,:,k,:] .= Float64(k)
        end

        depths = [-50.0]
        out = interp_z(var, z, depths)
        @test size(out) == (ny, nx, 1, nt)
        @test all(isfinite.(out))
        # -50 is between z=-50 (k=6) and z=-40 (k=7), so val ≈ 6.0
        @test out[1,1,1,1] ≈ 6.0 atol=0.01

        # out of range
        out2 = interp_z(var, z, [-200.0])
        @test all(isnan.(out2))
    end

    @testset "compute_metrics" begin
        obs   = [1.0, 2.0, 3.0, 4.0, 5.0]
        model = [1.1, 2.2, 2.8, 4.1, 5.3]

        m = compute_metrics(obs, model)
        @test m["n"] == 5.0
        @test abs(m["bias"]) < 0.2
        @test m["rmse"] > 0
        @test m["correlation"] > 0.99
        @test m["nse"] > 0.9

        # perfect match
        m2 = compute_metrics(obs, obs)
        @test m2["bias"] == 0.0
        @test m2["rmse"] == 0.0
        @test m2["correlation"] ≈ 1.0
        @test m2["nse"] ≈ 1.0
    end

    @testset "taylor_stats" begin
        obs   = [1.0, 2.0, 3.0, 4.0, 5.0]
        model = [1.1, 2.2, 2.8, 4.1, 5.3]
        ts = taylor_stats(obs, model)
        @test haskey(ts, "std_ref")
        @test haskey(ts, "std_test")
        @test haskey(ts, "correlation")
        @test haskey(ts, "rms_diff")
        @test ts["correlation"] > 0.99
    end

    @testset "bootstrap_metrics" begin
        obs   = randn(100)
        model = obs .+ 0.1 .* randn(100)
        bm = bootstrap_metrics(obs, model; n_boot=100, seed=42)
        @test haskey(bm, "rmse")
        @test bm["rmse"].lo < bm["rmse"].mean < bm["rmse"].hi
    end

    @testset "current_ellipse_metrics" begin
        n = 500
        t = range(0, 2π * 10; length=n)
        obs_u = cos.(t) .+ 0.1 .* randn(n)
        obs_v = sin.(t) .+ 0.1 .* randn(n)
        mod_u = 1.05 .* cos.(t)
        mod_v = 1.05 .* sin.(t)

        cem = current_ellipse_metrics(obs_u, obs_v, mod_u, mod_v)
        @test haskey(cem, "obs_semi_major")
        @test haskey(cem, "speed_rmse")
        @test cem["obs_semi_major"] > 0
    end

    @testset "rotary_spectrum" begin
        n = 256
        t = collect(0.0:n-1)
        u = cos.(2π * 0.1 .* t)
        v = sin.(2π * 0.1 .* t)

        freqs, S_ccw, S_cw = rotary_spectrum(u, v; dt_hours=1.0)
        @test length(freqs) > 0
        @test length(S_ccw) == length(freqs)
        @test length(S_cw) == length(freqs)
        @test all(S_ccw .>= 0)
        @test all(S_cw .>= 0)

        # CCW signal at 0.1 cph → peak in S_ccw
        peak_idx = argmax(S_ccw)
        @test abs(freqs[peak_idx] - 0.1) < 0.02
    end

    @testset "rotary_spectrum — jackknife CI and rotary coefficient" begin
        n = 256
        t = collect(0.0:n-1)
        u = cos.(2π * 0.1 .* t)
        v = sin.(2π * 0.1 .* t)

        spec = rotary_spectrum(u, v; dt_hours=1.0)
        @test spec isa ValTools.RotarySpectralEstimate

        # Backward-compatible tuple destructuring
        f2, ccw2, cw2 = spec
        @test f2 == spec.freq
        @test ccw2 == spec.S_ccw
        @test cw2 == spec.S_cw

        # Jackknife CI brackets the point estimate
        @test spec.ci_ccw !== nothing
        @test spec.ci_cw !== nothing
        lo_ccw, hi_ccw = spec.ci_ccw
        @test all(lo_ccw .<= spec.S_ccw .<= hi_ccw)
        lo_cw, hi_cw = spec.ci_cw
        @test all(lo_cw .<= spec.S_cw .<= hi_cw)

        # Rotary coefficient in [-1, 1], strongly CCW at the signal frequency
        @test all(-1.0 .<= spec.rotary_coefficient .<= 1.0)
        peak_idx = argmax(spec.S_ccw)
        @test spec.rotary_coefficient[peak_idx] > 0.9

        # ci=false disables the jackknife pass
        spec_noci = rotary_spectrum(u, v; dt_hours=1.0, ci=false)
        @test spec_noci.ci_ccw === nothing
        @test spec_noci.ci_cw === nothing
    end

    @testset "rotary_spectrum — F-test significance" begin
        Random.seed!(13)
        n = 256
        t = collect(0.0:n-1)
        u = cos.(2π * 0.1 .* t) .+ 0.05 .* randn(n)
        v = sin.(2π * 0.1 .* t) .+ 0.05 .* randn(n)

        spec = rotary_spectrum(u, v; dt_hours=1.0)
        @test spec.ftest_ccw !== nothing
        @test spec.ftest_cw !== nothing
        @test all(0.0 .<= spec.ftest_ccw .<= 1.0)
        @test all(0.0 .<= spec.ftest_cw .<= 1.0)

        # F-test correctly flags the true CCW line frequency as significant
        peak_idx = argmin(spec.ftest_ccw)
        @test abs(spec.freq[peak_idx] - 0.1) < 0.02

        # ftest=false disables it
        spec_noftest = rotary_spectrum(u, v; dt_hours=1.0, ftest=false)
        @test spec_noftest.ftest_ccw === nothing
        @test spec_noftest.ftest_cw === nothing

        # Pure noise: F-test shouldn't flag most frequencies as significant
        un, vn = randn(n), randn(n)
        spec_noise = rotary_spectrum(un, vn; dt_hours=1.0)
        @test mean(spec_noise.ftest_ccw .< 0.05) < 0.25
        @test mean(spec_noise.ftest_cw .< 0.05) < 0.25
    end

    @testset "rotary_coherence" begin
        Random.seed!(7)
        n = 512
        t = collect(0.0:n-1)
        f0 = 0.08

        # Two CCW-rotating signals sharing a common signal + independent noise
        u1 = cos.(2π * f0 .* t) .+ 0.2 .* randn(n)
        v1 = sin.(2π * f0 .* t) .+ 0.2 .* randn(n)
        u2 = cos.(2π * f0 .* t) .+ 0.2 .* randn(n)
        v2 = sin.(2π * f0 .* t) .+ 0.2 .* randn(n)

        rc = rotary_coherence(u1, v1, u2, v2; dt_hours=1.0)
        @test rc isa ValTools.RotaryCoherenceEstimate
        @test all(0.0 .<= rc.coh_ccw .<= 1.0)
        @test all(0.0 .<= rc.coh_cw .<= 1.0)
        @test all(-π .<= rc.phase_ccw .<= π)
        @test 0.0 < rc.significance_level < 1.0

        peak_idx = argmin(abs.(rc.freq .- f0))
        @test rc.coh_ccw[peak_idx] > rc.significance_level
        @test rc.coh_ccw[peak_idx] > rc.coh_cw[peak_idx]

        # Independent white noise: coherence should mostly stay below significance
        u3, v3, u4, v4 = randn(n), randn(n), randn(n), randn(n)
        rc_noise = rotary_coherence(u3, v3, u4, v4; dt_hours=1.0)
        @test mean(rc_noise.coh_ccw .> rc_noise.significance_level) < 0.25
        @test mean(rc_noise.coh_cw .> rc_noise.significance_level) < 0.25
    end

    @testset "cross_coherence — real (non-rotary) signals" begin
        Random.seed!(11)
        n = 512
        t = collect(0.0:n-1)
        f0 = 0.08

        # Two real series sharing a common oscillation + independent noise
        x = cos.(2π * f0 .* t) .+ 0.2 .* randn(n)
        y = cos.(2π * f0 .* t) .+ 0.2 .* randn(n)

        cc = cross_coherence(x, y; dt=1.0)
        @test cc isa ValTools.CrossSpectralEstimate
        @test length(cc.freq) == length(cc.cross_power) == length(cc.coherence) == length(cc.phase)
        @test all(0.0 .<= cc.coherence .<= 1.0)
        @test all(-π .<= cc.phase .<= π)
        @test 0.0 < cc.significance_level < 1.0

        peak_idx = argmin(abs.(cc.freq .- f0))
        @test cc.coherence[peak_idx] > cc.significance_level
        # Same-phase cosines at f0 -> near-zero cross-spectral phase there
        @test abs(cc.phase[peak_idx]) < 0.3

        # Independent white noise: coherence should mostly stay below significance
        x2, y2 = randn(n), randn(n)
        cc_noise = cross_coherence(x2, y2; dt=1.0)
        @test mean(cc_noise.coherence .> cc_noise.significance_level) < 0.25

        # detrend="none" should error on garbage detrend string, matching rotary_coherence
        @test_throws ErrorException cross_coherence(x, y; detrend="bogus")

        # mismatched lengths
        @test_throws ErrorException cross_coherence(x, y[1:end-1])
    end

    @testset "ellipse_polarization — pure rectilinear (linear) signal" begin
        n = 256
        t = collect(0.0:n-1)

        # Line along the u-axis: v == 0 exactly -> Syy == Sxy == 0
        u = cos.(2π * 0.1 .* t)
        v = zeros(n)
        ep = ellipse_polarization(u, v; dt_hours=1.0)
        @test ep isa ValTools.EllipsePolarizationEstimate
        @test length(ep.freq) == length(ep.d1) == length(ep.d2) == length(ep.P)

        peak_idx = argmax(ep.d1)
        @test abs(ep.freq[peak_idx] - 0.1) < 0.02
        # Fully polarized (rank-1 spectral matrix): P ~= 1, not 0
        @test ep.P[peak_idx] > 0.99
        @test ep.alpha[peak_idx] > 0.99          # all power on the u-axis
        @test abs(ep.theta[peak_idx]) < 0.05      # orientation ~= 0 (u-axis)

        # Line at 45 deg: u and v identical -> orientation should recover pi/4
        u45 = cos.(2π * 0.1 .* t)
        v45 = cos.(2π * 0.1 .* t)
        ep45 = ellipse_polarization(u45, v45; dt_hours=1.0)
        peak45 = argmax(ep45.d1)
        @test ep45.P[peak45] > 0.99
        @test abs(ep45.alpha[peak45]) < 0.05      # equal power on u and v
        @test abs(ep45.theta[peak45] - π/4) < 0.05
    end

    @testset "ellipse_polarization — pure circular (rotary) signal, cross-checked against rotary_spectrum" begin
        n = 256
        t = collect(0.0:n-1)

        # CCW rotation
        u = cos.(2π * 0.1 .* t)
        v = sin.(2π * 0.1 .* t)
        ep = ellipse_polarization(u, v; dt_hours=1.0)
        rs = rotary_spectrum(u, v; dt_hours=1.0, ci=false, ftest=false)

        peak_idx = argmax(ep.d1)
        # Fully polarized here too -- circular is NOT the P~=0 case (isotropic noise is)
        @test ep.P[peak_idx] > 0.99
        @test abs(ep.alpha[peak_idx]) < 0.05      # no Cartesian (linear) anisotropy
        @test imag(ep.beta[peak_idx]) > 0.9       # strongly CCW

        # Independent derivation cross-check: imag(beta) from the spectral-matrix
        # eigendecomposition (this function) should match rotary_coefficient from
        # the direct CW/CCW power-ratio derivation (rotary_spectrum.jl) at the
        # same frequency, since both are the same physical quantity computed two
        # genuinely different ways.
        rs_peak = argmax(rs.S_ccw)
        @test abs(ep.freq[peak_idx] - rs.freq[rs_peak]) < 1e-8
        @test isapprox(imag(ep.beta[peak_idx]), rs.rotary_coefficient[rs_peak]; atol=0.02)

        # CW rotation flips the sign
        vcw = -sin.(2π * 0.1 .* t)
        epcw = ellipse_polarization(u, vcw; dt_hours=1.0)
        peak_cw = argmax(epcw.d1)
        @test imag(epcw.beta[peak_cw]) < -0.9
    end

    @testset "ellipse_polarization — isotropic noise gives low P" begin
        Random.seed!(21)
        n = 512
        u = randn(n)
        v = randn(n)
        ep = ellipse_polarization(u, v; dt_hours=1.0)
        # Independent, equal-variance, uncorrelated noise: d1 ~= d2 on average,
        # so P should be small (this is the genuine "unpolarized" case).
        @test mean(ep.P) < 0.5
        @test all(0.0 .<= ep.P .<= 1.0)
    end

    @testset "ellipse_polarization — jackknife CI and error handling" begin
        Random.seed!(23)
        n = 256
        t = collect(0.0:n-1)
        u = cos.(2π * 0.1 .* t) .+ 0.1 .* randn(n)
        v = sin.(2π * 0.1 .* t) .+ 0.1 .* randn(n)

        ep = ellipse_polarization(u, v; dt_hours=1.0)
        @test ep.ci_d1 !== nothing
        @test ep.ci_d2 !== nothing
        @test ep.ci_P !== nothing
        lo_d1, hi_d1 = ep.ci_d1
        @test all(lo_d1 .<= ep.d1 .<= hi_d1)
        lo_d2, hi_d2 = ep.ci_d2
        @test all(lo_d2 .<= ep.d2 .<= hi_d2)
        lo_P, hi_P = ep.ci_P
        @test all(lo_P .<= ep.P .<= hi_P .+ 1e-8)
        @test all(0.0 .<= lo_P) && all(hi_P .<= 1.0)

        @test ep.ci_theta !== nothing
        lo_th, hi_th = ep.ci_theta
        @test all(lo_th .<= ep.theta .<= hi_th)

        ep_noci = ellipse_polarization(u, v; dt_hours=1.0, ci=false)
        @test ep_noci.ci_d1 === nothing
        @test ep_noci.ci_d2 === nothing
        @test ep_noci.ci_P === nothing
        @test ep_noci.ci_theta === nothing

        @test_throws ErrorException ellipse_polarization(u, v[1:end-1])
    end

    @testset "ellipse_polarization — circular jackknife CI handles the theta wraparound" begin
        # Direct, deterministic test of the circular-CI primitive itself:
        # hand-crafted delete-one theta estimates that are a TIGHT physical
        # cluster around the true value pi/2 (2*theta clusters around the
        # +-pi branch cut), but which look maximally spread out to plain
        # subtraction. A plain linear jackknife on these gives mean~=0
        # (wrong by ~pi/2) and a ~13 rad width (4x a full circle) --
        # meaningless. The circular jackknife must recover the true center
        # exactly and a sane, bounded width.
        # CI is centered on the point estimate (passed in explicitly), not
        # the delete-one replicates' own circular mean -- guarantees
        # bracketing by construction; the replicates only supply the width.
        ext = Base.get_extension(ValTools, :ValToolsMultitaperExt)
        theta_point = [1.55]
        theta_del = reshape([1.50, 1.53, 1.56, -1.56, -1.53, -1.50], 1, 6)
        lo, hi = ext._jackknife_ci_circular_half_angle(theta_point, theta_del, 0.95)
        @test isapprox((lo[1] + hi[1]) / 2, 1.55; atol=1e-8)
        @test lo[1] <= theta_point[1] <= hi[1]
        @test (hi[1] - lo[1]) < 1.0  # sane -- a broken linear jackknife gives ~13.4 here

        # Real end-to-end signal, true orientation placed EXACTLY at the
        # theta=+-pi/2 boundary (a pure v-axis line: u is noise-only, v
        # carries the signal), so per-taper delete-one estimates genuinely
        # straddle the branch cut -- confirmed theta flips sign between
        # adjacent frequency bins in this exact configuration.
        Random.seed!(7)
        n = 256
        t = collect(0.0:n-1)
        a_amp = cos.(2π * 0.1 .* t)
        u_w = 0.3 .* randn(n)
        v_w = a_amp .+ 0.3 .* randn(n)
        ep_w = ellipse_polarization(u_w, v_w; dt_hours=1.0)
        peak = argmax(ep_w.d1)
        @test abs(abs(ep_w.theta[peak]) - pi/2) < 0.05  # true orientation recovered near the boundary
        lo_w, hi_w = ep_w.ci_theta
        @test all(isfinite, lo_w) && all(isfinite, hi_w)
        @test all(lo_w .<= ep_w.theta .<= hi_w)  # CI must bracket the point estimate everywhere, always true by construction
        # Sanity/tightness only asserted NEAR THE SIGNAL PEAK -- far from it
        # (pure-noise frequency bins) orientation is genuinely undetermined,
        # and a wide CI there is the correct answer, not a bug.
        near_peak = max(1, peak - 3):min(length(ep_w.freq), peak + 3)
        @test all((hi_w[near_peak] .- lo_w[near_peak]) .< 1.0)
    end

    @testset "alongtrack_wavenumber_spectrum" begin
        n = 512
        dx = 2.0  # km
        x = range(0, step=dx, length=n)
        signal = sin.(2π .* x ./ 50.0)  # 50 km wavelength → k = 0.02 cpk

        k, psd = alongtrack_wavenumber_spectrum(signal, dx)
        @test length(k) > 0
        @test all(k .> 0)
        @test all(psd .>= 0)

        signal_nan = copy(signal)
        signal_nan[100:103] .= NaN
        @test_throws ErrorException alongtrack_wavenumber_spectrum(signal_nan, dx)
        k2, psd2 = alongtrack_wavenumber_spectrum(signal_nan, dx; allow_nan=true)
        @test length(k2) > 0
        @test all(psd2 .>= 0)
        # interpolating over 4 samples out of 512 shouldn't materially move
        # the recovered peak wavenumber
        @test isapprox(k[argmax(psd)], k2[argmax(psd2)]; rtol=0.1)
    end

    @testset "isotropic_2d_spectrum" begin
        ny, nx = 64, 64
        dx, dy = 5.0, 5.0
        field = randn(ny, nx)

        k_iso, psd_iso = isotropic_2d_spectrum(field, dx, dy)
        @test length(k_iso) > 0
        @test all(psd_iso[isfinite.(psd_iso)] .>= 0)

        field_nan = copy(field)
        field_nan[1:5, 1:5] .= NaN
        @test_throws ErrorException isotropic_2d_spectrum(field_nan, dx, dy)
        k_iso2, psd_iso2 = isotropic_2d_spectrum(field_nan, dx, dy; allow_nan=true)
        @test length(k_iso2) > 0
        @test all(psd_iso2[isfinite.(psd_iso2)] .>= 0)

        all_nan = fill(NaN, ny, nx)
        @test_throws ErrorException isotropic_2d_spectrum(all_nan, dx, dy; allow_nan=true)
    end

    @testset "resample_uniform" begin
        # Irregular sampling with a gap (NaN), reconstructing a known sinusoid
        rng = MersenneTwister(6)
        t_true = sort(100 .* rand(rng, 400))
        x_true = sin.(2π .* t_true ./ 10.0)
        t_irregular = copy(t_true)
        x_gappy = copy(x_true)
        x_gappy[50:55] .= NaN

        t_u, x_u = resample_uniform(t_irregular, x_gappy; dt=0.5)
        @test issorted(t_u)
        @test all(isfinite, x_u)
        @test isapprox(t_u[1], minimum(t_irregular[isfinite.(x_gappy)]); atol=1.0)

        # Reconstructed signal should still look like the same sinusoid at
        # every point (elementwise max deviation, not isapprox's default
        # whole-array 2-norm — with ~200 points a scattering of small
        # per-point errors sums in the norm to something that looks large
        # even though no individual point is off by much)
        x_expected = sin.(2π .* t_u ./ 10.0)
        @test maximum(abs.(x_u .- x_expected)) < 0.15

        @test_throws ErrorException resample_uniform([1.0, 2.0], [1.0])
        @test_throws ErrorException resample_uniform([2.0, 1.0], [1.0, 2.0])
        @test_throws ErrorException resample_uniform([NaN, NaN, 1.0], [NaN, NaN, 1.0])
    end

    @testset "reference_slope! — draws without error and returns the axis" begin
        fig = Figure()
        ax = Axis(fig[1, 1]; xscale=log10, yscale=log10)
        k, psd = alongtrack_wavenumber_spectrum(sin.(2π .* (0:511) ./ 50.0), 2.0)
        lines!(ax, k, psd)
        ax2 = reference_slope!(ax, k[1], psd[1], -3.0)
        @test ax2 === ax
    end

    @testset "msvd — SVD-based polarization analysis" begin
        # Shape consistency (jLab's own msvd_test), both the 3-D (J,N,K)
        # and single-band 2-D (N,K) input forms.
        J, N, K = 6, 3, 5
        W = randn(ComplexF64, J, N, K)
        r = msvd(W)
        @test size(r.d) == (J, min(N, K))
        @test size(r.u1) == (J, N) && size(r.u2) == (J, N)
        @test size(r.v1) == (J, K) && size(r.v2) == (J, K)
        @test length(r.trS) == J
        # Singular values are non-negative and sorted descending per band.
        @test all(r.d .>= 0)
        @test all(r.d[:, 1] .>= r.d[:, 2] .>= r.d[:, 3])

        r2 = msvd(W[1, :, :])
        @test r2.d ≈ r.d[1, :]
        @test r2.trS ≈ r.trS[1]

        # Cross-checked directly against real jLab MATLAB msvd.m on this
        # exact deterministic (J=4,N=3,K=5) input (2026-08-02): d and trS
        # matched to floating-point precision. jLab's own msvd.m has a
        # confusing internal variable-name collision (its `[N,J,M,K]=size(...)`
        # reassigns `M` to mean the CHANNEL count on the reshaped 4-D array,
        # not an outer batch dimension as the docstring's `M x J x N x K`
        # form might suggest) -- the true normalization is 1/sqrt(K*N_channels),
        # not 1/sqrt(K) alone; verified by the exact match below.
        Jc, Nc, Kc = 4, 3, 5
        Wc = Array{ComplexF64}(undef, Jc, Nc, Kc)
        idx = 1
        for j in 1:Jc, n in 1:Nc, k in 1:Kc
            Wc[j, n, k] = complex(sin(idx * 0.37) * 2.1, cos(idx * 0.53) * 1.7)
            idx += 1
        end
        rc = msvd(Wc)
        @test rc.d[1, :] ≈ [1.6443686292, 1.0433765420, 0.0241972961] atol=1e-9
        @test rc.trS[1] ≈ 3.7931683063 atol=1e-9
        @test rc.d[4, :] ≈ [1.8206930783, 0.7218753882, 0.0669310759] atol=1e-9
        @test rc.trS[4] ≈ 3.8405071302 atol=1e-9

        # SVD reconstruction identity: d/u1/v1/u2/v2 (the top 2 of min(N,K)=3
        # triplets) plus the implicit 3rd triplet must exactly reconstruct
        # the (K*N-rescaled) input at every band -- a MATLAB-independent
        # correctness check on the full decomposition, not just d/trS.
        for j in 1:J
            Wj_scaled = W[j, :, :] ./ sqrt(K * N)
            F = svd(Wj_scaled)
            recon = F.U * Diagonal(F.S) * F.Vt
            @test recon ≈ Wj_scaled atol=1e-10
            # And msvd's own u1/v1/d agree with this independently-computed F.
            @test r.d[j, :] ≈ F.S
            @test r.u1[j, :] ≈ F.U[:, 1]
            @test r.v1[j, :] ≈ F.V[:, 1]
        end

        # Pure rank-1 (perfectly coherent/polarized) signal: every
        # eigentransform k is the same channel vector p times a different
        # complex scalar amplitude -- physically, a single fully-polarized
        # source with no noise. Only one nonzero singular value should
        # result, and u1 should recover p's direction (up to the inherent
        # SVD phase ambiguity, checked via the phase-invariant |.| and the
        # reconstruction identity, not raw equality).
        p = normalize([1.0 + 0.5im, -0.3 + 1.2im, 0.8 - 0.1im])
        amps = [complex(cos(0.3k), sin(0.3k)) for k in 1:K]
        Wrank1 = reshape(p, 1, N, 1) .* reshape(amps, 1, 1, K)
        rrank1 = msvd(Wrank1)
        @test rrank1.d[1, 2] < 1e-10 * rrank1.d[1, 1]   # second singular value ~0
        @test isapprox(abs.(rrank1.u1[1, :]), abs.(p); atol=1e-8)

        @test_throws MethodError msvd(randn(3, 4))   # real input not accepted
    end

    @testset "cross_spectrum_kx_ky" begin
        ny, nx = 32, 32
        field1 = randn(ny, nx)
        field2 = field1 .+ 0.1 .* randn(ny, nx)

        k, coh, ph, cpsd = cross_spectrum_kx_ky(field1, field2, 5.0, 5.0)
        @test length(k) > 0
        @test all(coh[isfinite.(coh)] .>= 0)
        @test all(coh[isfinite.(coh)] .<= 1.0 + 1e-10)
    end

    @testset "detrend_2d_linear" begin
        ny, nx = 10, 10
        x_grid = repeat((0:nx-1)', ny, 1)
        y_grid = repeat(0:ny-1, 1, nx)
        plane = 2.0 .* x_grid .+ 3.0 .* y_grid .+ 5.0
        result = detrend_2d_linear(Float64.(plane))
        @test maximum(abs.(result)) < 1e-10
    end

    @testset "helmholtz_decomposition — energy conservation (exact identity)" begin
        # KErot(k) + KEdiv(k) ≡ 0.5*(Cu(k)+Cv(k)) is an exact identity of the
        # closed-form BCF14 integral (derivable directly from eqs. 2.27/2.30-2.31,
        # independent of what Cu, Cv actually are) — the strongest regression
        # test available since it doesn't depend on any external published
        # number, only on this function's own internal arithmetic being right.
        # Clean (noise-free) power laws: this test's purpose is to check the
        # identity holds to trapezoidal-discretization precision, not to
        # check robustness to noisy input — injecting noise here just adds
        # an unrelated, unbounded source of per-point scatter (noise
        # differentiated via the central-difference step) on top of the
        # thing actually being tested.
        n = 200
        k = 10 .^ range(-3, 0; length=n)
        Cu = 1.0 .* k .^ (-2.2)
        Cv = 0.7 .* k .^ (-1.8)
        hs = helmholtz_decomposition(k, Cu, Cv)

        @test hs.k == k
        interior = 10:(n - 10)   # exclude edges: backward-integration BC forces
                                  # Dpsi=Dphi=0 at k[end], and cumulative trapezoidal
                                  # error is largest near both ends
        lhs = hs.KErot[interior] .+ hs.KEdiv[interior]
        rhs = 0.5 .* (Cu[interior] .+ Cv[interior])
        @test maximum(abs.(lhs .- rhs) ./ abs.(rhs)) < 0.02

        @test_throws ErrorException helmholtz_decomposition([1.0, 2.0], [1.0], [1.0, 2.0])
        @test_throws ErrorException helmholtz_decomposition([2.0, 1.0, 3.0], ones(3), ones(3))
        @test_throws ErrorException helmholtz_decomposition([-1.0, 1.0, 2.0], ones(3), ones(3))
    end

    @testset "helmholtz_decomposition — pure-rotational and pure-divergent analytic cases" begin
        # Derived independently from the closed-form D-functions (not just
        # copied from BCF14): requiring Dphi ≡ 0 for ALL s forces, via its own
        # derivative, Dpsi(s) = Cu(s) and Cv(s) = n*Cu(s) when Cu = k^-n — which
        # matches BCF14 eq. 2.11 for a purely non-divergent flow. The mirrored
        # case (Cv = k^-n, Cu = n*Cv) analogously forces Dpsi ≡ 0.
        n = 400
        k = 10 .^ range(-4, 0; length=n)
        slope = 2.5
        interior = 30:(n - 30)

        # Pure rotational: Cu ~ k^-n, Cv = n*Cu ⟹ Dphi ≈ 0
        Cu_rot = k .^ (-slope)
        Cv_rot = slope .* Cu_rot
        hs_rot = helmholtz_decomposition(k, Cu_rot, Cv_rot)
        @test all(abs.(hs_rot.Dphi[interior]) .< 1e-3 .* maximum(abs.(hs_rot.Dpsi[interior])))

        # Pure divergent (mirrored): Cv ~ k^-n, Cu = n*Cv ⟹ Dpsi ≈ 0
        Cv_div = k .^ (-slope)
        Cu_div = slope .* Cv_div
        hs_div = helmholtz_decomposition(k, Cu_div, Cv_div)
        @test all(abs.(hs_div.Dpsi[interior]) .< 1e-3 .* maximum(abs.(hs_div.Dphi[interior])))
    end

    @testset "wave_vortex_decomposition" begin
        n = 100
        k = 10 .^ range(-2, 0; length=n)
        rng = MersenneTwister(3)
        Cu = k .^ (-2.0) .* (1 .+ 0.05 .* randn(rng, n))
        Cv = 0.8 .* k .^ (-1.9) .* (1 .+ 0.05 .* randn(rng, n))
        hs = helmholtz_decomposition(k, Cu, Cv)
        wv = wave_vortex_decomposition(hs)

        @test wv.k == hs.k
        @test isapprox(wv.Ewave, 2.0 .* hs.KEdiv)
        @test isapprox(wv.Evortex, 2.0 .* hs.KErot)
        # Etotal = Ewave + Evortex should reduce to Cu+Cv exactly (same
        # identity as the energy-conservation test above, just rearranged)
        interior = 15:(n - 15)
        @test isapprox((wv.Ewave .+ wv.Evortex)[interior], (Cu .+ Cv)[interior]; rtol=0.03)
    end

    @testset "velocity_structure_functions" begin
        rng = MersenneTwister(4)
        n = 60
        x = 100 .* rand(rng, n)
        y = 100 .* rand(rng, n)

        # Pure solid-body rotation: u = -Ω*y, v = Ω*x (rotational, zero-divergence
        # flow) ⟹ velocity differences are purely transverse to the separation
        # vector, so Dll should be ≈ 0 everywhere pairs exist.
        Ω = 0.05
        u = -Ω .* y
        v = Ω .* x
        rbins = collect(0.0:10.0:80.0)
        sf = velocity_structure_functions(x, y, u, v; rbins=rbins)

        @test length(sf.r) == length(rbins) - 1
        valid = sf.npairs .> 0
        @test any(valid)
        @test all(isapprox.(sf.Dll[valid], 0.0; atol=1e-8))
        @test all(sf.Dtt[valid] .> 0)   # rigid rotation: Dtt = (Ω r)^2 > 0 for r>0

        @test_throws ErrorException velocity_structure_functions([1.0], [1.0, 2.0], [1.0], [1.0, 2.0]; rbins=rbins)
    end

    @testset "helmholtz_structure_function" begin
        # exact-by-construction algebraic identity (documented as not an
        # independent correctness check, but confirms no arithmetic slip)
        r = collect(1.0:1.0:40.0)
        rng = MersenneTwister(5)
        Dll_id = r .^ 1.3 .* (1 .+ 0.02 .* rand(rng, length(r)))
        Dtt_id = 0.6 .* r .^ 1.1
        hsf_id = helmholtz_structure_function(r, Dll_id, Dtt_id)
        @test isapprox(hsf_id.Drr .+ hsf_id.Ddd, Dll_id .+ Dtt_id)

        # Exponent-dependent analytic cases, derived directly from this
        # function's own closed-form integral (the exact real-space analog
        # of how BCF14's Cv=n*Cu wavenumber relation was derived and
        # verified for helmholtz_decomposition above): for Dll(r) = r^p and
        # Dtt(r) = c*Dll(r), requiring Ddd ≡ 0 for ALL r algebraically
        # forces c = 1+p (NOT the naive guess c=1 — an earlier version of
        # this test used solid-body rotation / uniform strain, where Dll or
        # Dtt is identically zero rather than sharing the same power-law
        # shape; that's a boundary case of this family, not a
        # counterexample). The mirrored case Drr ≡ 0 requires c = 1/(1+p).
        #
        # p=2 is used deliberately, not arbitrarily: the code's near-origin
        # treatment assumes the integrand (Dtt-Dll)/τ → 0 as τ→0, which only
        # holds for p>1 (a real-space requirement discovered while writing
        # this test — an initial p=0.7 attempt failed because τ^(p-1)
        # diverges at the origin for p<1, outside the function's documented
        # domain). p=2 is also not just numerically convenient: it matches
        # this project's own target regime (Callies et al. 2015 "summer
        # submesoscale" k^-3 KE slope maps to structure-function exponent
        # p=n-1=2), and trapezoidal integration is exact for the resulting
        # linear integrand, so this case is exact, not just approximate.
        p = 2.0
        rr = collect(1.0:1.0:60.0)
        Dll_p = rr .^ p

        Dtt_rot = (1 + p) .* Dll_p
        h_rot = helmholtz_structure_function(rr, Dll_p, Dtt_rot)
        @test all(isapprox.(h_rot.Ddd, 0.0; atol=1e-8))
        @test isapprox(h_rot.Drr, Dll_p .+ Dtt_rot; rtol=1e-8)

        Dtt_div = Dll_p ./ (1 + p)
        h_div = helmholtz_structure_function(rr, Dll_p, Dtt_div)
        @test all(isapprox.(h_div.Drr, 0.0; atol=1e-8))
        @test isapprox(h_div.Ddd, Dll_p .+ Dtt_div; rtol=1e-8)

        @test_throws ErrorException helmholtz_structure_function([2.0, 1.0], [1.0, 1.0], [1.0, 1.0])
    end

    # ══════════════════════════════════════════════════════════════
    # Colocation tests
    # ══════════════════════════════════════════════════════════════

    @testset "colocate_model_obs (regular grid)" begin
        nx, ny, nt = 20, 15, 3
        lon = range(-95.0, -90.0; length=nx) |> collect
        lat = range(20.0, 25.0; length=ny) |> collect
        field = Array{Float64}(undef, ny, nx, nt)
        for it in 1:nt, j in 1:nx, i in 1:ny
            field[i, j, it] = lon[j] + lat[i]
        end
        times = [DateTime(2025, 1, d) for d in 1:nt]

        obs = DataFrame(
            lon = [-93.0, -92.0, -91.0],
            lat = [22.0, 23.0, 24.0],
            time = [DateTime(2025, 1, 1, 6), DateTime(2025, 1, 2, 3), DateTime(2025, 1, 3)]
        )

        result = colocate_model_obs(field, lon, lat, times, obs;
                                    out_col=:model_val)
        @test "model_val" in names(result)
        @test all(isfinite.(result.model_val))
        @test result.model_val[1] ≈ -93.0 + 22.0 atol=0.5
    end

    @testset "colocate_model_obs (curvilinear)" begin
        ny, nx, nt = 10, 10, 1
        lon2d = [Float64(-95 + j * 0.5) for i in 1:ny, j in 1:nx]
        lat2d = [Float64(20 + i * 0.5) for i in 1:ny, j in 1:nx]
        field = lon2d .+ lat2d
        field3d = reshape(field, ny, nx, 1)
        times = [DateTime(2025, 1, 1)]

        # Pick a point inside the grid domain
        obs = DataFrame(
            lon = [-92.5],
            lat = [23.0],
            time = [DateTime(2025, 1, 1)]
        )
        result = colocate_model_obs(field3d, lon2d, lat2d, times, obs;
                                    out_col=:model_val, max_dist_km=200.0)
        @test isfinite(result.model_val[1])
        @test result.model_val[1] ≈ -92.5 + 23.0 atol=1.0
    end

    @testset "colocate_model_grid" begin
        ny, nx = 20, 20
        lon = range(-95.0, -90.0; length=nx) |> collect
        lat = range(20.0, 25.0; length=ny) |> collect
        field = [lo + la for la in lat, lo in lon]

        tlon = range(-94.0, -91.0; length=5) |> collect
        tlat = range(21.0, 24.0; length=5) |> collect

        result = colocate_model_grid(field, lon, lat, tlon, tlat)
        @test size(result.data) == (5, 5)
        @test result.data[1, 1] ≈ tlon[1] + tlat[1] atol=0.5
    end

    @testset "colocate_model_mooring" begin
        nx, ny, nt = 20, 15, 5
        lon = range(-95.0, -90.0; length=nx) |> collect
        lat = range(20.0, 25.0; length=ny) |> collect
        field = Array{Float64}(undef, ny, nx, nt)
        for it in 1:nt
            field[:, :, it] .= Float64(it)
        end
        times = [DateTime(2025, 1, d) for d in 1:nt]

        result = colocate_model_mooring(field, lon, lat, times, -92.5, 22.5)
        @test length(result.data) == nt
        for it in 1:nt
            @test result.data[it] ≈ Float64(it) atol=0.01
        end
    end

    @testset "colocate_model_section" begin
        nx, ny, nt = 20, 15, 3
        lon = range(-95.0, -90.0; length=nx) |> collect
        lat = range(20.0, 25.0; length=ny) |> collect
        u_field = ones(ny, nx, nt)
        v_field = 2.0 .* ones(ny, nx, nt)
        times = [DateTime(2025, 1, d) for d in 1:nt]

        lon_sec = [-93.0, -92.0, -91.0]
        lat_sec = [22.5, 22.5, 22.5]

        result = colocate_model_section(u_field, v_field, lon, lat, times,
                                        lon_sec, lat_sec; angle_deg=0.0)
        @test size(result.u) == (nt, 3)
        @test size(result.v_along) == (nt, 3)
        # angle=0 → v_along = v = 2.0
        @test all(result.v_along .≈ 2.0)
    end

    @testset "pressure_to_depth" begin
        d = pressure_to_depth(100.0)
        @test d ≈ 100.0 * 1.019716 atol=0.01

        d_lat = pressure_to_depth(1000.0; latitude=45.0)
        @test 980 < d_lat < 1020

        p_back = depth_to_pressure(d_lat; latitude=45.0)
        @test p_back ≈ 1000.0 atol=0.1
    end

    @testset "MooringKnockdownModel" begin
        m = MooringKnockdownModel(k=0.1, line_length=500.0)
        @test delta_z(m, 0.0) ≈ 0.0
        @test delta_z(m, 1.0) > 0.0
        @test horizontal_excursion(m, 1.0) > 0.0

        # Zero knockdown
        m0 = MooringKnockdownModel()
        @test delta_z(m0, 1.0) ≈ 0.0
    end

    @testset "fit_knockdown" begin
        k_true = 0.05
        L_true = 400.0
        speeds = range(0.0, 1.5; length=100) |> collect
        dz_true = L_true .* (1.0 .- cos.(atan.(k_true .* speeds .^ 2)))
        depth_actual = 500.0 .+ dz_true .+ 0.5 .* randn(100)

        m = fit_knockdown(speeds, depth_actual, 500.0; line_length=L_true)
        @test m.k > 0
        @test m.line_length ≈ L_true
        pred = delta_z(m, speeds)
        rmse = sqrt(mean((pred .- dz_true) .^ 2))
        @test rmse < 5.0
    end

    @testset "project_to_fixed_depths" begin
        n_time = 50
        n_instr = 5
        time = [DateTime(2025, 1, 1) + Hour(i) for i in 1:n_time]
        depths = hcat([fill(Float64(d), n_time) for d in [50, 100, 200, 500, 1000]]...)
        T = hcat([fill(25.0 - d/100, n_time) for d in [50, 100, 200, 500, 1000]]...)

        result = project_to_fixed_depths(time, depths,
                                         Dict(:temperature => T),
                                         [75.0, 150.0, 350.0])
        @test size(result.temperature) == (n_time, 3)
        @test result.temperature[1, 1] ≈ 25.0 - 75.0/100 atol=0.1
        @test all(result.n_instruments .== 5)
    end

    # ══════════════════════════════════════════════════════════════
    # Observations tests (with synthetic NetCDF files)
    # ══════════════════════════════════════════════════════════════

    @testset "sound_speed_chen_millero" begin
        c = sound_speed_chen_millero(10.0, 35.0, 0.0)
        @test 1480 < c < 1510
        c_arr = sound_speed_chen_millero([10.0, 20.0], [35.0, 35.0], [0.0, 0.0])
        @test length(c_arr) == 2
        @test c_arr[2] > c_arr[1]  # warmer = faster
    end

    @testset "GHRSSTLoader (synthetic)" begin
        tmpdir = mktempdir()
        ncpath = joinpath(tmpdir, "ghrsst_test.nc")
        ds = NCDataset(ncpath, "c")
        defDim(ds, "lon", 10)
        defDim(ds, "lat", 8)
        defDim(ds, "time", 2)
        lon_var = defVar(ds, "lon", Float64, ("lon",))
        lat_var = defVar(ds, "lat", Float64, ("lat",))
        time_var = defVar(ds, "time", Float64, ("time",))
        sst_var = defVar(ds, "analysed_sst", Float64, ("lat", "lon", "time"))
        lon_var[:] = collect(range(-95.0, -90.0; length=10))
        lat_var[:] = collect(range(20.0, 25.0; length=8))
        time_var[:] = [0.0, 1.0]
        sst_data = fill(300.0, 8, 10, 2)  # Kelvin
        sst_var[:,:,:] = sst_data
        close(ds)

        r = GHRSSTLoader(ncpath)
        result = ghrsst_sst(r; celsius=true)
        @test result !== nothing
        @test length(r.lon) == 10
        @test length(r.lat) == 8
        @test all(result.data .≈ 300.0 - 273.15)
        close(r)
        rm(tmpdir; recursive=true)
    end

    @testset "DUACSLoader (synthetic)" begin
        tmpdir = mktempdir()
        ncpath = joinpath(tmpdir, "duacs_test.nc")
        ds = NCDataset(ncpath, "c")
        defDim(ds, "longitude", 5)
        defDim(ds, "latitude", 4)
        defDim(ds, "time", 3)
        defVar(ds, "longitude", collect(range(-95.0, -90.0; length=5)), ("longitude",))
        defVar(ds, "latitude", collect(range(20.0, 24.0; length=4)), ("latitude",))
        defVar(ds, "time", [0.0, 1.0, 2.0], ("time",))
        defVar(ds, "adt", fill(0.5, 4, 5, 3), ("latitude", "longitude", "time"))
        defVar(ds, "sla", fill(0.1, 4, 5, 3), ("latitude", "longitude", "time"))
        defVar(ds, "ugos", fill(0.05, 4, 5, 3), ("latitude", "longitude", "time"))
        defVar(ds, "vgos", fill(-0.02, 4, 5, 3), ("latitude", "longitude", "time"))
        close(ds)

        r = DUACSLoader(ncpath)
        ssh_r = duacs_ssh(r)
        @test ssh_r.data !== nothing
        @test all(ssh_r.data .≈ 0.5)
        sla_r = duacs_sla(r)
        @test all(sla_r.data .≈ 0.1)
        vel_r = duacs_geostrophic_velocity(r)
        @test all(vel_r.u .≈ 0.05)
        close(r)
        rm(tmpdir; recursive=true)
    end

    @testset "ThermistorLoader (synthetic)" begin
        tmpdir = mktempdir()
        ncpath = joinpath(tmpdir, "therm_test.nc")
        ds = NCDataset(ncpath, "c")
        nt = 10
        nz = 5
        defDim(ds, "time", nt)
        defDim(ds, "depth", nz)
        depths = [50.0, 100.0, 200.0, 500.0, 1000.0]
        times = [DateTime(2025,1,1) + Hour(i) for i in 1:nt]
        defVar(ds, "time", times, ("time",))
        defVar(ds, "depth", depths, ("depth",))
        T = [25.0 - d/100 + 0.1*sin(2π*t/10) for t in 1:nt, d in depths]
        defVar(ds, "temperature", T, ("time", "depth"))
        close(ds)

        r = ThermistorLoader(ncpath; site="test")
        result = thermistor_temperature(r)
        @test result.data !== nothing
        @test size(result.data) == (nt, nz)

        iso = thermistor_isotherm_depth(r, 20.0)
        @test iso !== nothing
        @test length(iso.data) == nt

        hc = thermistor_heat_content(r; z1=50.0, z2=500.0)
        @test hc !== nothing
        @test all(isfinite.(hc.data))
        close(r)
        rm(tmpdir; recursive=true)
    end

    @testset "MooringCurrentLoader (synthetic)" begin
        tmpdir = mktempdir()
        ncpath = joinpath(tmpdir, "mooring_test.nc")
        ds = NCDataset(ncpath, "c")
        nt = 100
        nz = 3
        defDim(ds, "time", nt)
        defDim(ds, "depth", nz)
        times = [DateTime(2025,1,1) + Hour(i) for i in 1:nt]
        defVar(ds, "time", times, ("time",))
        defVar(ds, "depth", [50.0, 100.0, 200.0], ("depth",))
        t_rad = range(0, 2π*5; length=nt)
        u_data = [0.3*cos(t) for t in t_rad, _ in 1:nz]
        v_data = [0.3*sin(t) for t in t_rad, _ in 1:nz]
        defVar(ds, "u", u_data, ("time", "depth"))
        defVar(ds, "v", v_data, ("time", "depth"))
        close(ds)

        r = MooringCurrentLoader(ncpath; site="test_mooring", lon=-92.0, lat=22.0)
        p = mooring_current_profiles(r)
        @test p.u !== nothing
        @test size(p.u) == (nt, nz)

        spd = mooring_speed(r)
        @test spd !== nothing
        @test all(spd.data .> 0)

        dir = mooring_direction(r)
        @test dir !== nothing

        ell = mooring_variance_ellipse(r)
        @test ell !== nothing
        @test ell["semi_major"] > 0

        pvd = mooring_progressive_vector(r)
        @test nrow(pvd) == nt
        close(r)
        rm(tmpdir; recursive=true)
    end

    @testset "IESLoader (synthetic)" begin
        tmpdir = mktempdir()
        ncpath = joinpath(tmpdir, "ies_test.nc")
        ds = NCDataset(ncpath, "c")
        nt = 50
        defDim(ds, "time", nt)
        times = [DateTime(2025,1,1) + Hour(i) for i in 1:nt]
        defVar(ds, "time", times, ("time",))
        tau = 1.5 .+ 0.001 .* sin.(2π .* (1:nt) ./ 25)
        defVar(ds, "tau", tau, ("time",))
        close(ds)

        r = IESLoader(ncpath; site="test_ies", depth=1000.0)
        tt = ies_travel_time(r)
        @test tt.data !== nothing
        @test length(tt.data) == nt

        ssh = ies_ssh_anomaly(r)
        @test ssh !== nothing
        @test length(ssh.data) == nt

        df = ies_to_dataframe(r)
        @test nrow(df) == nt
        close(r)
        rm(tmpdir; recursive=true)
    end

    @testset "GEMBuilder (synthetic)" begin
        g = GEMBuilder(; depth_grid=collect(0.0:50.0:500.0), site="test")
        @test length(g.depth_grid) == 11

        c = sound_speed_chen_millero(15.0, 35.0, 100.0)
        @test 1490 < c < 1520
    end

    # ══════════════════════════════════════════════════════════════
    # Plot extension tests (CairoMakie)
    # ══════════════════════════════════════════════════════════════

    @testset "taylor_diagram" begin
        fig = taylor_diagram(1.0; samples=[(std=0.9, corr=0.95, label="M1"),
                                            (std=1.1, corr=0.88, label="M2")])
        @test fig isa Figure
    end

    @testset "plot_comparison_map" begin
        obs = randn(8, 10)
        mod = obs .+ 0.1 .* randn(8, 10)
        lon = collect(range(-95, -90; length=10))
        lat = collect(range(20, 25; length=8))
        fig = plot_comparison_map(obs, mod, lon, lat; title="SSH test", units="m")
        @test fig isa Figure
    end

    @testset "plot_timeseries_comparison" begin
        obs_d = Dict("Obs" => (collect(1.0:20), randn(20)))
        mod_d = Dict("Model" => (collect(1.0:20), randn(20)))
        fig = plot_timeseries_comparison(obs_d, mod_d; variable="SSH [m]")
        @test fig isa Figure
    end

    @testset "lic_texture and plot_lic" begin
        u = [cos(0.1*i + 0.1*j) for i in 1:20, j in 1:20]
        v = [sin(0.1*i + 0.1*j) for i in 1:20, j in 1:20]
        tex = lic_texture(u, v; length=5)
        @test size(tex) == (20, 20)
        @test all(x -> isfinite(x) && 0 <= x <= 1, tex)

        fig = plot_lic(u, v; length=5, title="Flow")
        @test fig isa Figure
    end

    @testset "plot_streamlines" begin
        u = Float64.([sin(π * i/20) * cos(π * j/25) for i in 1:40, j in 1:50])
        v = Float64.([-cos(π * i/20) * sin(π * j/25) for i in 1:40, j in 1:50])
        fig = plot_streamlines(u, v; density=1.0, title="Streamlines test")
        @test fig isa Figure

        lon = collect(range(-95, -90; length=50))
        lat = collect(range(20, 25; length=40))
        speed = hypot.(u, v)
        fig2 = plot_streamlines(u, v; lon=lon, lat=lat, field=speed,
                                 field_label="Speed", title="With background")
        @test fig2 isa Figure
    end

    @testset "plot_flow" begin
        u = Float64.([sin(π * i/20) * cos(π * j/25) for i in 1:40, j in 1:50])
        v = Float64.([-cos(π * i/20) * sin(π * j/25) for i in 1:40, j in 1:50])
        fig = plot_flow(u, v; lic_length=8, title="LIC+Streamlines test")
        @test fig isa Figure
    end

    @testset "Stage 4a — typed plot recipes render without error" begin
        Random.seed!(31)
        n = 256
        t = DateTime(2026, 1, 1) .+ Hour.(0:n-1)
        uv = (cos.(2π * 0.1 .* (0:n-1)) .+ 0.05 .* randn(n)) .* u"m/s"
        vv = (sin.(2π * 0.1 .* (0:n-1)) .+ 0.05 .* randn(n)) .* u"m/s"
        uf, vf = Unitful.ustrip.(uv), Unitful.ustrip.(vv)

        # TimeSeriesVector / TimeSeriesMatrix / TimeSeriesCollection
        ts = TimeSeriesVector(t, uv, "u", (;))
        fig_ts = plot_timeseries(ts)
        @test fig_ts isa Figure
        ax_ts = fig_ts.content[1]
        @test occursin("m s", string(ax_ts.ylabel[]))  # unit string reaches the axis label

        tm = TimeSeriesMatrix(t, hcat(uf, vf) .* u"m/s", ["u", "v"], "currents", (;))
        @test plot_timeseries(tm) isa Figure

        tc = TimeSeriesCollection([ts, TimeSeriesVector(t, vv, "v", (;))], "uv", (;))
        @test plot_timeseries(tc) isa Figure

        # SpectralEstimate
        se = ValTools.JLab.spectral_multitaper(uf, 1.0; ntapers=5)
        @test plot_spectrum(se) isa Figure

        # RotarySpectralEstimate
        rse = rotary_spectrum(uf, vf; dt_hours=1.0)
        @test plot_rotary_spectrum(rse) isa Figure

        # RotaryCoherenceEstimate
        rce = rotary_coherence(uf, vf, uf .+ 0.1 .* randn(n), vf .+ 0.1 .* randn(n); dt_hours=1.0)
        @test plot_rotary_coherence(rce) isa Figure

        # CrossSpectralEstimate
        cse = cross_coherence(uf, uf .+ 0.1 .* randn(n); dt=1.0)
        @test plot_cross_spectrum(cse) isa Figure

        # EllipsePolarizationEstimate (ci=true by default, exercises the CI ribbons)
        epe = ellipse_polarization(uf, vf; dt_hours=1.0)
        @test plot_ellipse_polarization(epe) isa Figure

        # ColocatedObservation
        obs_ts = TimeSeriesVector(t, uv .+ 0.02 .* randn(n) .* u"m/s", "obs", (;))
        co = ValTools.ColocatedObservation(ts, obs_ts, 3.2, (rmse=0.05, correlation=0.98))
        @test plot_colocation(co) isa Figure
    end

    # Regression tests for the `:sym in names(::DataFrame)` bug — DataFrames
    # 1.x `names()` returns Vector{String}, so a Symbol is never `in` it and
    # the check was always false. Fixed to `hasproperty(df, :sym)`.
    @testset "ndbc_waves — :Hs column now actually reached" begin
        path = tempname() * "_42001.txt"
        write(path, "YY MM DD hh mm WVHT WSPD WDIR GST\n" *
                     "2026 01 01 00 00 1.5 10.0 180 12.0\n" *
                     "2026 01 01 01 00 1.6 10.5 185 12.5\n")
        try
            nd = NDBCLoader(path)
            @test nd.stations == ["42001"]

            waves = ndbc_waves(nd)
            @test !isempty(waves)
            @test haskey(waves, "42001")
            @test hasproperty(waves["42001"], :Hs)
            @test waves["42001"].Hs == [1.5, 1.6]

            # ndbc_winds already used the string form and should be unaffected
            winds = ndbc_winds(nd)
            @test !isempty(winds)
            @test hasproperty(winds["42001"], :wspd)
        finally
            rm(path; force=true)
        end
    end

    @testset "rafos_velocity_estimates — no longer always empty" begin
        df = DataFrame(float_id=["F1", "F1", "F2", "F2"],
                       time=[DateTime(2026,1,1,0,0), DateTime(2026,1,2,0,0),
                             DateTime(2026,1,1,0,0), DateTime(2026,1,2,0,0)],
                       lon=[-90.0, -90.1, -91.0, -91.2],
                       lat=[25.0, 25.1, 26.0, 26.2])
        r = RAFOSLoader(df)  # bypasses the file-parsing outer constructor

        vel = rafos_velocity_estimates(r)
        @test !isempty(vel)
        @test nrow(vel) == 2  # one 24h estimate per float, within the default [12,96]h window
        @test hasproperty(vel, :u) && hasproperty(vel, :v) && hasproperty(vel, :speed)
        @test all(isfinite, vel.speed)
        @test Set(vel.float_id) == Set(["F1", "F2"])
    end

    @testset "RAFOSLoader outer constructor — bbox filter and :depth column now applied" begin
        path = tempname() * ".csv"
        write(path, "float_id,lon,lat,pressure\n" *
                     "F1,-90.0,25.0,100.0\n" *
                     "F1,-91.0,26.0,200.0\n" *
                     "F2,-70.0,10.0,150.0\n")
        try
            # bbox = (lon_min, lat_min, lon_max, lat_max); F2's lon=-70 falls outside
            rl = RAFOSLoader(path; bbox=(-95.0, 20.0, -85.0, 30.0))
            @test hasproperty(rl.df, :depth)
            @test all(isapprox.(rl.df.depth, rl.df.pressure .* 1.019716))
            @test nrow(rl.df) == 2  # F2 dropped by the (now-working) bbox filter
        finally
            rm(path; force=true)
        end
    end

    # Stage 2 of the typed-ingest plan: additive `_ts` accessors returning
    # Types.TimeSeriesVector/Matrix/Collection alongside the existing raw
    # accessors (which are left unchanged and still tested above).
    @testset "ies_travel_time_ts" begin
        path = tempname() * ".nc"
        ds = NCDataset(path, "c")
        defDim(ds, "n", 5)
        defVar(ds, "time", collect(DateTime(2026,1,1):Hour(1):DateTime(2026,1,1,4)), ("n",))
        v = defVar(ds, "tau", Float64, ("n",)); v[:] = [1.0, 1.1, 1.2, 1.15, 1.05]
        v.attrib["units"] = "s"
        close(ds)
        try
            loader = IESLoader(path; site="TEST1", depth=1000.0)
            ts = ies_travel_time_ts(loader)
            @test ts isa ValTools.TimeSeriesVector
            @test Unitful.unit(ts.value[1]) == u"s"
            @test length(ts.value) == 5
            @test ts.time[1] == DateTime(2026,1,1)
            close(loader)
        finally
            rm(path; force=true)
        end
    end

    @testset "mooring_current_ts" begin
        path = tempname() * ".nc"
        ds = NCDataset(path, "c")
        defDim(ds, "time", 4); defDim(ds, "depth", 3)
        defVar(ds, "time", collect(DateTime(2026,1,1):Hour(6):DateTime(2026,1,1,18)), ("time",))
        defVar(ds, "depth", [10.0, 50.0, 100.0], ("depth",))
        um = defVar(ds, "u", Float64, ("time","depth")); um[:,:] = reshape(1.0:12.0, 4, 3)
        vm = defVar(ds, "v", Float64, ("time","depth")); vm[:,:] = reshape(-1.0:-1:-12, 4, 3)
        um.attrib["units"] = "m s-1"
        close(ds)
        try
            loader = MooringCurrentLoader(path; site="M1")
            res = mooring_current_ts(loader)
            @test res.u isa ValTools.TimeSeriesMatrix
            @test res.u.channels == ["10.0", "50.0", "100.0"]
            @test Unitful.unit(res.u.value[1,1]) == u"m/s"
            @test size(res.u.value) == (4, 3)
            close(loader)
        finally
            rm(path; force=true)
        end

        # 1-D point meter (no depth dimension) collapses to a single channel
        path1d = tempname() * ".nc"
        ds1d = NCDataset(path1d, "c")
        defDim(ds1d, "time", 4)
        defVar(ds1d, "time", collect(DateTime(2026,1,1):Hour(6):DateTime(2026,1,1,18)), ("time",))
        u1 = defVar(ds1d, "u", Float64, ("time",)); u1[:] = [1.0,2.0,3.0,4.0]
        v1 = defVar(ds1d, "v", Float64, ("time",)); v1[:] = [-1.0,-2.0,-3.0,-4.0]
        close(ds1d)
        try
            loader1d = MooringCurrentLoader(path1d; site="PT1")
            res1d = mooring_current_ts(loader1d)
            @test size(res1d.u.value) == (4, 1)
            @test res1d.u.channels == ["0"]
            close(loader1d)
        finally
            rm(path1d; force=true)
        end

        # Swapped (depth, time) axes must be caught, not silently misread
        pathbad = tempname() * ".nc"
        dsbad = NCDataset(pathbad, "c")
        defDim(dsbad, "time", 4); defDim(dsbad, "depth", 3)
        defVar(dsbad, "time", collect(DateTime(2026,1,1):Hour(6):DateTime(2026,1,1,18)), ("time",))
        ub = defVar(dsbad, "u", Float64, ("depth","time")); ub[:,:] = reshape(1.0:12.0, 3, 4)
        vb = defVar(dsbad, "v", Float64, ("depth","time")); vb[:,:] = reshape(-1.0:-1:-12, 3, 4)
        close(dsbad)
        try
            loaderbad = MooringCurrentLoader(pathbad; site="BAD")
            @test_throws ErrorException mooring_current_ts(loaderbad)
            close(loaderbad)
        finally
            rm(pathbad; force=true)
        end
    end

    @testset "ndbc_winds_ts" begin
        path = tempname() * "_42001.txt"
        write(path, "YY MM DD hh mm WSPD WDIR GST\n" *
                     "2026 01 01 00 00 10.0 180 12.0\n" *
                     "2026 01 01 01 00 10.5 185 12.5\n")
        try
            nd = NDBCLoader(path)
            coll = ndbc_winds_ts(nd)
            @test coll isa ValTools.TimeSeriesCollection
            @test length(coll) == 1
            s1 = coll["42001"]
            @test s1.name == "42001"
            @test Unitful.unit(s1.value[1]) == u"m/s"
            @test Unitful.ustrip.(s1.value) == [10.0, 10.5]
        finally
            rm(path; force=true)
        end

        # No wind columns anywhere -> errors rather than an empty collection
        path2 = tempname() * "_99999.txt"
        write(path2, "YY MM DD hh mm WVHT\n2026 01 01 00 00 1.5\n")
        try
            nd2 = NDBCLoader(path2)
            @test_throws ErrorException ndbc_winds_ts(nd2)
        finally
            rm(path2; force=true)
        end
    end

    # Stage 3 of the typed-ingest plan: multi-series Dispatch methods
    # (TimeSeriesMatrix / TimeSeriesCollection) and typed compute_metrics/
    # taylor_stats. `Dispatch.` prefix needed since these names collide
    # with Statistics' mean/std/var (imported separately above).
    @testset "Dispatch — TimeSeriesMatrix validate (by-name, not by position)" begin
        t = Dates.now() .+ Dates.Second.(0:99)
        model = TimeSeriesMatrix(t, hcat(randn(100), randn(100), randn(100)) .* u"m/s",
                                 ["10m", "50m", "100m"], "model", (;))
        # obs built with channels in a DIFFERENT order than model
        obs = TimeSeriesMatrix(t, hcat(Unitful.ustrip.(model.value[:,3]),
                                        Unitful.ustrip.(model.value[:,1]),
                                        Unitful.ustrip.(model.value[:,2])) .* u"m/s" .+ 0.05u"m/s",
                               ["100m", "10m", "50m"], "obs", (;))

        res = ValTools.Dispatch.validate(model, obs)
        @test Set(keys(res)) == Set(["10m", "50m", "100m"])
        for ch in ("10m", "50m", "100m")
            @test isapprox(Unitful.ustrip(res[ch].rmse), 0.05; atol=1e-8)
        end

        bad = TimeSeriesMatrix(t, hcat(randn(100), randn(100)) .* u"m/s", ["10m", "10m"], "bad", (;))
        @test_throws ErrorException ValTools.Dispatch.validate(bad, obs)
    end

    @testset "Dispatch — TimeSeriesCollection mean/validate (ragged, by-name)" begin
        t1 = Dates.now() .+ Dates.Second.(0:49)
        t2 = Dates.now() .+ Dates.Hour.(0:19)  # different length AND different clock
        model_c = TimeSeriesCollection([
            TimeSeriesVector(t1, (randn(50) .+ 5.0) .* u"m/s", "F1", (;)),
            TimeSeriesVector(t2, (randn(20) .+ 2.0) .* u"m/s", "F2", (;)),
        ], "model_floats", (;))
        # obs built with series in swapped order relative to model_c
        obs_c = TimeSeriesCollection([
            TimeSeriesVector(t2, (Unitful.ustrip.(model_c["F2"].value) .+ 0.1) .* u"m/s", "F2", (;)),
            TimeSeriesVector(t1, (Unitful.ustrip.(model_c["F1"].value) .+ 0.1) .* u"m/s", "F1", (;)),
        ], "obs_floats", (;))

        m = ValTools.Dispatch.mean(model_c)
        @test Set(keys(m)) == Set(["F1", "F2"])

        vres = ValTools.Dispatch.validate(model_c, obs_c)
        @test Set(keys(vres)) == Set(["F1", "F2"])
        for k in ("F1", "F2")
            @test isapprox(Unitful.ustrip(vres[k].rmse), 0.1; atol=1e-8)
        end
    end

    @testset "compute_metrics / taylor_stats — typed unit auto-convert" begin
        n = 200
        t3 = Dates.now() .+ Dates.Second.(0:n-1)
        obsv = randn(n) .+ 5.0
        obs_ts = TimeSeriesVector(t3, obsv .* u"m/s", "obs", (;))
        model_ms = TimeSeriesVector(t3, (obsv .+ 0.02) .* u"m/s", "model_ms", (;))
        # Same physical values as model_ms, expressed in cm/s -- this is the
        # exact mixed-unit scenario commit 801b386 fixed for Dispatch;
        # compute_metrics/taylor_stats must not reintroduce that bug class.
        model_cms = TimeSeriesVector(t3, (obsv .+ 0.02) .* 100 .* u"cm/s", "model_cms", (;))

        cm1 = compute_metrics(obs_ts, model_ms)
        cm2 = compute_metrics(obs_ts, model_cms)
        @test isapprox(cm1["rmse"], cm2["rmse"]; atol=1e-9)
        @test isapprox(cm1["correlation"], cm2["correlation"]; atol=1e-9)

        ts1 = taylor_stats(obs_ts, model_ms)
        ts2 = taylor_stats(obs_ts, model_cms)
        @test isapprox(ts1["rms_diff"], ts2["rms_diff"]; atol=1e-9)
    end

    @testset "Stage 4 — spectral estimators accept TimeSeriesVector" begin
        n = 256
        t = DateTime(2026,1,1) .+ Hour.(0:n-1)  # regular, dt = 1 hour
        uv = cos.(2π * 0.1 .* (0:n-1)) .* u"m/s"
        vv = sin.(2π * 0.1 .* (0:n-1)) .* u"m/s"
        uts = TimeSeriesVector(t, uv, "u", (;))
        vts = TimeSeriesVector(t, vv, "v", (;))

        spec_typed = rotary_spectrum(uts, vts)
        spec_raw = rotary_spectrum(Unitful.ustrip.(uv), Unitful.ustrip.(vv); dt_hours=1.0)
        @test spec_typed.freq == spec_raw.freq
        @test spec_typed.S_ccw == spec_raw.S_ccw

        # v's SAME physical values reported in cm/s must recover the exact
        # same spectrum after conversion to u's unit (m/s)
        vv_cms = Unitful.ustrip.(vv) .* 100 .* u"cm/s"
        vts_cms = TimeSeriesVector(t, vv_cms, "v_cms", (;))
        spec_mixed = rotary_spectrum(uts, vts_cms)
        @test isapprox(spec_mixed.S_ccw, spec_typed.S_ccw; rtol=1e-8)
        @test isapprox(spec_mixed.S_cw, spec_typed.S_cw; rtol=1e-8)

        # Mismatched time axes must error, not silently misalign
        vts_shifted = TimeSeriesVector(t .+ Minute(1), vv, "v_shifted", (;))
        @test_throws ErrorException rotary_spectrum(uts, vts_shifted)

        # Irregular sampling must error, not silently average a dt
        t_irr = copy(t); t_irr[end] += Hour(5)
        uts_irr = TimeSeriesVector(t_irr, uv, "u_irr", (;))
        vts_irr = TimeSeriesVector(t_irr, vv, "v_irr", (;))
        @test_throws ErrorException rotary_spectrum(uts_irr, vts_irr)

        # cross_coherence: x, y need not share a physical unit
        x = TimeSeriesVector(t, (cos.(2π*0.08 .* (0:n-1)) .+ 0.2 .* randn(n)) .* u"m/s", "x", (;))
        y = TimeSeriesVector(t, (cos.(2π*0.08 .* (0:n-1)) .+ 0.2 .* randn(n)) .* u"Pa", "y", (;))
        cc = cross_coherence(x, y)
        @test cc isa ValTools.CrossSpectralEstimate

        # ellipse_polarization typed matches raw
        ep_typed = ellipse_polarization(uts, vts)
        ep_raw = ellipse_polarization(Unitful.ustrip.(uv), Unitful.ustrip.(vv); dt_hours=1.0)
        @test ep_typed.d1 == ep_raw.d1
    end
end
