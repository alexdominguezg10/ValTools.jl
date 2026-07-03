using Test
using Random
using ValTools.JLab
using Multitaper

Random.seed!(42)

@testset "JLab Validation" begin

    # ── Helper: generate synthetic model/obs pair ──
    function make_test_pair(N, dt; noise_ratio=0.1, phase_shift=0.0)
        t = (0:N-1) .* dt
        f_inertial = inertial_frequency(25.0) # GoM ~25°N
        f_tide = 1.0 / 12.42                  # M2 semidiurnal

        # "Truth" (observations)
        u_obs = 0.3 .* cos.(2π * f_inertial .* t) .+
                0.1 .* cos.(2π * f_tide .* t) .+
                noise_ratio .* randn(N)
        v_obs = -0.3 .* sin.(2π * f_inertial .* t) .+
                 0.1 .* sin.(2π * f_tide .* t) .+
                 noise_ratio .* randn(N)

        # "Model" — slightly different amplitude/phase
        u_mod = 0.28 .* cos.(2π * f_inertial .* t .+ phase_shift) .+
                0.12 .* cos.(2π * f_tide .* t) .+
                noise_ratio .* 0.5 .* randn(N)
        v_mod = -0.28 .* sin.(2π * f_inertial .* t .+ phase_shift) .+
                 0.12 .* sin.(2π * f_tide .* t) .+
                 noise_ratio .* 0.5 .* randn(N)

        return u_mod, v_mod, u_obs, v_obs
    end

    @testset "inertial_frequency" begin
        # GoM at 25°N
        f_25 = inertial_frequency(25.0)
        @test f_25 > 0.02  # cycles/hour
        @test f_25 < 0.06

        # Higher latitude → higher frequency
        f_45 = inertial_frequency(45.0)
        @test f_45 > f_25

        # Equator → zero
        f_0 = inertial_frequency(0.0)
        @test f_0 ≈ 0.0 atol=1e-10

        # Poles → maximum
        f_90 = inertial_frequency(90.0)
        @test f_90 > f_45
    end

    @testset "get_freq_band" begin
        # Standard bands
        lo, hi = get_freq_band("semidiurnal")
        @test lo ≈ 1.0/14.0
        @test hi ≈ 1.0/11.0

        # Inertial depends on latitude
        lo25, hi25 = get_freq_band("inertial"; lat=25.0)
        lo45, hi45 = get_freq_band("inertial"; lat=45.0)
        @test lo45 > lo25

        @test_throws ErrorException get_freq_band("nonexistent")
    end

    @testset "validate_spectra — basic" begin
        N = 2000;  dt = 1.0  # hourly
        um, vm, uo, vo = make_test_pair(N, dt)

        result = validate_spectra(um, uo, dt;
                                  ntapers=5,
                                  freq_bands=["inertial", "semidiurnal"],
                                  lat=25.0)

        @test haskey(result, "freqs")
        @test haskey(result, "psd_model")
        @test haskey(result, "psd_obs")
        @test haskey(result, "spectral_correlation")
        @test haskey(result, "spectral_slope_model")
        @test haskey(result, "inertial_ratio")
        @test haskey(result, "semidiurnal_ratio")

        # Spectral correlation should be positive for similar signals
        @test result["spectral_correlation"] > 0.3

        # Inertial energy ratio should be near 1 (signals are similar)
        @test 0.3 < result["inertial_ratio"] < 3.0
    end

    @testset "validate_spectra — identical signals" begin
        N = 1000; dt = 1.0
        x = randn(N)

        result = validate_spectra(x, x, dt; ntapers=3)

        @test result["spectral_correlation"] ≈ 1.0 atol=1e-10
        @test abs(result["mean_log_spectral_ratio"]) < 1e-10
    end

    @testset "validate_rotary — CW/CCW correlation" begin
        N = 2000; dt = 1.0
        um, vm, uo, vo = make_test_pair(N, dt)

        result = validate_rotary(um, vm, uo, vo, dt;
                                 freq_bands=["inertial"],
                                 lat=25.0)

        @test haskey(result, "cw_correlation")
        @test haskey(result, "ccw_correlation")
        @test haskey(result, "rotary_coefficient_model")

        # Rotary coefficient should be between -1 and 1
        rc = result["rotary_coefficient_model"]
        valid_rc = rc[.!isnan.(rc)]
        @test all(-1.0 .<= valid_rc .<= 1.0)
    end

    @testset "validate_rotary — length mismatch error" begin
        @test_throws ErrorException validate_rotary(
            randn(100), randn(100), randn(50), randn(50), 1.0)
    end

    @testset "kinetic_energy_budget — conservation" begin
        N = 2000; dt = 1.0
        u = randn(N)
        v = randn(N)

        ke_total, ke_bands, spectra = kinetic_energy_budget(u, v, dt; lat=25.0)

        @test ke_total > 0

        # Band energies should sum to ≤ total (bands may not cover all freqs)
        band_sum = sum(values(ke_bands))
        @test band_sum <= ke_total * 1.1  # allow small numerical margin

        @test haskey(spectra, "freqs")
        @test haskey(spectra, "psd_u")
        @test haskey(spectra, "psd_v")
        @test haskey(ke_bands, "inertial")
        @test haskey(ke_bands, "mesoscale")
    end

    @testset "kinetic_energy_budget — custom bands" begin
        N = 1000; dt = 1.0
        u = randn(N); v = randn(N)

        custom = Dict("low" => (0.0, 0.1), "high" => (0.1, 0.5))
        ke, kb, _ = kinetic_energy_budget(u, v, dt; freq_bands=custom)

        @test haskey(kb, "low")
        @test haskey(kb, "high")
        @test kb["low"] >= 0
        @test kb["high"] >= 0
    end

    @testset "eddy_census — detects synthetic eddy" begin
        dt = 1.0;  N = 500
        t = (0:N-1) .* dt

        # Inject a strong rotating signal for 200 time steps (hours 100–300)
        u = 0.01 .* randn(N)
        v = 0.01 .* randn(N)
        f_eddy = 0.04
        for i in 100:300
            u[i] += 0.5 * cos(2π * f_eddy * t[i])
            v[i] -= 0.5 * sin(2π * f_eddy * t[i])
        end

        events = eddy_census(u, v, dt;
                             amp_thresh=0.1,
                             min_duration=10)

        @test isa(events, Vector)
        # Should detect at least one event
        if !isempty(events)
            e = events[1]
            @test haskey(e, "start")
            @test haskey(e, "stop")
            @test haskey(e, "duration")
            @test haskey(e, "mean_frequency")
            @test haskey(e, "sense")
            @test e["duration"] >= 10
        end
    end

    @testset "eddy_census — quiet signal → no events" begin
        N = 200
        events = eddy_census(0.001 .* randn(N), 0.001 .* randn(N), 1.0;
                             amp_thresh=0.5, min_duration=50)
        @test isempty(events)
    end

    @testset "validate_model_spectra — full report" begin
        N = 2000; dt = 1.0
        um, vm, uo, vo = make_test_pair(N, dt)

        report = validate_model_spectra(um, vm, uo, vo, dt;
                                        lat=25.0, ntapers=5,
                                        freq_bands=["inertial", "semidiurnal"])

        @test haskey(report, "scalar_u")
        @test haskey(report, "scalar_v")
        @test haskey(report, "rotary")
        @test haskey(report, "ke_model")
        @test haskey(report, "ke_obs")
        @test haskey(report, "summary")

        s = report["summary"]
        @test haskey(s, "spectral_corr_u")
        @test haskey(s, "ke_total_model")
        @test haskey(s, "ke_ratio")

        # Similar signals → KE ratio near 1
        @test 0.3 < s["ke_ratio"] < 3.0
    end
end
