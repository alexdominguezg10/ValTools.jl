using Test
using ValTools
using DataFrames
using Dates
using Statistics
using NCDatasets
using CairoMakie

@testset "ValTools.jl" begin

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

    @testset "alongtrack_wavenumber_spectrum" begin
        n = 512
        dx = 2.0  # km
        x = range(0, step=dx, length=n)
        signal = sin.(2π .* x ./ 50.0)  # 50 km wavelength → k = 0.02 cpk

        k, psd = alongtrack_wavenumber_spectrum(signal, dx)
        @test length(k) > 0
        @test all(k .> 0)
        @test all(psd .>= 0)
    end

    @testset "isotropic_2d_spectrum" begin
        ny, nx = 64, 64
        dx, dy = 5.0, 5.0
        field = randn(ny, nx)

        k_iso, psd_iso = isotropic_2d_spectrum(field, dx, dy)
        @test length(k_iso) > 0
        @test all(psd_iso[isfinite.(psd_iso)] .>= 0)
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
end
