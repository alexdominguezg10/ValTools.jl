# Reproduce Fig. 1 of Jonathan Lilly's unfunded Flare NSF proposal: a map
# of the deployment location and full track of GDP drifter 44000 (WMO
# 4100571), identified via NOAA AOML ERDDAP against four independent checks
# (see memory: project_wavenumber_mooring_plan). Uses the FULL record
# (2004-10-13 -> 2007-05-15), not the clean 2005-only segment used for the
# spectral figures (Fig. 2/3), since Fig. 1 in the proposal shows the whole
# deployment track.
#
# No new library code: this is a plain lon/lat scatter/line, following this
# project's plotting house style (CairoMakie via ValTools, axis-equal map,
# professional labeling) -- see memory: feedback_plotting_standards.
# Configuration is via the CONST block below (this repo's established
# convention for one-off analysis/plot scripts -- see e.g.
# jlab_crosscheck_export.jl's `const N_CASES` -- rather than a CLI-arg
# package, which isn't used anywhere else in scripts/).
#
# Usage (from ValTools.jl root):
#   julia --project=envs/cpu scripts/flare_fig1_map.jl

using ValTools, CairoMakie
using Dates, Printf

# ── Config ───────────────────────────────────────────────────────────────
const CSV_PATH = "/Volumes/DATA_SSD/gdp_44000_full.csv"
const OUT_PATH = joinpath(@__DIR__, "..", "results", "flare_reproduction", "flare_fig1_map.png")
const FIGSIZE = (1000, 850)
const DPI = 2               # px_per_unit for CairoMakie save
const MARKERSIZE = 3.0
const TITLESIZE = 16
const LABELSIZE = 14
const TICKSIZE = 12
const LAND_COLOR = (:gray70, 1.0)   # matches this project's land_color convention (ocean_panel.jl)

# GSHHS (Global Self-consistent Hierarchical High-resolution Shoreline)
# low-resolution land polygons (level 1), already on disk locally from
# another project's MATLAB coastline toolbox -- no GeoMakie/Shapefile.jl
# dependency added just for one map's land outline. Plain WGS84 lon/lat
# (confirmed via the .prj file), so no reprojection needed.
const GSHHS_SHP = "/Users/alexdominguez/ADominguez/TOOLS/MATLAB/costas/GSHHS_shp/l/GSHHS_l_L1.shp"

# ── Minimal dependency-free ESRI Shapefile (.shp) polygon reader ─────────
# Only supports what GSHHS_l_L1.shp actually contains: shape type 5
# (Polygon). Format: 100-byte header (big-endian file code/length,
# little-endian version/shape type/bbox), then records: 8-byte
# big-endian (record#, content length in 16-bit words) header, followed
# by little-endian shape content (type, bbox, numParts, numPoints, parts
# index array, then numPoints (x,y) pairs). Each part of a polygon record
# is plotted as its own filled ring -- correct for a level-1 (land) layer,
# where holes (lakes) are a separate GSHHS level, not part-of-polygon
# holes here.
function read_shp_polygons(path::String; lon_min=-180.0, lon_max=180.0, lat_min=-90.0, lat_max=90.0)
    rings = Vector{Point2f}[]
    open(path, "r") do io
        seek(io, 100)   # skip the 100-byte main file header
        while !eof(io)
            _recnum = ntoh(read(io, Int32))
            content_words = ntoh(read(io, Int32))
            content_bytes = content_words * 2
            rec_end = position(io) + content_bytes
            shape_type = read(io, Int32)   # little-endian on this platform
            if shape_type != 5   # not a Polygon record; skip its content
                seek(io, rec_end)
                continue
            end
            bxmin, bymin, bxmax, bymax = read(io, Float64), read(io, Float64), read(io, Float64), read(io, Float64)
            overlaps = bxmax >= lon_min && bxmin <= lon_max && bymax >= lat_min && bymin <= lat_max
            num_parts = read(io, Int32)
            num_points = read(io, Int32)
            parts = [read(io, Int32) for _ in 1:num_parts]
            if !overlaps
                seek(io, rec_end)
                continue
            end
            pts = Vector{Point2f}(undef, num_points)
            for i in 1:num_points
                x, y = read(io, Float64), read(io, Float64)
                pts[i] = Point2f(x, y)
            end
            for p in 1:num_parts
                i0 = parts[p] + 1
                i1 = p < num_parts ? parts[p + 1] : num_points
                push!(rings, pts[i0:i1])
            end
            seek(io, rec_end)
        end
    end
    return rings
end

# ── Minimal CSV reader (no CSV.jl dep in this project; two header lines:
# names then units) ─────────────────────────────────────────────────────
function read_gdp_csv(path::String)
    lines = readlines(path)
    header = split(lines[1], ",")
    col = Dict(name => i for (i, name) in enumerate(header))
    n = length(lines) - 2
    time = Vector{DateTime}(undef, n)
    lat = Vector{Float64}(undef, n)
    lon = Vector{Float64}(undef, n)
    ve = Vector{Float64}(undef, n)
    vn = Vector{Float64}(undef, n)
    for (k, line) in enumerate(@view lines[3:end])
        f = split(line, ",")
        tstr = f[col["time"]]
        tstr = endswith(tstr, "Z") ? tstr[1:end-1] : tstr
        time[k] = DateTime(tstr)
        lat[k] = parse(Float64, f[col["latitude"]])
        lon[k] = parse(Float64, f[col["longitude"]])
        ve[k] = parse(Float64, f[col["ve"]])
        vn[k] = parse(Float64, f[col["vn"]])
    end
    return (; time, lat, lon, ve, vn)
end

function main()
    data = read_gdp_csv(CSV_PATH)
    n = length(data.time)
    println("Loaded $n rows from $CSV_PATH")
    println("Time span: $(data.time[1]) -> $(data.time[end])")

    # Longitude in the ERDDAP CSV is signed (-73.08 etc); the proposal's
    # Fig. 1 x-axis runs 270-295 degrees_east (0-360 convention) per the
    # memory note ("2005 lon span 271.1-294.6E vs Fig. 1 x-axis exactly
    # 270-295E") -- convert for a directly comparable axis.
    lon360 = mod.(data.lon, 360.0)

    deploy_lat, deploy_lon = data.lat[1], mod(data.lon[1], 360.0)
    @printf("Deployment: %.4f N, %.4f E (%.4f W), %s\n",
            deploy_lat, deploy_lon, -data.lon[1], data.time[1])

    # Crop the coastline query to the track's bounding box (+padding), in
    # GSHHS's native signed lon convention -- avoids parsing/plotting the
    # rest of the world's coastline for a regional drifter map.
    pad = 3.0
    lon_lo, lon_hi = minimum(data.lon) - pad, maximum(data.lon) + pad
    lat_lo, lat_hi = minimum(data.lat) - pad, maximum(data.lat) + pad
    println("Reading GSHHS land polygons (lon $lon_lo:$lon_hi, lat $lat_lo:$lat_hi) ...")
    land_rings = read_shp_polygons(GSHHS_SHP; lon_min=lon_lo, lon_max=lon_hi, lat_min=lat_lo, lat_max=lat_hi)
    println("  $(length(land_rings)) land ring(s) overlapping the map region")

    fig = Figure(size=FIGSIZE, fontsize=14)

    ax = Axis(fig[1, 1];
        xlabel="Longitude [°E]", ylabel="Latitude [°N]",
        title="GDP drifter 44000 (WMO 4100571) — full track\n$(Dates.format(data.time[1], "yyyy-mm-dd")) to $(Dates.format(data.time[end], "yyyy-mm-dd"))",
        titlesize=TITLESIZE, xlabelsize=LABELSIZE, ylabelsize=LABELSIZE,
        xticklabelsize=TICKSIZE, yticklabelsize=TICKSIZE,
        aspect=DataAspect())

    # Land, drawn first so the track renders on top of it. Ring longitudes
    # converted to the same 0-360 convention as lon360 below.
    for ring in land_rings
        ring360 = [Point2f(mod(p[1], 360.0), p[2]) for p in ring]
        poly!(ax, ring360; color=LAND_COLOR, strokecolor=(:gray40, 1.0), strokewidth=0.5)
    end

    # Track colored by elapsed time (days since deployment) so the temporal
    # evolution of the path is visible, not just its shape.
    t_days = [Dates.value(Dates.Millisecond(t - data.time[1])) / (1000 * 3600 * 24) for t in data.time]

    lines!(ax, lon360, data.lat; color=(:steelblue, 0.35), linewidth=1.0)
    sc = scatter!(ax, lon360, data.lat; color=t_days, colormap=:viridis,
                  markersize=MARKERSIZE)
    Colorbar(fig[1, 2], sc; label="Days since deployment")

    # Deployment marker (orange circle, matching the proposal's Fig. 1 styling)
    scatter!(ax, [deploy_lon], [deploy_lat]; color=:orange, markersize=22,
             marker=:circle, strokewidth=2, strokecolor=:black, label="Deployment (off Cuba)")
    text!(ax, deploy_lon, deploy_lat; text="  deploy\n  $(Dates.format(data.time[1], "yyyy-mm-dd"))",
          fontsize=11, align=(:left, :top), color=:black)

    # End-of-record marker
    scatter!(ax, [lon360[end]], [data.lat[end]]; color=:red, markersize=16,
             marker=:xcross, strokewidth=2, label="End of record")

    axislegend(ax; position=:lt, framevisible=true, labelsize=11)

    # Pin view to the track's own extent -- a matched land ring's bounding
    # box only needs to OVERLAP the padded query region, not fit inside
    # it, so an unbounded polygon (e.g. the mainland coastline) could
    # otherwise blow the axis limits out to whatever it spans.
    view_pad = 1.0
    xlims!(ax, minimum(lon360) - view_pad, maximum(lon360) + view_pad)
    ylims!(ax, minimum(data.lat) - view_pad, maximum(data.lat) + view_pad)

    mkpath(dirname(OUT_PATH))
    save(OUT_PATH, fig; px_per_unit=DPI)
    println("Saved: $OUT_PATH")
end

main()
