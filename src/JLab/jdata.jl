"""
JData — Reference Oceanographic Datasets

Loaders for curated datasets from Jonathan M. Lilly's jLab/jData,
available as public NetCDF files on Zenodo.

These provide ground-truth observations for model validation:
- **GulfDrifters**: Consolidated surface drifter data for GoM
- **GulfFlow**: Gridded surface currents from drifter data
- **GOMED**: Eddy database derived from drifter wavelet ridge analysis

**References:**
- Lilly, J. M. & P. Perez-Brunius (2021a). A gridded surface current product
  for the Gulf of Mexico from consolidated drifter measurements.
  Earth System Science Data, 13, 645–677.
  https://doi.org/10.5194/essd-13-645-2021
- Lilly, J. M. & P. Perez-Brunius (2021b). Extracting statistically
  significant eddy signals from large Lagrangian datasets using wavelet ridge
  analysis, with application to the Gulf of Mexico.
  Nonlinear Processes in Geophysics, 28, 181–212.
  https://doi.org/10.5194/npg-28-181-2021

**Data DOIs:**
- GulfDriftersOpen: https://doi.org/10.5281/zenodo.3985916
- GulfFlow:         https://doi.org/10.5281/zenodo.3978793
- GOMED:            https://doi.org/10.5281/zenodo.3978803
"""

# ============================================================================
# CATALOG
# ============================================================================

"""
    JDATA_CATALOG

Available JData datasets with Zenodo download URLs and descriptions.
"""
const JDATA_CATALOG = Dict{String, NamedTuple{(:url, :filename, :description, :doi, :citation),
    Tuple{String, String, String, String, String}}}(

    "gulfdrifters" => (
        url = "https://zenodo.org/api/records/4421585/files/GulfDriftersOpen.nc/content",
        filename = "GulfDriftersOpen.nc",
        description = "Surface drifter trajectories in the Gulf of Mexico " *
                      "(publicly available subset). Hourly position, velocity.",
        doi = "10.5281/zenodo.3985916",
        citation = "Lilly, J. M. & P. Perez-Brunius (2021). GulfDrifters: " *
                   "A consolidated surface drifter dataset for the Gulf of Mexico."
    ),

    "gulfflow_quarter" => (
        url = "https://zenodo.org/api/records/4421958/files/GulfFlow_Quarter.nc/content",
        filename = "GulfFlow_Quarter.nc",
        description = "Quarter-degree gridded surface currents for the GoM, " *
                      "derived from drifter data. Monthly climatology.",
        doi = "10.5281/zenodo.3978793",
        citation = "Lilly, J. M. & P. Perez-Brunius (2021). GulfFlow: " *
                   "A gridded surface current product for the Gulf of Mexico."
    ),

    "gulfflow_twelfth" => (
        url = "https://zenodo.org/api/records/4421958/files/GulfFlow_Twelfth.nc/content",
        filename = "GulfFlow_Twelfth.nc",
        description = "Twelfth-degree gridded surface currents for the GoM.",
        doi = "10.5281/zenodo.3978793",
        citation = "Lilly, J. M. & P. Perez-Brunius (2021). GulfFlow."
    ),

    "gomed" => (
        url = "https://zenodo.org/api/records/4453875/files/GOMED.nc/content",
        filename = "GOMED.nc",
        description = "Gulf of Mexico Eddy Dataset — coherent eddy-like events " *
                      "detected via wavelet ridge analysis of drifter data.",
        doi = "10.5281/zenodo.3978803",
        citation = "Lilly, J. M. & P. Perez-Brunius (2021). GOMED: " *
                   "The Gulf of Mexico Eddy Dataset."
    ),
)

# ============================================================================
# DOWNLOAD & CACHE
# ============================================================================

function _jdata_cache_dir()
    cache = get(ENV, "VALTOOLS_JDATA_DIR",
                joinpath(homedir(), ".valtools", "jdata"))
    mkpath(cache)
    return cache
end

"""
    jdata_list()

Print available JData datasets with descriptions.
"""
function jdata_list()
    println("Available JData datasets:")
    println("=" ^ 60)
    for (name, info) in sort(collect(JDATA_CATALOG))
        println("  $(name)")
        println("    $(info.description)")
        println("    DOI: $(info.doi)")
        println()
    end
end

"""
    jdata_download(name::String; force=false)

Download a JData dataset from Zenodo to local cache.

# Arguments
- `name`: Dataset name (see `jdata_list()`)
- `force`: Re-download even if cached

# Returns
- `path::String`: Local file path to the downloaded NetCDF

# Example
```julia
path = jdata_download("gulfdrifters")
```
"""
function jdata_download(name::String; force::Bool=false)
    haskey(JDATA_CATALOG, name) || error(
        "Unknown dataset: '$name'. Run jdata_list() to see available datasets.")

    info = JDATA_CATALOG[name]
    cache_dir = _jdata_cache_dir()
    local_path = joinpath(cache_dir, info.filename)

    if isfile(local_path) && !force
        @info "JData '$name' already cached at $local_path"
        return local_path
    end

    @info "Downloading JData '$name' from Zenodo..." url=info.url
    try
        download(info.url, local_path)
        @info "Downloaded to $local_path"
    catch e
        error("Failed to download '$name': $e\n" *
              "You can manually download from: $(info.url)\n" *
              "Place the file at: $local_path")
    end

    return local_path
end

# ============================================================================
# DATASET LOADERS
# ============================================================================

"""
    load_gulfdrifters(; download=true)

Load Gulf of Mexico surface drifter dataset.

# Returns
NamedTuple with:
- `lon::Vector{Vector{Float64}}`: Longitude trajectories (per drifter)
- `lat::Vector{Vector{Float64}}`: Latitude trajectories
- `u::Vector{Vector{Float64}}`: Eastward velocity (m/s)
- `v::Vector{Vector{Float64}}`: Northward velocity (m/s)
- `time::Vector{Vector{Float64}}`: Time (days since reference)
- `id::Vector{Int}`: Drifter IDs
- `n_drifters::Int`: Number of drifters

# References
Lilly & Perez-Brunius (2021a); DOI: 10.5281/zenodo.3985916
"""
function load_gulfdrifters(; download::Bool=true)
    if download
        path = jdata_download("gulfdrifters")
    else
        path = joinpath(_jdata_cache_dir(), JDATA_CATALOG["gulfdrifters"].filename)
        isfile(path) || error("GulfDrifters not found at $path. " *
                              "Run with download=true or jdata_download(\"gulfdrifters\")")
    end

    # NCDatasets is available via ValTools
    ds = _nc_open(path)

    try
        # GulfDriftersOpen.nc structure: ragged array format
        # Variables: lon, lat, cv (complex velocity), num, id
        result = _read_gulfdrifters(ds)
        return result
    finally
        close(ds)
    end
end

function _read_gulfdrifters(ds)
    var_names = keys(ds)

    lon_all = Float64.(ds["lon"][:])
    lat_all = Float64.(ds["lat"][:])
    time_all = Float64.(ds["time"][:])

    # Velocity: u/v in cm/s → m/s, or complex cv
    if "cv" in var_names
        cv = ds["cv"][:]
        u_all = Float64.(real.(cv)) ./ 100.0
        v_all = Float64.(imag.(cv)) ./ 100.0
    elseif "u" in var_names && "v" in var_names
        u_all = Float64.(ds["u"][:]) ./ 100.0   # cm/s → m/s
        v_all = Float64.(ds["v"][:]) ./ 100.0
    else
        u_all = zeros(length(lon_all))
        v_all = zeros(length(lon_all))
    end

    # Segment lengths: "row_size" (NetCDF) or "num" (Matlab-style)
    seg_key = "row_size" in var_names ? "row_size" :
              "num" in var_names ? "num" : nothing
    if seg_key !== nothing
        num = Int.(ds[seg_key][:])
    else
        num = [length(lon_all)]
    end

    n_drifters = length(num)
    lon  = Vector{Vector{Float64}}(undef, n_drifters)
    lat  = Vector{Vector{Float64}}(undef, n_drifters)
    u    = Vector{Vector{Float64}}(undef, n_drifters)
    v    = Vector{Vector{Float64}}(undef, n_drifters)
    time = Vector{Vector{Float64}}(undef, n_drifters)

    idx = 1
    for i in 1:n_drifters
        n = num[i]
        rng = idx:idx+n-1
        lon[i]  = lon_all[rng]
        lat[i]  = lat_all[rng]
        u[i]    = u_all[rng]
        v[i]    = v_all[rng]
        time[i] = time_all[rng]
        idx += n
    end

    ids = "id" in var_names ? Int.(ds["id"][:]) : collect(1:n_drifters)

    return (lon=lon, lat=lat, u=u, v=v, time=time,
            id=ids, n_drifters=n_drifters,
            source="GulfDriftersOpen.nc",
            citation=JDATA_CATALOG["gulfdrifters"].citation)
end

"""
    load_gulfflow(; resolution="quarter", download=true)

Load gridded GoM surface current product.

# Arguments
- `resolution`: "quarter" (1/4°) or "twelfth" (1/12°)

# Returns
NamedTuple with:
- `lon::Vector{Float64}`: Longitude grid
- `lat::Vector{Float64}`: Latitude grid
- `u::Array{Float64, 3}`: Mean eastward velocity (lon, lat, time) [m/s]
- `v::Array{Float64, 3}`: Mean northward velocity (lon, lat, time) [m/s]
- `time`: Time vector

# References
Lilly & Perez-Brunius (2021a); DOI: 10.5281/zenodo.3978793
"""
function load_gulfflow(; resolution::String="quarter", download::Bool=true)
    name = "gulfflow_$(resolution)"
    haskey(JDATA_CATALOG, name) || error("Unknown resolution: $resolution")

    if download
        path = jdata_download(name)
    else
        path = joinpath(_jdata_cache_dir(), JDATA_CATALOG[name].filename)
        isfile(path) || error("GulfFlow not cached. Run with download=true")
    end

    ds = _nc_open(path)
    try
        var_names = keys(ds)

        lon = Float64.(ds["lon"][:])
        lat = Float64.(ds["lat"][:])

        u_key = "u" in var_names ? "u" : "um"
        v_key = "v" in var_names ? "v" : "vm"

        u = Float64.(ds[u_key][:])
        v = Float64.(ds[v_key][:])

        t = "time" in var_names ? ds["time"][:] : nothing

        return (lon=lon, lat=lat, u=u, v=v, time=t,
                resolution=resolution,
                source=JDATA_CATALOG[name].filename,
                citation=JDATA_CATALOG[name].citation)
    finally
        close(ds)
    end
end

"""
    load_gomed(; download=true)

Load Gulf of Mexico Eddy Dataset (wavelet-ridge-detected eddies from drifters).

# Returns
NamedTuple with eddy properties:
- `lon`, `lat`: Eddy center positions
- `radius`: Estimated radius (km)
- `frequency`: Ridge frequency (cycles/hour)
- `amplitude`: Wavelet amplitude
- `sense`: Rotation sense (+1 = CCW, -1 = CW)
- `duration`: Event duration (hours)

# References
Lilly & Perez-Brunius (2021b, NPG); DOI: 10.5281/zenodo.3978803
"""
function load_gomed(; download::Bool=true)
    if download
        path = jdata_download("gomed")
    else
        path = joinpath(_jdata_cache_dir(), JDATA_CATALOG["gomed"].filename)
        isfile(path) || error("GOMED not cached. Run with download=true")
    end

    ds = _nc_open(path)
    try
        var_names = keys(ds)
        result = Dict{String, Any}()

        for vn in ["lon", "lat", "radius", "frequency", "amplitude",
                    "sense", "duration", "id"]
            if vn in var_names
                result[vn] = ds[vn][:]
            end
        end

        result["source"] = "GOMED.nc"
        result["citation"] = JDATA_CATALOG["gomed"].citation

        return result
    finally
        close(ds)
    end
end

# ============================================================================
# NC HELPER
# ============================================================================

function _nc_open(path::String)
    # Use NCDatasets from ValTools parent
    try
        # NCDatasets should be available through ValTools
        return Main.NCDatasets.Dataset(path, "r")
    catch
        try
            # Fallback: try importing directly
            @eval import NCDatasets
            return NCDatasets.Dataset(path, "r")
        catch e
            error("Cannot open NetCDF file. Ensure NCDatasets.jl is available.\n" *
                  "File: $path\nError: $e")
        end
    end
end

# ============================================================================
# EXPORTS
# ============================================================================

# (exported via JLab.jl module)
