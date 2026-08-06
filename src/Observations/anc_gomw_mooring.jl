"""
    ANCMooringLoader

Loader for the ANC_GoMW_deep mooring format (CANEK-program Gulf of Mexico
ADCP current-meter `.mat` files, MATLAB v5 binary — NOT the v7.3/HDF5
format [`CANEKSectionLoader`](@ref) reads, and a different data product
entirely: a single moored ADCP's own depth-bin profile time series, not a
cross-channel transport section).

# Filename convention
`SITE-T####-INSTRUMENT-NS#####-Z####-INS##-REC##.mat`, e.g.
`PER-T3500-WH600DW-NS9723-Z3481-INS21-REC24.mat`: `SITE` (mooring site,
e.g. PER=Perdido), `T####` design depth (m), instrument model
(`LR75DW`=RDI Long Ranger 75 kHz, `WH300DW`/`WH600DW`=RDI Workhorse
300/600 kHz; `DW`=downward-looking), `NS#####` serial number, `Z####`
realized depth (m), `INS##`/`REC##` instrument/recovery cycle numbers.
Parsed opportunistically into `meta` — a non-matching filename just
leaves `meta = nothing`, it does not error.

# Fields (from the `.mat` file itself, not the filename)
`u`, `v`, `w`: velocity `[n_time x n_bins]`, m/s. `bins`: relative range
along the beam from the instrument, m — **not** absolute depth.
`instrument_depth`: per-timestep instrument depth, m (`profinst` in the
file). `depths`: a static, nominal absolute depth per bin,
`mean(instrument_depth) .+ sign * bins` — see the `looking` keyword.
`lon`/`lat`: mooring position (inside the file, not the filename).
`time`: converted from the file's `jd` (confirmed MATLAB `datenum`, both
by its value range — spanning the deployment's actual year, not a raw
Julian Day number — and directly by the data's own author) via the same
conversion [`CANEKSectionLoader`](@ref) uses.

# Depth-sign convention — flagged, not independently re-verified
`depths = instrument_depth .+ bins` when `looking=:down` (the default,
and what's auto-detected from a `DW` filename code), `.- bins` for
`:up`. This is the standard ADCP convention for a downward-looking
instrument (bins are range cells below the transducer) but has **not**
been independently checked against this array's own raw processing
scripts or known local bathymetry — if a computed depth looks physically
implausible (e.g. exceeding the site's water depth), check that before
trusting it.

# Requires MAT.jl
This is a weakdep-gated package extension (`ValToolsMATExt`, matching how
`CairoMakie`/`Multitaper`/`CUDA` support work elsewhere in this package):
the struct and `anc_mooring_profiles` live here and need nothing extra,
but the constructor itself (`ANCMooringLoader(filepath)`) only becomes
usable once `using MAT` is loaded alongside `using ValTools` — calling it
without that raises a plain `MethodError`, not a custom message, since a
same-signature stub here would conflict with the extension's real method
(the same constraint documented on `Metrics.rotary_spectrum`'s typed stub
in `spectral.jl`).
"""
mutable struct ANCMooringLoader
    filepath::String
    site::String
    lon::Float64
    lat::Float64
    time::Vector{DateTime}
    u::Matrix{Float64}
    v::Matrix{Float64}
    w::Union{Matrix{Float64}, Nothing}
    bins::Vector{Float64}
    instrument_depth::Vector{Float64}
    depths::Vector{Float64}
    temp::Union{Vector{Float64}, Nothing}
    meta::Union{NamedTuple, Nothing}
end

# The `ANCMooringLoader(filepath::AbstractString; looking=nothing, site=nothing)`
# convenience constructor is defined ONLY in ext/ValToolsMATExt/ (needs
# MAT.matread). No stub is declared here: unlike the plotting/GPU stubs
# elsewhere in this package (`function taylor_diagram end` etc.), this
# struct's auto-generated positional inner constructor already exists as
# a zero-method-conflict-free slot, so there is nothing to pre-declare —
# adding an empty `function ANCMooringLoader(filepath::AbstractString; kwargs...) end`
# stub here would itself be a real method occupying that call signature,
# and Julia forbids a package extension from later redefining the same
# method signature ("Method overwriting is not permitted"). Calling
# `ANCMooringLoader(path)` without `using MAT` therefore raises a plain
# `MethodError: no method matching ANCMooringLoader(::String)`, not a
# custom hint — the same tradeoff already documented on
# `Metrics.rotary_spectrum`'s typed-stub note in spectral.jl.

const _ANC_FILENAME_RE = r"^([A-Z]+)-T(\d+)-([A-Z0-9]+)-NS(\d+)-Z(\d+)-INS(\d+)-REC(\d+)"

function _parse_anc_filename(filepath::AbstractString)
    m = match(_ANC_FILENAME_RE, basename(filepath))
    m === nothing && return nothing
    site, design_depth, instrument, serial, realized_depth, ins_num, rec_num = m.captures
    return (
        site = site,
        design_depth_m = parse(Float64, design_depth),
        instrument = instrument,
        serial = serial,
        realized_depth_m = parse(Float64, realized_depth),
        ins_number = parse(Int, ins_num),
        recovery_number = parse(Int, rec_num),
        looking = occursin("DW", instrument) ? :down : (occursin("UP", instrument) ? :up : nothing),
    )
end

"""
    anc_mooring_profiles(r::ANCMooringLoader)

Returns `(u, v, time, depths, site, lon, lat, w, bins, instrument_depth,
temp)` — the first five fields match [`mooring_current_profiles`](@ref)'s
shape exactly (drop-in compatible with [`mooring_speed`](@ref),
[`mooring_direction`](@ref), [`mooring_variance_ellipse`](@ref),
[`mooring_progressive_vector`](@ref)), with the extra raw fields this
format provides (vertical velocity, relative bin range, the per-timestep
instrument depth `depths` was derived from, and temperature) appended for
anyone who needs them.
"""
function anc_mooring_profiles(r::ANCMooringLoader)
    return (u=r.u, v=r.v, time=r.time, depths=r.depths, site=r.site,
            lon=r.lon, lat=r.lat, w=r.w, bins=r.bins,
            instrument_depth=r.instrument_depth, temp=r.temp)
end

Base.close(::ANCMooringLoader) = nothing
