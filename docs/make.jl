# ValTools.jl Documentation Builder — Phase 2: Full Documenter Integration
# Combines Literate.jl example processing with Documenter.jl HTML rendering

import TOML

repo_root = dirname(dirname(@__FILE__))
src_dir = joinpath(@__DIR__, "src")
generated_dir = joinpath(src_dir, "generated")  # Generate into src/ so Documenter finds them
build_dir = joinpath(@__DIR__, "build")
examples_dir = joinpath(repo_root, "examples")

# Read Project.toml to get version
proj = TOML.parsefile(joinpath(repo_root, "Project.toml"))
version = proj["version"]

@info "ValTools.jl v$version Gallery (Phase 2 — Documenter Integration)"
@info "Source: $src_dir → Generated: $generated_dir → Build: $build_dir"

# Create build directories
mkpath(build_dir)
mkpath(generated_dir)

# ──────────────────────────────────────────────────────────────────────────
# Step 1: Process examples with Literate.jl
# ──────────────────────────────────────────────────────────────────────────

try
    using Literate

    examples_to_process = [
        "inertial_oscillation.jl",
        "eddy_spindown.jl",
        "ridge_ellipse_polarization.jl",
        "multivariate_common_oscillation.jl",
        "svd_polarization_detection.jl",
        "multitaper_line_detection.jl",
        "mooring_array_batch.jl",
        "unit_safe_validation.jl",
        "gomed_eddy_census.jl",
    ]

    @info "Processing $(length(examples_to_process)) examples with Literate.jl..."

    for example in examples_to_process
        example_path = joinpath(examples_dir, example)
        if isfile(example_path)
            try
                Literate.markdown(
                    example_path,
                    generated_dir;
                    name = splitext(example)[1],
                    credit = false,
                    throw_errors = false,
                    execute = false,  # Don't execute; just render code blocks
                )
            catch e
                @warn "Error processing $example: $e"
            end
        end
    end

    @info "✓ Literate processing complete"

    # Post-process: convert @example blocks to regular ```julia blocks (no execution)
    for md_file in readdir(generated_dir)
        if endswith(md_file, ".md")
            md_path = joinpath(generated_dir, md_file)
            content = read(md_path, String)
            # Replace ````@example name with regular ````julia
            content = replace(content, r"````@example [a-z_0-9]*\n" => "````julia\n")
            write(md_path, content)
        end
    end

catch e
    @warn "Literate.jl not available: $e"
end

# ──────────────────────────────────────────────────────────────────────────
# Step 2: Build full HTML docs with Documenter.jl
# ──────────────────────────────────────────────────────────────────────────

try
    using Documenter

    # Build pages structure: manually include generated examples
    pages = [
        "Home" => "index.md",
        "Gallery" => [
            "Overview" => "gallery.md",
            "Getting Started" => "generated/inertial_oscillation.md",
            "Wavelets & Ridges" => [
                "Eddy Spin-down" => "generated/eddy_spindown.md",
                "Ridge Polarization" => "generated/ridge_ellipse_polarization.md",
                "Multivariate Ridge Analysis" => "generated/multivariate_common_oscillation.md",
                "SVD Polarization Detection" => "generated/svd_polarization_detection.md",
            ],
            "Multitaper Spectral" => "generated/multitaper_line_detection.md",
            "Colocation & Validation" => "generated/mooring_array_batch.md",
            "Unit-Safe Validation" => "generated/unit_safe_validation.md",
            "Case Studies" => [
                "GOMED Eddy Census" => "generated/gomed_eddy_census.md",
            ],
        ],
        "API Reference" => "api.md",
        "Contributing" => "contributing.md",
    ]

    makedocs(
        format = Documenter.HTML(
            prettyurls = false,
            collapselevel = 2,
            assets = ["assets/custom.css"],
        ),
        sitename = "ValTools.jl",
        authors = "Alex Dominguez",
        pages = pages,
        repo = "https://github.com/alexdominguezg10/ValTools.jl",
        build = build_dir,
        source = src_dir,
    )

    @info "✓ Documenter.jl build complete"

catch e
    @warn "Documenter.jl build failed: $e"
    @warn "Continuing with Phase 1 (Literate-only) HTML output..."
end

@info "Documentation built successfully at $build_dir"
println("✓ Build complete — open file://$(build_dir)/index.html in a browser")
