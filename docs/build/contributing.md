# Contributing to ValTools.jl

We welcome contributions! Whether you're adding a new loader, fixing a bug, or writing an example, here's how to get started.

## Types of Contributions

### Examples & Gallery
- **New example scripts** for the gallery (each with synthetic data, figure, and verification)
- **Case studies** using real ocean data (GOMED eddies, GulfDrifters, mooring validations, etc.)
- **Roadmap stubs** for planned features (math sketches + placeholder API)

### Methods & Algorithms
- **New loaders** for observational data (gliders, buoys, satellite products, etc.)
- **Spectral analysis methods** (parametric spectral, transfer functions, spectrograms)
- **GPU kernels** (batched transforms, LIC, etc.)
- **Validation metrics** (new skill scores, error frameworks)

### Bug Fixes & Refactoring
- Performance improvements
- Type refinements
- Documentation clarifications

### Documentation
- API improvements (docstring clarity)
- Tutorial notebooks
- Methods papers & citations

## Contribution Workflow

### 1. Fork & Clone
```bash
git clone https://github.com/yourusername/ValTools.jl.git
cd ValTools.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

### 2. Create a Feature Branch
```bash
git checkout -b feature/your-feature-name
```

### 3. Make Changes

**For new examples:**
- Place the script in `examples/your_example.jl`
- Use Literate.jl comments (`# ## Section Title`, `#src` markers)
- Include:
  - 1-2 paragraphs of oceanographic context at the top
  - Synthetic data generation (keep it self-contained)
  - Clear variable names & comments
  - 1-2 figures following CairoMakie + professional styling (see `plotting_standards.md`)
  - Verification: an assertion or println() confirming the result (e.g., "Recovered frequency within 1%")
  - Optional: jLab/MATLAB reference in a comment
  
**For new methods:**
- Add source code to appropriate `src/Module/` subdirectory
- Write docstrings (reference Lilly's jLab documentation style: short + concrete examples)
- Add unit tests in `test/`
- Verify against MATLAB/Python where a reference exists
- Update `FLARE_TABLE_1.md` if filling a void space

**For new loaders:**
- Create in `src/Observations/`
- Include minimal example in the docstring
- If downloading test data, use DataDeps.jl or Artifacts.jl (never commit raw data)
- Add colocation tests (ensure lat/lon/depth match a known model point)

### 4. Write Tests
```julia
using Test, ValTools

@testset "your feature" begin
    # Test the core functionality
    @test ... == ...
    # Test edge cases
    @test_throws ErrorType func(bad_input)
end
```

Run the full test suite locally:
```bash
julia --project test/runtests.jl
```

### 5. Commit & Push
```bash
git add src/... test/... examples/...
git commit -m "Add feature: your clear description (fixes #issue-number if applicable)"
git push origin feature/your-feature-name
```

### 6. Open a Pull Request

- **Title**: concise (e.g., "Add Matérn spectral model")
- **Description**:
  - What does this add? (1-2 sentences)
  - Why is it needed? (ocean relevance, void space fill, etc.)
  - How was it verified? (test results, MATLAB cross-check, etc.)
  - Closes #issue-number (if applicable)
  - Roadmap entry (if this fills a Flare void space, link it)

## Example Contribution Checklist

### For a New Example Script
- [ ] Script lives in `examples/your_example.jl`
- [ ] Literate.jl formatted (section headers as `# ## Title`)
- [ ] Synthetic or bundled data (self-contained, no external downloads)
- [ ] Figure generated with CairoMakie (professional styling)
- [ ] Verification: println() or @assert confirming the result
- [ ] Passes `julia --project examples/your_example.jl` without errors
- [ ] Renders in Documenter (PR includes updated `docs/src/gallery.md` or auto-detected)

### For a New Method
- [ ] Source code in `src/Module/methodname.jl` with docstring
- [ ] Unit tests in `test/test_methodname.jl` (aim for >80% coverage)
- [ ] Cross-checked against MATLAB/Python (note in PR description)
- [ ] Type signatures are generic where appropriate (e.g., `T<:Real` not `Float64`)
- [ ] Passes full test suite: `julia --project test/runtests.jl`
- [ ] Updates `GALLERY_PLAN.md` or void-space tracker if this closes a gap

## Code Style

- **Naming**: `snake_case` for functions, `CamelCase` for types
- **Documentation**: docstrings follow jLab style (short, with example)
- **Comments**: only where intent is unclear; code is the primary documentation
- **Performance**: use type annotations, avoid global variables, test with `@time`
- **Plotting**: all plotting uses CairoMakie (via ValToolsCairoMakieExt), consistent with existing examples

## Gallery Roadmap Structure

The gallery is organized into sections (Getting Started, Wavelets, Loaders, etc.). Each section has:
- ✓ **Complete** examples (live demos with figures)
- ○ **Planned** examples (stubs with math sketches + API placeholders)
- **Roadmap** section (future capabilities not yet started)

When contributing, state clearly which category your addition fills:
```markdown
## Your Example

**Status**: ✓ Complete / ○ Planned / Roadmap  
**Category**: Wavelets & Ridges (or whichever applies)  
**Void Space** (if applicable): IDEA-017 or Flare Table 1 row 42  
```

## Publication & Citation

If your contribution becomes part of a methods paper or publication, we'll acknowledge you. If you have a paper to cite or reference, include it in the docstring or example comment.

## Community

- **Questions?** Open a GitHub Discussion or issue
- **Ideas?** Check `RESEARCH_IDEAS.md` for ongoing projects
- **Contact**: adomingu@cicese.edu.mx

Thanks for contributing to ValTools.jl!
