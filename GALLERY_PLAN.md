# ValTools.jl Examples Gallery — General Plan

## Goal
A browsable, visual gallery where every ValTools capability — current and planned — has a self-contained, runnable example with a figure. It doubles as documentation, regression testing, and the "educational materials" deliverable from the Flare roadmap.

## 1. Infrastructure (the skeleton)

**Documenter.jl + Literate.jl + DemoCards.jl** — the standard Julia stack for this exact purpose (JuliaImages, Makie use it):

- Each example lives as a plain `.jl` script with Literate-style comments (`# ## Section`, `#src` markers). One source file generates three outputs: gallery page with rendered figures, downloadable script, and optionally a Jupyter notebook.
- DemoCards renders the thumbnail-grid gallery page automatically from folder structure + a cover image per demo.
- Deployed to GitHub Pages via `deploydocs()` (repo is private, so local HTML build serves as fallback), with option for future public release.
- CI runs every example on push — examples become executable tests, so the gallery can never silently rot.

**Data strategy** (keeps repo light, per DATA_SSD rule):
- Default: synthetic data generated in-script (inertial oscillations, red noise, virtual eddies) — always runnable anywhere.
- Small bundled samples (< 1 MB) for reader/loader demos: tiny NetCDF slices of CROCO/Argo/NDBC etc., via lazy-download Artifacts or DataDeps — never committed raw into git.
- Flagship case studies (GOMED, GulfDrifters) run on pre-computed results shipped as small CSVs (already in `results/`).

## 2. Gallery sections — mirroring the capability map

| Section | Examples (existing ✓ / to write ○) |
|---|---|
| **Getting Started** | ○ 5-minute tour; ✓ `unit_safe_validation` (Unitful + typed pipeline) |
| **Types & Dispatch** | ○ TimeSeries* types tour; ○ typed `plot_*` dispatch showcase |
| **Model Readers** | ○ CROCO/ROMS/NEMO read + sigma-to-z; ○ GPU `sigma_to_z_gpu` speedup demo |
| **Observation Loaders** | ○ one compact "zoo" page (Argo, DUACS, SWOT, NDBC, RAFOS, IES, thermistor, glider, CloudDrift, CANEK…) + ○ GEM builder deep-dive |
| **Colocation** | ✓ `mooring_array_batch`; ○ trajectory colocation (drifter); ○ knockdown model + virtual mooring |
| **Validation Metrics** | ○ Taylor diagram + bootstrap CI; ○ metrics_by_group; ○ current ellipse metrics |
| **Multitaper Spectral** | ✓ `multitaper_line_detection` (F-test); ○ rotary spectrum + coherence with CIs; ○ 2-D wavenumber spectra + along-track SWOT |
| **Wavelets & Ridges** (10/10 complete — the crown jewels) | ✓ `inertial_oscillation`, ✓ `eddy_spindown`, ✓ `ridge_ellipse_polarization`, ✓ `multivariate_common_oscillation`, ✓ `svd_polarization_detection`; ○ N-D batched wavetrans; ○ wavelet significance (Monte Carlo); ○ instmom/jointmom |
| **GPU Acceleration** | ○ CPU-vs-GPU wavetrans benchmark (H200 numbers as static results); ○ LIC texture visualization |
| **Case Studies** (flagship) | ○ GOMED eddy census; ○ GulfDrifters significance survey; ○ a mooring validation end-to-end (model → colocate → spectra → Taylor) |
| **Roadmap** (future) | Stub cards for parametric spectral: Matérn, debiased Whittle, composite oMp, transfer functions, spectrograms, gapped spectra — each card = math sketch + planned API, flipping to live demos as they land |

The **Roadmap section** is the key device for "future capabilities": the gallery's structure itself advertises the Flare vision, and each void-space fill has a pre-reserved slot.

## 3. Per-example standard
Every demo follows one template:
- **Title + cover figure** (150×150 px thumbnail auto-generated from top plot)
- **Oceanographic context** (~1 paragraph: what question this answers)
- **Literate code** (Markdown cells + executable blocks)
- **Figure(s)** following your plotting standards (CairoMakie, professional styling, full-width)
- **Verification footer** (assertion against a known value — e.g. recovered inertial frequency within tolerance)
- **jLab/MATLAB cross-reference** where a port exists

## 4. Phasing

1. **Phase 0 — Scaffold** (~4–6 h): `docs/` with Documenter+Literate+DemoCards, CI workflow, one example migrated end-to-end to prove the pipeline.
2. **Phase 1 — Migrate** (~6–8 h): convert the 8 existing scripts to literate format with covers; gallery goes live with the wavelet section essentially complete.
3. **Phase 2 — Coverage** (~15–20 h): write the ○ examples so every exported capability area has ≥1 demo (loaders zoo, colocation, metrics, GPU).
4. **Phase 3 — Flagships + Roadmap** (~10–12 h): GOMED/GulfDrifters case studies, roadmap stub cards.
5. **Phase 4 — Ongoing rule**: every new capability PR must include its gallery example (enforced culturally, optionally by CI checklist).

**Total ≈ 35–45 h**, front-loaded so a respectable gallery exists after ~12 h (Phases 0–1). 

## 5. Deployment & Maintenance

- **Local rendering**: `julia --project=docs docs/make.jl` generates `docs/build/index.html`
- **GitHub Pages** (once public): `deploydocs()` in CI, live at GitHub Pages URL
- **Verification CI**: Each example is its own test job; failures block merge
- **Stale example policy**: Gallery examples are part of the public API; breaking changes to `main` require simultaneous example updates

---

## Strategic Alignment

**Flare Proposal Goal**: Unified cross-language signal analysis toolbox  
**ValTools.jl Contribution**: Fill Julia language gap + ocean domain layer  
**Gallery Role**: Living showcase of Flare Table 1 completeness + educational reference for ocean signal analysis

This gallery becomes the artifact for positioning ValTools.jl as "Flare signal analysis + oceanographic validation + GPU acceleration in Julia."
