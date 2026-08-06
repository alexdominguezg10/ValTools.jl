# Shared plotting helper: automatic, resolution-aware font sizing for
# CairoMakie figures, so every plotting script in this repo derives its
# title/label/tick/legend sizes from the actual FINAL rendered pixel
# dimensions instead of a hand-picked constant that only happens to look
# right at one specific figsize/px_per_unit combination (see
# feedback_plotting_standards.md's "Concrete font-size floor" rule --
# this generalizes that one-off GOFLOW measurement into a reusable formula).
#
# Makie's `save(path, fig; px_per_unit=k)` rasterizes the WHOLE figure --
# including text -- at `k` times the figure's logical size, so
# final_px = figsize_logical .* px_per_unit for every element, fonts
# included. `auto_fontsizes` works backwards from a target FINAL pixel
# size (what actually determines legibility once the PNG is viewed/embedded)
# to the logical Makie fontsize to pass in.

"""
    auto_fontsizes(width_px, height_px; px_per_unit=1.0,
                    title_frac=0.026, label_frac=0.020,
                    tick_frac=0.016, legend_frac=0.016,
                    min_px=(title=30, label=24, tick=20, legend=20))

Compute `(title=, label=, tick=, legend=)` Makie font sizes, in the figure's
LOGICAL units, that will render at a legible size in the final PNG.

`width_px`/`height_px` are the figure's FINAL rendered pixel dimensions
(i.e. `figsize .* px_per_unit`, not the raw `figsize` passed to `Figure()`).
Sizes scale with the shorter dimension as a fraction (`*_frac`) of it, with
an absolute floor (`min_px`) enforced in final pixels so small figures don't
shrink text below a readable minimum -- a large empty-margin figure (like a
tightly-zoomed ellipse plot) should not get illegibly small ticks just
because its logical figsize is modest.
"""
function auto_fontsizes(width_px::Real, height_px::Real; px_per_unit::Real=1.0,
                          title_frac::Real=0.026, label_frac::Real=0.020,
                          tick_frac::Real=0.016, legend_frac::Real=0.016,
                          min_px::NamedTuple=(title=30, label=24, tick=20, legend=20))
    shorter = min(width_px, height_px)
    px(frac, floor_px) = max(shorter * frac, floor_px)
    to_logical(target_px) = target_px / px_per_unit
    return (title  = to_logical(px(title_frac,  min_px.title)),
            label  = to_logical(px(label_frac,  min_px.label)),
            tick   = to_logical(px(tick_frac,   min_px.tick)),
            legend = to_logical(px(legend_frac, min_px.legend)))
end
