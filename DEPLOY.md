# Deployment Guide: ValTools.jl Gallery

The ValTools.jl examples gallery is automatically built and deployed to GitHub Pages on every push to `main`, via the GitHub Actions Pages deployment (not a committed-branch build).

## How it works

1. **Push to main** → GitHub Actions triggers
2. **Julia builds docs** via `julia docs/make.jl` → `docs/build/` (not committed; gitignored)
3. **`actions/upload-pages-artifact`** uploads `docs/build/` as the Pages artifact
4. **`actions/deploy-pages`** publishes it → live at `https://alexdominguezg10.github.io/ValTools.jl/`

## Setup (one-time)

### 1. Enable GitHub Pages

Repo must be public (or on a plan supporting private-repo Pages) — GitHub Pages does not commit any HTML into git. Then:

Go to repo **Settings** → **Pages** → **Source**: GitHub Actions.

### 2. Workflow permissions

`.github/workflows/docs.yml` requests `pages: write` and `id-token: write` — required by `actions/deploy-pages`. No branch-push permission is needed anymore.

### 3. That's it!

Every push to main will auto-build and deploy the gallery; PRs get the build uploaded as a downloadable artifact instead (no deploy).

## Manual local preview

```bash
julia --project=docs docs/make.jl
open docs/build/index.html
```

`docs/build/` is gitignored — nothing here needs to be committed.

## Monitoring

- Check **Actions** tab to see build status
- Look for workflow runs named "Documentation"
- Builds take ~2-3 minutes (Julia startup + example processing)
- GitHub Pages propagates within ~1 minute

## Troubleshooting

### Docs not updating?
- Check Actions tab for build errors
- Ensure you pushed to `main` (not a branch)
- Clear GitHub Pages cache: Settings → Pages → Save again

### Build fails on Julia dependencies?
- Update `docs/Project.toml` with new deps
- CI will auto-install on next push

### Build takes too long?
- Most time is Julia's startup + Documenter processing
- Consider splitting examples into phases if >5 min builds become an issue

## Gallery structure

```
docs/
├── make.jl              # Build script (Literate → Documenter)
├── Project.toml         # Doc dependencies
├── src/                 # Source Markdown pages
│   ├── index.md
│   ├── gallery.md
│   ├── api.md
│   ├── contributing.md
│   └── generated/       # Auto-generated example pages (from Literate)
└── build/               # Built HTML (committed to repo, served by GitHub Pages)
    ├── index.html
    ├── gallery.html
    ├── generated/       # Rendered example HTML pages
    └── assets/          # CSS, JS, themes
```

## CI Configuration

See `.github/workflows/docs.yml`:
- Triggers: Push to main, PRs on main
- Only rebuilds if `docs/`, `examples/`, `src/`, or workflow itself changed
- Commits back to main (avoids infinite loops via `github.event_name == 'push'` check)
- PRs upload docs as artifact (for preview, not committed)

## Adding new examples

1. Create `examples/my_example.jl` (Literate format)
2. Update `docs/make.jl` to include it in the build
3. Push to main → CI auto-rebuilds → gallery updates within 1-2 min

Done!
