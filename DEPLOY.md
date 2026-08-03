# Deployment Guide: ValTools.jl Gallery

The ValTools.jl examples gallery is automatically built and deployed to GitHub Pages on every push to `main`.

## How it works

1. **Push to main** → GitHub Actions triggers
2. **Julia builds docs** via `julia docs/make.jl`
3. **Generated HTML** in `docs/build/` is committed back to the repo
4. **GitHub Pages** serves from `/docs/build/` → live at `https://alexdominguezg10.github.io/ValTools.jl/`

## Setup (one-time)

### 1. Enable GitHub Pages

Go to repo **Settings** → **Pages**:
- **Source**: Deploy from a branch
- **Branch**: `main`
- **Folder**: `/docs/build`
- Click Save

### 2. Allow CI to push

The `.github/workflows/docs.yml` already has the right permissions (`contents: write`). No additional setup needed.

### 3. That's it!

Every push to main will auto-build and deploy the gallery.

## Manual deployment

To build and commit docs locally:

```bash
# Build
julia docs/make.jl

# Commit (if changed)
git add docs/build/
git commit -m "Deploy gallery: [your message]"
git push origin main
```

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
