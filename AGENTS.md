# AGENTS.md — repository conventions

This is a **Jekyll blog** published on GitHub Pages at
https://chaobin.github.io. Branch: `master`. Deploy: GitHub Actions
(`.github/workflows/pages.yml`). Local preview: Ruby 3.3 + `bundle exec
jekyll serve` (see repo memory / `_dev` docs for the full recipe).

## Layout
- `_posts/*.md` — blog posts (Jekyll front matter: `layout`, `title`,
  `description`, optional `tags`). Filenames `YYYY-MM-DD-slug.md`.
- `_layouts/`, `_includes/`, `stylesheets/`, `javascripts/`, `images/` —
  theme and assets. Footer credit lives in `_layouts/post.html` and
  `_layouts/post_mathjax.html`.
- `creature/` — interactive experiments ("creatures"), each in `creature/NNN/`
  and listed in `_data/creatures.yml`. `creature/index.html` is the listing
  page; `_includes/header.html` links to `/creature/`.

## Privacy discipline — read before touching `creature/`
- The **readable source** for the creatures lives **only** in the local,
  gitignored `_dev/` directory (on this machine). It must **never** be
  committed, pushed, or copied into the public tree.
- The public `creature/NNN/` folders must contain **only the masked,
  build-time artifacts** (e.g. `creature/001/app.js` + `index.html` +
  `styles.css`). Do not add source `.js`, `values.csv`, or dev docs there.
- Rebuild the masked bundle with `cd _dev && npm run build -- NNN` (e.g. `001`).
  Never commit an un-obfuscated or debug build as `app.js`.
- Before committing anything under `creature/`, run the leak check:
  ```bash
  git status --short          # nothing under _dev/
  git ls-files creature/      # only masked build files
  ```

## Publishing
Push to `master` — the Actions workflow builds and deploys automatically.
