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

## Publishing workflow
1. **Preview locally** — the site runs the official GH Pages stack (Jekyll
   3.10 / `github-pages` gem) on Ruby 3.3:
   ```bash
   export PATH="/opt/homebrew/opt/ruby@3.3/bin:$PATH"
   bundle exec jekyll serve --host 127.0.0.1 --port 4000
   ```
   (Auto-regeneration is unreliable in the VS Code sandbox — restart the
   server or run `bundle exec jekyll build` after edits. The full setup and
   all the workarounds that make this 2016 blog build on modern macOS are in
   the "Local Jekyll setup — the fix" section of `_dev/AGENTS.md`.)
2. **Add content**
   - Post → new `_posts/YYYY-MM-DD-slug.md` with front matter (`layout: post`,
     `title`, `description`).
   - Creature → add `_dev/creatures/NNN/` (readable source + `values.csv`),
     build with `cd _dev && npm run build -- NNN`, add the public
     `creature/NNN/` app files, and list it in `_data/creatures.yml`.
3. **Commit** — for anything under `creature/`, run the leak check first
   (see the Privacy discipline section above).
4. **Publish** — `git push origin master`; the Actions workflow
   (`.github/workflows/pages.yml`) builds and deploys automatically.
   Note: pushing needs macOS keychain access — if it hangs in the sandbox,
   run the push unsandboxed.

## Publishing
Push to `master` — the Actions workflow builds and deploys automatically.
