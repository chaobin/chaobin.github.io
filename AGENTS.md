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

## Windows & worktrees (per-creature workflow)
Each creature gets its own VS Code window = a git **worktree**, so experiment
work never touches the public `master` tree.

- **Where**: worktrees live in `~/Projects/garage/chaobin.github.io.wt/<name>`
  on branch `experiments/<name>`. Current set:
  - `nn-skin` → creature 003 (tiger), `chaobin.github.io.wt/nn-skin`
  - `skyline` → creature 002 (LED-facade skyline), `chaobin.github.io.wt/skyline`
  - `snake` → new creature, `chaobin.github.io.wt/snake` (master-based; `_dev`
    symlinked to the main repo's shared `_dev`)
  - `floating` → creature 004, `chaobin.github.io.wt/floating` (master-based;
    `_dev` symlinked to the shared `_dev`)
- **Create a new one** — ALWAYS base off `master` (a new creature starts from
  the clean public tree; never base off another experiment branch):
  ```bash
  git -C ~/Projects/garage/chaobin.github.io worktree add \
      -b experiments/<name> ~/Projects/garage/chaobin.github.io.wt/<name> master
  ```
  Other branches (e.g. `nn-skin`) are only ever read/copied for reference —
  never used as a worktree base.
- **Wire it as an agent window**:
  `node ~/Projects/tools/agent-bus/install.mjs ~/Projects/garage/chaobin.github.io.wt/<name>`
  → writes the **gitignored** `.vscode/mcp.json` with
  `AGENT_BUS_WINDOW_ID=<name>` (local `.vscode/` is ignored; see commit
  `adb918b`). Then open the folder, approve the `agent-bus` MCP server, and
  reload. The window then appears on the bus as `<name>` (kind `mcp` + `toast`).
- **`_dev/` in a master-based worktree** is not tracked on `master`. Either
  **symlink** to the main repo's shared `_dev` (gitignored; shared edits, not
  versioned on the branch — snake uses this), or **copy** the force-tracked
  dev state with `git archive experiments/nn-skin _dev | tar -x -C .`
  (independent copy, can be force-added to the branch for versioning).
- **Rules**: keep the main repo on `master` (public). Never push `_dev`.
  Run the leak check before committing anything under `creature/`. Agents in
  these worktrees poll channel `chaobin.github.io` at the start of every turn.

## Starting a new creature (one-shot recipe)

Every new creature is the same procedure, and it's scriptable end-to-end:

```bash
cd _dev && node new-creature.mjs <name> <NNN>   # e.g. node new-creature.mjs floating 004
```

`_dev/new-creature.mjs` does all of the following in order (keep this doc in
sync if you change it):

1. **Worktree** — always from `master` (never another experiment branch):
   `git -C ~/Projects/garage/chaobin.github.io worktree add -b experiments/<name> ~/Projects/garage/chaobin.github.io.wt/<name> master`
2. **Symlink `_dev`** into the worktree (shared readable source, untracked):
   `ln -s /Users/chaobintang/Projects/garage/chaobin.github.io/_dev ~/Projects/garage/chaobin.github.io.wt/<name>/_dev`
3. **Scaffold `_dev/creatures/NNN/`** from the `_dev/new-creature/` template
   (mirrors 001): `index.html`, `styles.css`, `values.csv`,
   `src/{liquid.js, creature.js, values.js}`, `AGENTS.md`. Build contract:
   `window.creature = { cfg, init, update, draw }` and
   `window.__liquid = { config, rebuild }` (build.mjs default order is
   `src/liquid.js` → `src/creature.js`; values map `creature`→`cfg`,
   `liquid`→`config`). build.mjs does NOT create the output dir — `mkdir -p
   creature/NNN` before building (only needed at publish).
4. **Wire as an agent window**:
   `node ~/Projects/tools/agent-bus/install.mjs ~/Projects/garage/chaobin.github.io.wt/<name>`
   → writes gitignored `.vscode/mcp.json` (`AGENT_BUS_WINDOW_ID=<name>`); open
   the folder, approve the `agent-bus` MCP server, reload.
5. **Verify + leak check**: `node --check` the `src/*.js` files; optionally test
   the masked build (`mkdir -p creature/NNN && cd _dev && node build.mjs NNN`)
   then delete the public `creature/NNN/` test artifact. `git status --short`
   must show nothing under `_dev/`; `git ls-files creature/` only masked files.
6. **Dev loop**: `cd _dev && npx vite --port 5173 --strictPort` →
   http://localhost:5173/creatures/NNN/

Register the creature in `_data/creatures.yml` and create the public
`creature/NNN/` folder only when publishing.

## Agent bus (cross-window messaging)
- A shared cross-window message bus is available as the MCP server **`agent-bus`** (docs: `/Users/chaobintang/Projects/tools/agent-bus/`).
- At the START of every turn, call `agent_bus_poll` on channel `chaobin.github.io` with `since` = the last id you saw, and act on any pending handoffs/questions.
- After meaningful work another window needs, `agent_bus_send` a short summary with `from: <this window's id>` and `to: <recipient window id>` (or `all`). This window's id = the workspace folder name unless overridden by `agentBus.windowId`.

## Milestone journal (agent milestones)
One-line milestone summaries from the worktree agents, **kept here by the main
`chaobin.github.io` window** for human review.

**Convention** — every worktree agent occasionally posts a **one-liner** of its
latest milestone on channel `chaobin.github.io`, marked as a milestone:
```bash
agent_bus_send { channel: 'chaobin.github.io', text: '<one line>',
                 from: '<window id>', to: 'all', type: 'milestone' }
```
The main window then journals each into this section as
`YYYY-MM-DD — <window id>: <one line>` (newest first).

**Entries**
- (none yet — first milestone check-in to come)

---

## Current work — creature 003 (the tiger) — 2026-08-19

**WHERE TO WORK**: the experimental branch is `experiments/nn-skin`, checked out
in the **worktree at `/Users/chaobintang/Projects/garage/chaobin.github.io.wt/nn-skin`**.
Open that folder (not the main repo) for creature-003 work. The main repo stays
on `master` (public GH Pages). If you work in the main repo instead, switch it
to `experiments/nn-skin` and prune the stale worktree
(`git worktree remove <path> --force` / `git worktree prune`).

**Branch/commit map** (all LOCAL, never push `_dev`):
- `experiments/nn-skin` @ `d4a9e1f` — current work; force-added `_dev/` (sources,
  models, skills, journal) on top of `master`(6ef4984).
- `checkpoint/creature003` @ `d9bba7f` — earlier checkpoint of the same work.
- `master` @ `6ef4984` — public site.
- `_dev/` is gitignored but force-tracked on the two local branches; if the
  working tree loses `_dev/creatures/*`, restore with:
  `git archive d4a9e1f _dev | tar -x -C .`

**Creature 003 — what exists & is VERIFIED (all in `_dev/creatures/003/`)**:
- Rig: procedural 24-joint skeleton, geodesic skinning + guards, FK/LBS,
  per-limb 2-bone IK, leg-support gravity physics.
- **Skinning (no tearing)**: `K = 8` top influences + `finalSmooth(2, 0.45)` on
  the skinned field (from the libigl BBW comparison); bind exact (row sums
  1.000000±4e-8). User-confirmed: the over-stretch is gone.
- **Physics**: `BEND_SLACK = 1.08` in `computeLegLens` (the walk's legs were
  locking straight — now they stay ~7% bent); settle bias keeps idle at static
  height.
- **Walk (recalibrated 2026-08-19)**: `legSwingSign` now compares paw **Y** vs
  shoulder (was Z) — previously 3 of 4 legs (RF, LH, RH) swung BACKWARD; now all
  four protract forward with symmetric reach 0.32. `ph` = LF 0 / RF π / LH 0.5π
  / RH 1.5π gives the lateral-sequence landing LH→LF→RH→RF at 0.13/0.38/0.63/0.88.
  Do NOT "swap the hind pair" — that was compensating for the swing-direction bug.
- **Gaits**: presets walk/trot/gallop + `emergent` mode (Fukuoka/Pearson —
  speed-driven blend + leg-loading feedback; Experiment C done; `SK.gaitStep`,
  `SK.setSpeed`, `SK.setLoads`). Public build keeps the presets; `emergent` is a
  dev UI option (gait dropdown + speed/load-gain sliders).
- **NN/skin experiment**: `_dev/experiments/nn-skin/` (libigl BBW vs geodesic).
  Run: `.venv/bin/python export… no — `node export.mjs` → `data.json`, then
  `.venv/bin/python bbw.py` and `.venv/bin/python stress.py` (venv is Python
  3.13; PyPI package is `libigl`, import name `igl`).

**Dev loop** (in the worktree or main `_dev`):
```
cd _dev && npx vite --port 5173 --strictPort   # http://localhost:5173/creatures/003/
```
- Hidden/background tab halts `requestAnimationFrame` — keep the tab visible.
- Edit `src/*.js` + `values.csv`, hot-reload; the gait-tune panel + joint editor
  (A = adjust, C = UI toggle) mutate `SKEL.gait`/joints live. Journal every
  fix in `_dev/creatures/003/AGENTS.md` ("Hiccups & lessons", entries 1–23).
- Never rebuild the public `creature/003/app.js` until publishing.

**Verification discipline**: after any rig/gait change, re-check in the browser
(e.g. via `window.SKEL`): bind exact (row sums = 1), no leg locking
(`shoulder→paw` < `legLens[n].rest`), swing direction (all paws move +Z during
swing), footfall order (landing trace LH→LF→RH→RF), no NaN, no console errors.

**Docs**: the two skills in `.github/skills/` are the source of truth:
`skeletonize-glb-physics` (rig/skin/IK/physics, incl. BEND_SLACK + libigl notes)
and `quadruped-gaits` (footfall tables, emergent gaits / Experiment C).
