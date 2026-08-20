# CLAUDE.md

## Project

**medgemma-studio** — Streamlit app for analyzing medical text and images using Google MedGemma on Apple Silicon with MLX. Single-file app: `streamlit_app.py`.

## Commands

```bash
uv sync                                 # Install dependencies
uv run streamlit run streamlit_app.py   # Run the app
uv run ruff check .                      # Lint
uv run ruff format .                     # Format
uv run ty check                          # Type check
uv run pytest                            # Run tests
```

When working with Python, invoke the relevant `/astral:<skill>` (`/astral:uv`, `/astral:ty`, `/astral:ruff`).

## Architecture

`main()` loads the model once (`@st.cache_resource load_model()` → `mlx_vlm.load()` + `load_config()` → `(model, processor, config)`; model `mlx-community/medgemma-1.5-4b-it-8bit`) and renders a research-only safety disclaimer (`DISCLAIMER_TEXT`, a persistent `st.warning`) above four `st.tabs`, each delegating to a `@st.fragment`-decorated `render_*_tab` with its own keyed widgets:

- **Ask** (`render_ask_tab`) — text-only Q&A. `DEFAULT_INSTRUCTION_TEXT`.
- **Chest X-ray** (`render_cxr_tab`) — single image; two-image **comparison** (both in one prompt labeled "First/Second image:", previewed in `st.columns(2)`, `DEFAULT_INSTRUCTION_COMPARE`, 600-token budget); or anatomy **localization** (single image only — padded to square, `LOCALIZATION_INSTRUCTION` asks for `[y0,x0,y1,x1]` boxes normalized to `[0,1000]`, then parsed → scaled → drawn → cropped back). Comparison and localization are mutually exclusive.
- **Computed Tomography** (`render_ct_tab`) — DICOM multi-file upload → `load_ct_volume` (HU) → `window_ct_slice` (false-color RGB) → `build_messages` with `"SLICE n"` labels → `run_model`. `DEFAULT_INSTRUCTION_CT`.
- **Pathology (WSI)** (`render_wsi_tab`) — single slide (`.svs/.ndpi/.tif/.tiff`) + magnification `segmented_control` (5/10/20/40×) → `load_wsi_patches` (896px tissue patches) → `build_messages` with `"PATCH n"` labels → `run_model`. `DEFAULT_INSTRUCTION_WSI`.

CT and WSI share a 2000-token budget (2500 with thinking) and the same shape: preprocess inside an `st.status` (narrates phases, resolves `complete`/`error`) then stream generation in the main area below. Slice/patch counts are capped by `ram_aware_slice_cap` (multi-image inference is memory-heavy on unified memory).

**Pure helpers** (no Streamlit/model — unit-tested):
- `parse_response` — splits the `<unused94>`/`<unused95>` thinking trace from the answer
- `build_messages` — chat message list; one image placeholder per image, optional per-image label text
- `get_generation_params` — per-mode full instruction + `max_new_tokens` budget
- `pad_to_square`, `parse_boxes`, `scale_box`, `draw_boxes` — localization geometry (top-left pad; parse fenced/bare JSON boxes; scale `[0,1000]` → pixels; draw labeled boxes)
- `normalize_hu`, `window_ct_slice` — CT windowing; `CT_WINDOWS` (wide/soft-tissue/brain → R/G/B) is the model's **trained** CT format, so it is fixed, not user-tunable
- `subsample_indices` — uniform slice/patch indices, endpoints included
- `load_ct_volume` — reads per-slice DICOMs, sorts by InstanceNumber, `seek(0)`-rewinds each upload before `dcmread`, converts to HU via `apply_rescale`
- `ram_aware_slice_cap` — `(default, max)` slice/patch counts scaled to installed RAM (`(10,20)` on 32 GiB, hard max 64); RAM detection memoized via `_cached_total_ram_gib`
- WSI: `mag_from_mpp`, `effective_magnification`, `pick_level`, `patch_grid` (non-overlapping 896px tiles, partial edges dropped), `tissue_mask` (saturation proxy `max−min(RGB)` excludes glass), `tissue_patches` (keep ≥25% tissue), `mark_patches`, `load_wsi_patches` (OpenSlide loader; `read_region` takes a **level-0** location but a target-level size; raises `ValueError` for unreadable / too-small / no-tissue slides)

**UI helpers** (touch Streamlit):
- `load_uploaded_image` — `image.load()` forces the decode so invalid data fails here, not later at `st.image`
- `run_model` — streams via `mlx_vlm.stream_generate` through `st.write_stream`, returns the accumulated string; passes `REPETITION_PENALTY`/`REPETITION_CONTEXT_SIZE` unless `penalize_repetition=False` (the localization path opts out — the penalty truncates the JSON box list); `temperature=0` (deterministic)
- `render_thought` — thinking trace → expander, returns the answer
- `tab_settings` — per-tab system instruction + thinking toggle (independent session-state keys) in a collapsed "Model settings" expander

**Result persistence** — Each tab runs inference **only** inside its `Run` block, stores output + `is_thinking` + a `sig` of the run-defining inputs (prompt, `_file_sig`=name+size, localize/compare mode, slice/patch count, magnification) in `st.session_state["{ask,cxr,ct,wsi}_result"]`, and renders it **outside** the button gate via `fresh_result_or_hint(key, live_sig)` — returns the stored dict when the `sig` matches, else shows an `st.info` "Inputs changed…" hint and drops it (so a stale result never mismatches the visible inputs). `thinking` is deliberately excluded from `sig`. On a successful run each tab `st.rerun()`s, shedding the streamed raw copy for the clean persisted render.

## Claude Code hooks

`.claude/settings.json` (shared) runs the same gates automatically; guarded structurally **and** behaviorally (executes each command, asserts exit codes) by `TestHooksConfig`. All hooks under one matcher run **in parallel**, so the per-`.py`-edit cost is `max()`, not `sum()` — ~0.64s, of which `ty check` is ~99%:

- **`permissions.deny`** — `Read(.env)`, `Read(.env.local)`, `Read(.streamlit/secrets.toml)`. The **read** side, declarative: it also covers `cat`/`head`/`tail`/`sed` through `Bash` (but *not* an arbitrary subprocess that opens the file itself), which no `Edit`/`Write` hook can see. Two rules to keep in mind, both load-bearing: a `Read` deny rule **also blocks `Edit` and `Write`** on the same path (Claude Code ≥ 2.1.228), and **a deny rule carries no allowlist exceptions** — an `allow` rule cannot rescue a denied path, and neither can a `PreToolUse` hook returning allow. So the entries are **enumerated, never `Read(.env.*)`**: the wildcard would make `.env.example` uncreatable and silently turn the guard's carve-out into dead configuration. The trade is deliberate — a future `.env.production` is write-blocked by the guard but not read-blocked. Keep the **bare** forms too: they are gitignore-style and match at any depth, whereas `Read(/.env)` would narrow to the root file only.
- **PreToolUse** (`Edit`/`Write`) — blocks **writes** to `.env`/`.env.*` (except `.env.example`/`.sample`/`.template`), `.streamlit/secrets.toml`, `uv.lock`. Case-insensitive; matches relative as well as absolute paths; **fails closed** if `jq` is absent *or* the event has no parseable `file_path`. Accident-guard only (does not intercept `Bash` writes).
- **PostToolUse** (`Edit`/`Write`) — all three entries share a prologue that resolves a relative `file_path` against the project root and then **ignores anything outside `$CLAUDE_PROJECT_DIR`** (a scratch `.py` elsewhere is not ours: linting it blocks on unrelated violations, and arming the sentinel bills the ~21s suite for a file no test covers). Then, on `.py`: `ruff check --fix` + `ruff format` (silent) then a **blocking `ruff check` re-check** (exit 2 on whatever `--fix` can't repair — E501 on a long comment is the common one, and `ruff format` doesn't rescue it); separately `ty check` (whole project — scoping it to one file is *slower*, the cost is fixed startup). A third, separate entry writes a `.claude/.tests-needed` sentinel (gitignored) on any `.py`/`.toml`/`.claude/settings.json` edit — it stays separate **on purpose**, since folding it into the `ty` command would let that hook's `exit 2` short-circuit the `touch` and silently disarm the Stop gate.
- **Stop** — if the sentinel exists, runs `uv run pytest -q --maxfail=3 --tb=short` (~21s green); clears the sentinel on pass, blocks + feeds output back on fail. The flags are tuned to what survives `tail -n 40`, measured across both failure regimes: bare `-q` names 3/3 small failures but **0** root causes on a broad breakage (39 truncated `ERROR …` lines), while `-x` names the cause but only **1 of 3** failures. `--maxfail=3 --tb=short` is the only setting that wins in both. Skipped on turns with no pending sentinel — so a docs-only turn is free, though one *following an interrupted code turn* still pays the suite. `stop_hook_active` guards against a stop→fix loop, and because it stays true for **every** later stop in a turn, exactly one blocking run happens per user message (it resets on the next one, so a deferred failure is caught then).

**All of the above applies only to sessions started at the repo root** — Claude Code reads `.claude/settings.json` from the directory the session runs in, so `cd tests && claude` gets no deny rules and none of the five hooks. (`.claude/settings.local.json` is the exception: it is read from the git root, so it covers any subdirectory.) Personal overrides go there (gitignored).

## Continuous integration

`.github/workflows/ci.yml` runs the same four gates (`ruff check` · `ruff format --check` · `ty check` · `pytest`) via `uv sync --locked` on a `macos-15` (arm64 — matches the MLX target so `TestMlxVlmContract` exercises the shipped backend), on push to `main` / PR / `workflow_dispatch`. Least-privilege `contents: read` token, 15-min timeout, badge in `README.md`. A version bump must be lock-synced (`uv lock`), else `uv sync --locked` fails.

Four details are load-bearing rather than boilerplate, and a test now pins each — `TestCiWorkflow` for three of them, `TestAutoReleaseWorkflow` for the second (an audit found all four removable with a green suite):

- **`cancel-in-progress` exempts `main`** (`${{ github.ref != 'refs/heads/main' }}`). Superseding a stale PR run is right; superseding a main run is not — a cancelled run reports `cancelled`, not `failure`, so the commit would land on main having never passed the gates, *and* `tag-and-release.yml` (which publishes only off a **successful** CI run) would silently skip its release.
- **`name: CI` is an API.** `tag-and-release.yml`'s `workflow_run` matches this workflow's `name` value, **not** its path — renaming it un-gates the publisher invisibly. `TestAutoReleaseWorkflow.test_gates_on_ci_by_name_not_path` reads the name out of `ci.yml` so a rename fails a test instead.
- **`uv` itself is pinned** (`version: "0.12.5"` on `setup-uv`). `uv sync --locked` errors unless `uv.lock` is exactly what the *running* uv would produce, so an unpinned uv lets an upstream release turn a green PR red with no repo change — indistinguishable from a genuinely stale lock. Bump it alongside `uv lock`.
- **`cache-dependency-glob: uv.lock` is narrower than setup-uv's default**, which also keys on `**/pyproject.toml` — under the default, every ruff-rule or `[project.urls]` edit would evict an 86-package cache. `enable-cache` is deliberately *absent*: its `auto` default already resolves to on for a GitHub-hosted runner. `persist-credentials: false` on `checkout` keeps the token out of `.git/config`, since nothing in CI pushes and the steps after it execute ~100 third-party packages' code.

## Releases

Two workflows publish, and **the automatic one is the normal path**.

**`.github/workflows/tag-and-release.yml` (automatic).** Triggered by `workflow_run` on **CI's completion**, filtered to `branches: [main]`, not by `push: branches: [main]` — gating on CI's *conclusion* is what makes unattended publishing safe, since it buys the whole four-gate suite including the `uv sync --locked` step that goes red when a bump lands without `uv lock`. A `push:` trigger would race CI and publish in seconds. The `branches:` filter matters for a second reason: without it every *pull-request* CI run also starts a (skipped) run here, and a skipped run still queues in the shared publish group.

There are two entry paths and **both** end up CI-gated, by different means:

- **`workflow_run`** — the job `if:` requires the upstream run to be `success`, from a `push`, on `main`, **and** `head_repository.full_name == github.repository` (a fork's default branch is also called `main`, so the branch check alone is a trap).
- **`workflow_dispatch`** — the recovery lever. Its clause is **ANDed with `github.ref == 'refs/heads/main'`**; as a bare disjunct it let anyone with write access publish from any branch or tag, because on a dispatch `github.sha` is the tip of the **dispatched** ref, which both `head_sha || github.sha` fallbacks resolve to. It still cannot check a CI conclusion (there is no upstream run), so a **"Require a green CI run for this commit"** step re-imposes the invariant on both paths: the tested SHA must have a successful, `event=push` CI run, or the job fails. Without it, dispatching while main is red — precisely what the lever is for — would publish from a failing commit.

`TestAutoReleaseWorkflow` pins the job `if:` as one **whole normalised expression**, not as four substring checks: a substring guard is satisfied just as happily when the `&&`s are flipped to `||`, and says nothing about the dispatch clause. Mutation-tested — under the old substring form, both tightening *and deleting* the dispatch clause left the suite green.

The job checks out `workflow_run.head_sha` — on a `workflow_run` event `github.sha` is the **default-branch tip**, not the commit CI tested — then reads `version` from `pyproject.toml`, requires a plain `^[0-9]+\.[0-9]+\.[0-9]+$` (a `0.9.0rc1` left in place during WIP is skipped), and publishes.

The idempotency key is repository **state** — *"does a release already exist for `v$version`?"* — never a commit diff. `git diff HEAD^ HEAD -- pyproject.toml` breaks on squash merges, force pushes, several commits in one push, and job re-runs; a state check survives all four, so re-running the job is always a safe no-op. Publishing is a single `gh release create "$tag" --target "$SHA" --generate-notes`: `--target` **creates the tag and publishes in one API call**, so there is no window where the tag exists but the release doesn't. That is also why `--verify-tag` is absent here (it requires a pre-existing tag) — pinning `--target` to the exact tested SHA is the stronger guarantee. But `target_commitish` is documented as **"Unused if the Git tag already exists"**, so a tag left behind *without* a release (a hand-pushed tag whose `release.yml` run failed) would silently anchor the release at that tag's commit instead: the job therefore **refuses outright** if the tag already exists, pointing the operator at `release.yml`. Releases are **published, never `--draft`**: a draft is absent from the public releases API, so the README's shields.io release badge would freeze on the previous version, and a draft doesn't materialise the tag until someone clicks Publish.

**The constraint that forces this shape:** GitHub documents that *events triggered by the default `GITHUB_TOKEN` do not create new workflow runs* (only `workflow_dispatch` / `repository_dispatch` are excepted). A bot-pushed tag is therefore **invisible** to `release.yml`'s `on: push: tags` — so the obvious "tag here, let `release.yml` publish" design is dead without a standing PAT or GitHub App secret. Tagging and publishing in one job is what avoids introducing one. Don't "simplify" it back.

**`.github/workflows/release.yml` (manual backstop).** Still fires on a pushed `vX.Y.Z` tag, and still **verifies the tag matches `pyproject.toml`'s `version`** (a `v0.7.6` tag pushed while pyproject says `0.7.5` fails the job — the tag==version check lives at release time, not as a pytest that would fail between a bump and its tag), then runs `gh release create --generate-notes --verify-tag`. It is now **idempotent** (`gh release view` → skip), because the automatic path will usually have got there first and a hand-pushed tag for an already-published version should be a quiet no-op, not a red job. `github.ref_name` reaches the shell via `env`, never interpolated into a `run:` step.

Both publishers run on `ubuntu-latest` (no MLX — read a version, call `gh`) with a `contents: write` token and a 10-min timeout, and both declare the **same literal `concurrency` group, `publish-release`**. Groups are repository-wide, which is what makes the two paths mutually exclusive. In `tag-and-release.yml` it sits on the **job**, not the workflow: a workflow-level group is claimed when the *run* starts, before the `if:` is evaluated, so skipped no-op runs would queue in it — and GitHub cancels a previously **pending** run when a newer one queues, even under `cancel-in-progress: false`, which could silently drop a queued manual release. Guarded by `TestAutoReleaseWorkflow` / `TestReleaseWorkflow`, with `TestWorkflowsAreGuarded` as the reverse guard so a *new* workflow can't land unguarded. A release badge sits in `README.md`, and `pyproject.toml`'s `[project.urls]` `Changelog` points at the Releases page.

**To cut a release:** bump `version` in `pyproject.toml` → `uv lock` → commit → push to `main`. That's all: once CI is green the tag and the release appear. Recovery levers if the gate is missed: push anything else to main, `workflow_dispatch` `tag-and-release.yml`, or hand-push the tag (which takes the `release.yml` path). Two states go **silently quiet** — a non-`X.Y.Z` version, and a version whose release already exists — and a `::notice::` in the job log is the only signal. If a tag ruleset on `refs/tags/v*` is ever added, `github-actions[bot]` must be on its bypass list or the publish is rejected at `gh release create` time.

## Tests

- **`tests/test_streamlit_app.py`** — pure helpers (no Streamlit/model): `load_ct_volume` runs against real in-memory DICOMs; `load_wsi_patches` against a `_FakeSlide` mock. Plus real-asset guards that catch upstream/config drift: `TestMlxVlmContract` (introspects the real mlx-vlm API `run_model`/`load_model` depend on — the **only** guard that catches an mlx-vlm upgrade, since every other test mocks it), `TestThemeConfig` (`.streamlit/config.toml`), `TestHooksConfig` (`.claude/settings.json`), `TestCiWorkflow` (`.github/workflows/ci.yml`), `TestAppTestHarness` (pins that every `AppTest` in `tests/` is built by `_app_test()`, so none silently reverts to the flaky 3s default), `TestReleaseWorkflow` (`.github/workflows/release.yml` — the manual, tag-driven publisher), `TestAutoReleaseWorkflow` (`.github/workflows/tag-and-release.yml` — the CI-gated automatic publisher), `TestWorkflowsAreGuarded` (reverse guard: every file under `.github/workflows/` must be named by a guard class, so a new workflow can't land with zero coverage). The three workflow guards share a `_WorkflowGuard` base holding the properties **every** workflow must satisfy — no `${{ }}` in a `run:` step, no `continue-on-error`, no job-level `permissions:` **widening** the workflow token (job-level permissions *replace* the workflow default, so a top-level-only check misses them — narrowing to `permissions: {}` stays legal, since it compares scope ranks rather than demanding equality), and `persist-credentials: false` on every `checkout` (nothing here pushes with git; both publishers go through `gh` with `GH_TOKEN` from env). Also `TestClaudeMd` (this file — asserts every documented path, `tests/` module, and app-spine symbol stays current), `TestLicense` (the `LICENSE` file ↔ its `pyproject` SPDX declaration ↔ the README License + medical-use Disclaimer sections stay mutually consistent), and `TestDocsMatchSource` (cross-checks CLAUDE.md **and** README.md against the code for the model id, WSI extensions, ruff rule set, and license id — the extensions matched as a delimited token so a `.tif`-vs-`.tiff` prefix collision can't hide a gap), and `TestReadmeAssets` (every repo-relative README link/image resolves — the `docs/screenshot.webp` hero, the sample-data guide, and the LICENSE/workflow links — and the README keeps embedding that hero) — so neither doc can silently drift from the code.
- **`tests/test_app_ui.py`** — UI flow via `streamlit.testing.v1.AppTest` (built through `_app_test()`, warmed once by `_warm_streamlit_once`); asserts per-mode `num_images`/token budgets reach the mocked `stream_generate`, the mutual-exclusivity guards, result persistence vs staleness, and CT/WSI error `st.status` states.
- **`tests/dicom_helpers.py`** — shared in-memory DICOM builder `dicom_bytes`, imported by **both** test files (aliased `_dicom_bytes` in `test_streamlit_app.py`); a test support module, not a test file.
- Manual-test assets live in `samples/` (gitignored except `samples/README.md`); the suite builds its own in-memory fixtures.

## Screenshots

The README hero (`docs/screenshot.webp`) shows the **Chest X-ray** tab analyzing the sample radiograph (`samples/cxr/longitudinal_cxr_before.png` — see `samples/README.md`). Regenerate it with headless **Playwright**, run ephemerally (`uv run --with playwright …`) so it stays **out of the project deps** — it is not a dependency:

- Start the app, then drive the CXR tab: attach the sample via the `input[type="file"]`, ask a plain-analysis prompt (localization boxes are unreliable on this 4B model — don't use them for the hero), Run, and wait for the `Response`.
- **Force `color_scheme="dark"`** (headless Chromium defaults to light; the app follows `prefers-color-scheme`) and use a **viewport taller than the whole app** — Streamlit scrolls *inside* `section[data-testid="stMain"]`, so `full_page` otherwise captures one viewport band and clips the title/response.
- Hide the dev chrome (`stToolbar`/`stHeader`/fullscreen buttons), screenshot, then crop to the X-ray→`Response` region (element bounding boxes), downscale to ~1200px, and save as **WebP** with Pillow (~5× smaller than PNG for a photographic radiograph; GitHub renders WebP). The full-page intermediate (`docs/screenshot-full.png`) is a throwaway — gitignored.
- `TestReadmeAssets` guards that the hero stays embedded and every repo-relative README link resolves.

## Gotchas

- **AppTest re-execs the script each run** — patch `mlx_vlm.*` (and `openslide.OpenSlide`) at the **source**, not `streamlit_app`; select widgets **by key** (tabs render every widget, so position is ambiguous); chain multiple `.upload()` on one element before a single `.run()`.
- **`AppTest`'s 3s `default_timeout` is a cold-start trap** — a script run takes ~0.03s warm but 16–20s on the *first* run that renders model output in a freshly synced venv — Streamlit imports pandas/pyarrow lazily, and the first `st.write_stream` pulls in ~350 modules off a cold page cache (the base render alone is ~3s and never imports pandas, so the warmup has to click **Run**, not just `.run()`). A session-scoped `_warm_streamlit_once` fixture pays that once so the per-run `APP_RUN_TIMEOUT` can stay tight (the bound is **per test**, so a generous one multiplies across the file and would blow CI's job cap on a hang); the warmup itself gets `APP_WARMUP_TIMEOUT`. Build every AppTest via `_app_test()` — a bare `AppTest.from_file` passes warm and fails cold, so `TestAppTestHarness` (in `test_streamlit_app.py`, scanning every module under `tests/`) pins that no call site reverts to the default.
- **The success path `st.rerun()`s** — the CT/WSI success-path `st.status` is gone by the time AppTest captures the tree, so assert only the *error* state via `at.status[0].state`.
- **Streamlit API** — use `width="stretch"`, not deprecated `use_container_width`.

## Constraints

- **No multi-turn chat** — single Q&A per interaction.
- **Package management** — uv (`pyproject.toml` + `uv.lock`); no `requirements.txt`.
- **OpenSlide** — `openslide-python` + `openslide-bin` (native lib ships as an arm64/universal2 wheel; no Homebrew). Single-file WSI formats only; `.mrxs` (multi-file) excluded.
- **HF token** — optional, from `.env` via `python-dotenv`; the MLX repo is ungated, so a token only avoids download rate limits.
- **Theme** — clinical theme in `.streamlit/config.toml`: shared `[theme]` + `[theme.light]`/`[theme.dark]` palettes (defining both enables OS/browser auto-switch; a lone `[theme]` locks one mode). Keys validated against the installed Streamlit.
- **Linting** — ruff rule set `E`, `F`, `I`, `UP`, `B`, `SIM`, `C4`; see `[tool.ruff.lint]`.
- **License** — app source is Apache-2.0 (`LICENSE`, declared via `license`/`license-files` in `pyproject.toml`); the downloaded MedGemma weights are **not** redistributed here and are separately governed by Google's Health AI Developer Foundations terms. `README.md` carries a research-only, not-a-medical-device disclaimer.
