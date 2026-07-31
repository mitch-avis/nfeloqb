# AGENTS.md for nfeloqb

## 0) Mission + Non-Negotiables

This repository maintains a weekly-updated `qb_elos.csv` for downstream NFL modeling.

- **Primary objective:** keep `qb_elos.csv` current and compatible with the historical 538-style
  schema consumed downstream.
- **Secondary outputs:** `Other Data/defensive_adjustments.csv`, `Other Data/weekly_qb_states.csv`,
  `Other Data/meta_data.csv`, and `package_meta.json` should remain consistent with a successful
  run.
- **Preserve the output contract.** Do not casually change `qb_elos.csv` columns, meanings, or merge
  logic without checking downstream usage in `nfl-predictor`.
- **Respect the historical baseline.** `Manual Data/original_elo_file.csv` is the historical source
  for older games; `EloConstructor` appends modeled rows on top of it.
- **Do not change the hardcoded `2023-02-12` cutover in**
  `nfeloqb/Resources/elo_file_constructor.py` unless you intentionally rework the historical merge
  boundary.
- **Local runs must keep working without Airtable.** `workflow.py` defaults
  `NFELOQB_DISABLE_AIRTABLE=1` for local execution.

## Weekly Workflow

- `workflow.py` is the normal entrypoint.
- `workflow.py run` respects skip logic.
- `workflow.py run_now` forces a refresh.
- `nfeloqb/nfeloqb.py` is the orchestration path:
  1. load `model_config.json` and `package_meta.json`
  2. create `AirtableWrapper` and evaluate skip logic
  3. `DataLoader` pulls `games`, `players`, and `player_stats` from `nfelodcm`
  4. `QBModel.run_model()` computes QB values and team adjustments
  5. `Elo.run()` computes Elo ratings with QB adjustments
  6. `EloConstructor.construct_elo_file()` writes `qb_elos.csv`
  7. `MetaConstructor` refreshes metadata output
- `package_meta.json` stores `last_updated` and `last_full_week` and drives skip behavior.
- Current downstream usage is manual: the refreshed `qb_elos.csv` is copied into
  `nfl-predictor/data/qb_elos.csv`.

## Project Shape

- `nfeloqb/Resources/data_loader.py`: data ingest, game flattening, team stat aggregation, and game
  metadata joins.
- `nfeloqb/Resources/qb_model.py`: the stateful QB model loop over weekly games.
- `nfeloqb/Resources/elo.py`: team Elo updates, HFA/rest adjustments, and QB Elo blending.
- `nfeloqb/Resources/elo_file_constructor.py`: merges historical 538-format data with new rows and
  next-week projections.
- `nfeloqb/Resources/airtable_wrapper.py`: optional Airtable sync for starters and QB metadata.
- `nfeloqb/Resources/meta_constructor.py`: builds player metadata mappings from elo names to GSIS
  IDs.
- `nfeloqb/DataModels/`: `QB`, `Team`, `GameContext`, `ModelConfig`, `ModelParam`, and small
  modeling utilities.
- `nfeloqb/Manual Data/`: curated inputs and historical baselines. These are not throwaway files.
- `Other Data/`: generated outputs. Expect updates after successful runs.
- `Development/`: comparison/debug utilities, not part of the weekly pipeline.
- `nfeloqb/Optimizer/` and `nfeloqb/feature_optimization.py`: offline parameter tuning, not the
  weekly path.
- `.github/workflows/run now.yml` and `.github/workflows/run_all.yml`: runtime automation only.
  There is currently no dedicated validation workflow.

## Command Execution Rules

This project has a local `.venv`. Use its tools explicitly.

- Python: `.venv/bin/python`
- pytest: `.venv/bin/python -m pytest`
- Ruff: `.venv/bin/ruff`
- Pyright: `.venv/bin/pyright`
- Ty: `.venv/bin/ty`
- uv: `uv`

Recommended commands:

- Weekly run with skip logic: `.venv/bin/python workflow.py run`
- Forced weekly run: `.venv/bin/python workflow.py run_now`
- Focused tests: `.venv/bin/python -m pytest tests/<file>.py -q`
- Full tests: `.venv/bin/python -m pytest`
- Format touched files: `.venv/bin/ruff format <paths>`
- Lint touched files: `.venv/bin/ruff check <paths>`
- Type check: `.venv/bin/pyright .` and `.venv/bin/ty check .`
- Dependency verification: `uv lock --check` and `uv sync --check --active`

Avoid `./update_requirements.sh` in automated agent sessions unless you intentionally want its
interactive prompts and activated-shell checks. For non-interactive work, prefer `uv lock --upgrade`
and `uv sync --active` directly.

## Data And Schema Constraints

- `qb_elos.csv` is the external contract. Preserve its 538-style shape unless the user explicitly
  asks to change it.
- `game_id` is a critical join key throughout the repo. External loaders may expose it as either a
  dataframe column or the index; normalize it before joins when touching loader code.
- This repo is pandas-first. Most modeling and ETL code expects pandas dataframes, not Polars.
- Use `pd.NA` rather than `numpy.nan` when assigning missing values into pandas `string` columns.
- Team alias normalization matters in join-heavy code. Keep legacy mappings like `OAK -> LV`, `SD ->
  LAC`, `STL -> LAR`, `WSH -> WAS`, and `JAC -> JAX` consistent when modifying merges.
- `MetaConstructor` warnings about unmatched historical 538 QB names are not automatically fatal.
  Investigate when counts change materially or duplicate GSIS IDs appear.

## Engineering Standards

- TDD is the default for code changes. Add a focused failing test before changing production code.
- Prefer small regression tests around data-loader joins, Elo formatting, and constructor edge
  cases.
- Preserve existing CSV output contracts unless the user explicitly asks to change them.
- Add type hints on new or modified public functions. Improve local clarity when touching legacy
  code, but do not churn unrelated modules just to modernize them.
- Add concise docstrings or comments only where the code path is non-obvious.
- Keep changes minimal and local. This codebase has several stateful modeling classes, so broad
  refactors are risky without new tests.
- If a workflow failure looks data-related, inspect loaded dataframe columns and index shape first.
  `nfelodcm` and `nflreadpy` schema drift is a realistic failure mode here.
- If you touch error handling, prefer surfacing actionable failures over returning `None` and
  letting a later step fail with a less specific crash.

## Validation Expectations

- Minimum for code changes on the weekly pipeline:
  - run the focused pytest slice for the touched behavior
  - rerun `.venv/bin/python workflow.py run_now` if the weekly path was touched
- When feasible, also run:
  - `.venv/bin/ruff check` on touched files
  - `.venv/bin/pyright .`
  - `.venv/bin/ty check .`
- There is currently no validation GitHub Action, so local validation matters.

## Known Repo Risks

- GitHub Actions runtime workflows still use Python 3.12 and `pip install -r requirements.txt`,
  while `pyproject.toml` targets Python 3.14 and `uv`. Treat the workflow files as stale operational
  automation, not the canonical development environment.
- Test coverage is narrow. Core behavior in `qb_model.py`, `elo.py`, and optimizer code has little
  or no direct unit coverage.
- Several files under `Manual Data/` are curated inputs. Do not overwrite them casually.
- A successful weekly run updates generated artifacts, including `qb_elos.csv`, `Other Data/*.csv`,
  and `package_meta.json`. Expect those files to become dirty after a run.

## Practical Guidance For Future Sessions

- Start from the concrete surface:
  - workflow failure: `workflow.py` then `nfeloqb/nfeloqb.py`
  - data issue: `nfeloqb/Resources/data_loader.py`
  - Elo output issue: `nfeloqb/Resources/elo.py` and `nfeloqb/Resources/elo_file_constructor.py`
  - starter or Airtable issue: `nfeloqb/Resources/airtable_wrapper.py`
  - metadata or missing-name issue: `nfeloqb/Resources/meta_constructor.py`
- For architecture questions, understand the weekly path before opening `Development/` or
  `Optimizer/`.
- Keep Airtable disabled unless the task explicitly involves starter synchronization or remote base
  updates.
- If aligning `nfeloqb` with `nfl-predictor`, favor shared standards like Ruff, Pyright, Ty, TDD,
  and explicit `.venv` commands without breaking the `qb_elos.csv` contract.
