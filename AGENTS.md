# AGENTS.md

Instructions for AI coding agents working in this repository.

`snt_lib` is a library installed by other projects (the SNT pipelines running on **OpenHexa**). Anything published
from a release branch reaches them. The full development and release procedure is in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md): read it before touching branches, versions or releases, and follow it
exactly.

## Commands

```bash
pip install -e ".[dev]"                 # install in development mode, with test and lint tools

ruff check snt_lib/ tests/              # lint
ruff format snt_lib/ tests/             # format

pytest                                  # all tests
pytest tests/test_snt_pipeline_utils.py::test_function_name  # single test
```

## Development and release rules

### Must

- Start every change from a branch created from an up-to-date `main`, and target `main` with the pull request.
- Run `ruff check snt_lib/ tests/`, `ruff format --check snt_lib/ tests/` and `pytest` before declaring work done.
- Update README.md when a public function is added, removed, or changes signature.
- Treat removing or renaming a public function, or removing, renaming or making a parameter required, as a breaking
  change. Say so explicitly to the developer: it can only ship in a new major version.
- Use Conventional Commits for commit messages and PR titles (`type(scope): description`, `!` for breaking changes).

### Never

- Never change `version` in `pyproject.toml` on `main`. Versions are bumped only on `release-v*` branches.
- Never merge a `release-v*` branch into `main`.
- Never merge `main` into `release-v1`. v1 receives fixes only, by cherry-pick (see the backport section of
  docs/DEVELOPMENT.md).
- Never create, move or delete a git tag. Tags are created by the release workflow only.
- Never push to a `release-v*` branch, publish a release, or edit the release workflows without an explicit request
  from the developer.

## Architecture

Single-module Python package: all functions live in [snt_lib/snt_pipeline_utils.py](snt_lib/snt_pipeline_utils.py),
and `snt_lib/__init__.py` re-exports the module (`from . import snt_pipeline_utils`). Tests are in
[tests/test_snt_pipeline_utils.py](tests/test_snt_pipeline_utils.py), with fixtures in `tests/fixtures/`.

### Runtime environment

Functions assume an **OpenHexa pipeline context**: `openhexa.sdk.current_run` is used for logging throughout
(`log_info`, `log_warning`, `log_error`, `log_debug`), and `openhexa.sdk.workspace` provides access to datasets,
files and the workspace database URL. Outside an OpenHexa run (e.g. in tests) those objects must be mocked.

### Key functional areas

| Area | Functions |
|------|-----------|
| Script sync | `sync_pipeline_scripts` — copy a pipeline's R half (`code/*.ipynb`, `reporting/*.ipynb`, `utils/*.r`) from its own bundle to the workspace, archiving replaced copies under `archive/self_sync/` |
| Notebook execution | `run_notebook`, `run_report_notebook`, `resolve_notebook`, `get_notebook_overrides` — execute `.ipynb` files via Papermill (R kernel by default); run a `custom/` notebook (with `use_custom_notebooks`) or a country variant (`{stem}_{country_code}.ipynb`) instead of the generic one |
| Dataset I/O | `get_file_from_dataset`, `add_files_to_dataset`, `get_new_dataset_version`, `dataset_file_exists`, `check_outputs_generated` — read/write OpenHexa datasets; supports parquet, CSV, GeoJSON, JSON |
| Configuration | `load_configuration_snt_from_dataset`, `validate_config` — load `SNT_config.json` from the latest version of the `snt-configuration` dataset and validate it; required top-level keys are `SNT_CONFIG`, `SNT_DATASET_IDENTIFIERS`, `DHIS2_DATA_DEFINITIONS` |
| Database | `push_data_to_db_table` — write a DataFrame to a PostgreSQL table via SQLAlchemy (replaces existing table) |
| Provenance | `save_pipeline_parameters` — serialise pipeline run parameters to a JSON file for traceability |
| R error handling | `handle_rkernel_error_with_labels` — parse labelled error strings from Papermill R-kernel exceptions and map them to OpenHexa log severities |

## Code Style

- Line length: 120 characters (ruff enforced)
- Python 3.12+
- Ruff rules enabled: pyflakes, pycodestyle, isort, pydocstyle, pyupgrade, flake8-annotations, bugbear, pydoclint,
  and others (see `pyproject.toml`)
- **Docstrings:** Google-style. Every function/method docstring follows a standard structure — a brief one-line
  description, an `Args:` section briefly describing each parameter, and a `Returns:` section if the function returns
  a value.
