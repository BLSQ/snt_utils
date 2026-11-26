# snt_pipeline_utils

**SNT Pipeline Utilities** – A collection of utility functions designed to be reused across SNT pipelines.

---

## Installation

### Install a specific version (recommended)
Pin to a release tag (see [Releases](https://github.com/BLSQ/snt_utils/releases)):

```bash
pip install git+https://github.com/BLSQ/snt_utils.git@v2.0.0
```

### Install the latest release
Installs from the `release-v2` branch, which always points to the latest v2 release:

```bash
pip install git+https://github.com/BLSQ/snt_utils.git@release-v2
```

v1 is still available from the `release-v1` branch or a `v1.x.y` tag.

### Usage

```python
from snt_lib.snt_pipeline_utils import load_configuration_snt_from_dataset, run_notebook
```

---

## Functions

All functions live in `snt_lib/snt_pipeline_utils.py` and assume an OpenHexa pipeline context.

### Scripts
| Function | Description | Returns |
|----------|-------------|---------|
| `sync_pipeline_scripts` | Copy a pipeline's `code/*.ipynb`, `reporting/*.ipynb` and `utils/*.r` from its own bundle to the workspace when the sha256 differs, archiving the replaced copy under `archive/self_sync/`; never touches `custom/` or country variants. | `list[Path]` |

### Notebooks & reports
| Function | Description | Returns |
|----------|-------------|---------|
| `run_notebook` | Execute a Jupyter notebook with Papermill (R kernel by default); runs a custom notebook or country variant instead when one applies (see `resolve_notebook`), with a `[WARNING]`. | `None` |
| `run_report_notebook` | Execute a report notebook and convert the output to HTML; same notebook choice as `run_notebook`. | `None` |
| `resolve_notebook` | Decide which file runs for a generic notebook: `custom/<name>` (with `use_custom_notebooks`), else `<stem>_<CC>`, else the generic one. | `tuple[Path, str]` |
| `get_notebook_overrides` | List the notebooks that will not run from the repository copy (path, sha256), for `{CC}_parameters.json`; `[ERROR]` if custom notebooks are on and none matches. | `list[dict]` |
| `generate_html_report` | Convert a notebook to HTML and register it as a run output. | `None` |
| `handle_rkernel_error_with_labels` | Parse labelled R-kernel errors and log them with the matching severity. | `None` |

### Configuration
| Function | Description | Returns |
|----------|-------------|---------|
| `load_configuration_snt_from_dataset` | Load `SNT_config.json` from the latest version of the `SNT_CONFIGURATION` dataset (`snt-configuration`); log and raise an `[ERROR]` if it is missing. | `tuple[dict, DatasetVersion]` |
| `validate_config` | Validate the configuration, reporting all errors together. | `None` |

### Datasets
| Function | Description | Returns |
|----------|-------------|---------|
| `get_file_from_dataset` | Download and load a file (parquet, CSV, GeoJSON, JSON) from the latest dataset version. | `DataFrame \| GeoDataFrame \| dict` |
| `dataset_file_exists` | Check if a file exists in the latest dataset version. | `bool` |
| `get_new_dataset_version` | Create a new dataset version, creating the dataset if needed. | `DatasetVersion` |
| `add_files_to_dataset` | Add files to a new dataset version. | `bool` |
| `check_outputs_generated` | Raise if any expected output was not written during the current run. | `None` |

### Database
| Function | Description | Returns |
|----------|-------------|---------|
| `push_data_to_db_table` | Write a DataFrame or file to a database table, replacing it if it exists. | `None` |

### Files & provenance
| Function | Description | Returns |
|----------|-------------|---------|
| `save_pipeline_parameters` | Save pipeline run parameters to JSON for traceability. | `Path` |
| `delete_raw_files` | Delete files matching a glob pattern in a directory. | `None` |

---

## Releasing

Releases are created automatically from the `release-v2` branch:

1. Merge `main` into `release-v2`.
2. Bump `version` in `pyproject.toml` on `release-v2` and push.

The [release workflow](.github/workflows/release.yml) then creates the `v<version>` tag and a GitHub release. Pushes without a version bump are skipped.

The full procedure, including backporting fixes to v1 and starting a new major version, is in
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md). AI coding agents follow the rules in [AGENTS.md](AGENTS.md).
