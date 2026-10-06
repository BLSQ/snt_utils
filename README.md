# snt_pipeline_utils

**SNT Pipeline Utilities** – A collection of utility functions designed to be reused across SNT pipelines.

---

## Installation

### Install a specific version (recommended)
Pin to a release tag (see [Releases](https://github.com/BLSQ/snt_utils/releases)):

```bash
pip install git+https://github.com/BLSQ/snt_utils.git@v1.0.0
```

### Install the latest release
Installs from the `release-v1` branch, which always points to the latest v1 release:

```bash
pip install git+https://github.com/BLSQ/snt_utils.git@release-v1
```

### Usage

```python
from snt_lib.snt_pipeline_utils import load_configuration_snt, run_notebook
```

---

## Functions

All functions live in `snt_lib/snt_pipeline_utils.py` and assume an OpenHexa pipeline context.

### Scripts
| Function | Description | Returns |
|----------|-------------|---------|
| `pull_scripts_from_repository` | Pull the latest pipeline scripts from the SNT repository into the workspace. | `None` |
| `load_scripts_for_pipeline` | Clone the SNT repository and copy the requested scripts into the workspace. | `None` |

### Notebooks & reports
| Function | Description | Returns |
|----------|-------------|---------|
| `run_notebook` | Execute a Jupyter notebook with Papermill (R kernel by default). | `None` |
| `run_report_notebook` | Execute a report notebook and convert the output to HTML. | `None` |
| `generate_html_report` | Convert a notebook to HTML and register it as a run output. | `None` |
| `handle_rkernel_error_with_labels` | Parse labelled R-kernel errors and log them with the matching severity. | `None` |

### Configuration
| Function | Description | Returns |
|----------|-------------|---------|
| `load_configuration_snt` | Load the SNT configuration from a JSON file. | `dict` |
| `validate_config` | Validate the configuration, reporting all errors together. | `None` |

### Datasets
| Function | Description | Returns |
|----------|-------------|---------|
| `get_file_from_dataset` | Download and load a file (parquet, CSV, GeoJSON, JSON) from the latest dataset version. | `DataFrame \| GeoDataFrame \| dict` |
| `get_matching_filename_from_dataset_last_version` | List filenames in the latest dataset version matching a pattern. | `list[str]` |
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
| `copy_file` | Copy a file to a destination folder, creating it if needed. | `None` |
| `remove_all_files` | Remove all files from a folder (subfolders untouched). | `None` |
| `delete_raw_files` | Delete files matching a glob pattern in a directory. | `None` |

---

## Releasing

Releases are created automatically from the `release-v1` branch:

1. Merge `main` into `release-v1`.
2. Bump `version` in `pyproject.toml` on `release-v1` and push.

The [release workflow](.github/workflows/release.yml) then creates the `v<version>` tag and a GitHub release. Pushes without a version bump are skipped.
