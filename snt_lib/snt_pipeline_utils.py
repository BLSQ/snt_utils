import hashlib
import json
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any

import geopandas as gpd
import pandas as pd
import polars as pl
import papermill as pm
import requests
from nbclient.exceptions import CellTimeoutError
from openhexa.sdk import current_run, workspace
from openhexa.sdk.datasets.dataset import DatasetVersion
from papermill.exceptions import PapermillExecutionError
from papermill.translators import RTranslator, papermill_translators
from sqlalchemy import create_engine

# Papermill only registers its R translator under the language key "R" (capital), not under
# the IRkernel kernel name "ir" that run_notebook/run_report_notebook default to. Without this,
# execute_notebook raises PapermillException("No parameter translator functions specified for
# kernel 'ir' or language 'r'") for any notebook whose language_info.name is lowercase "r".
papermill_translators.register("ir", RTranslator())


# Folders and file patterns of a pipeline's R half, synced from its bundle by sync_pipeline_scripts.
SYNCED_SCRIPT_PATTERNS = (("code", "*.ipynb"), ("reporting", "*.ipynb"), ("utils", "*.r"))
# Folder, beside code/ and reporting/, holding hand-edited notebooks (never synced).
CUSTOM_NOTEBOOKS_FOLDER = "custom"


def _sha256(path: Path) -> str:
    """Return the sha256 hex digest of a file's raw bytes.

    Args:
        path (Path): File to hash.

    Returns:
        str: The hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _list_bundle_scripts(bundle_path: Path) -> list[Path]:
    """List the R-half files of a pipeline bundle, as matched by `SYNCED_SCRIPT_PATTERNS`.

    Args:
        bundle_path (Path): Folder of the pipeline bundle.

    Returns:
        list[Path]: The matching files, sorted.
    """
    sources = sorted(
        source
        for folder, pattern in SYNCED_SCRIPT_PATTERNS
        for source in (bundle_path / folder).glob(pattern)
        if source.is_file()
    )
    if not sources:
        raise FileNotFoundError(f"No notebook or R utils file found in the pipeline bundle {bundle_path}")
    return sources


def _sync_script(source: Path, target_root: Path, archive_dir: Path, rel_path: Path) -> bool:
    """Copy one bundle file to the workspace unless the copy there is identical; archive what it replaces.

    Args:
        source (Path): File in the bundle.
        target_root (Path): The pipeline's folder in the workspace.
        archive_dir (Path): Folder receiving the replaced copy, under the same relative path.
        rel_path (Path): Path of the file relative to the bundle and to `target_root`.

    Returns:
        bool: True if the file was written, False if the workspace copy was already identical.
    """
    target = target_root / rel_path
    if target.exists():
        if _sha256(target) == _sha256(source):
            return False
        backup = archive_dir / rel_path
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(target, backup)
        current_run.log_warning(
            f"[WARNING] {target} differed from this pipeline version's copy: moved to {backup} and "
            f"replaced. To run an edited notebook, put it in {target_root / CUSTOM_NOTEBOOKS_FOLDER} "
            "and switch on 'use custom notebooks'."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return True


def sync_pipeline_scripts(pipeline_name: str, bundle_path: Path, workspace_root: Path | None = None) -> list[Path]:
    """Copy a pipeline's R half from its bundle to the workspace, archiving any copy it replaces.

    Every `code/*.ipynb`, `reporting/*.ipynb` and `utils/*.r` in the bundle (the folder holding the
    running `pipeline.py`) is copied to `pipelines/<pipeline_name>/` in the workspace when the workspace
    copy is missing or its sha256 differs. A replaced copy is first moved to
    `archive/self_sync/<UTC stamp>/pipelines/<pipeline_name>/<path>`, with a `[WARNING]` naming it.
    Nothing is deleted, and `custom/`, country variants (`*_<CC>.ipynb`) and any other file are never
    touched. On failure, logs an `[ERROR]` and raises.

    Args:
        pipeline_name (str): Name of the pipeline, also its folder under `pipelines/`.
        bundle_path (Path): Folder of the running pipeline's bundle, e.g. `Path(__file__).parent`.
        workspace_root (Path | None): Workspace files root. Defaults to None (`workspace.files_path`).

    Returns:
        list[Path]: The workspace paths written (copied or replaced).
    """
    root = workspace_root if workspace_root is not None else Path(workspace.files_path)
    target_root = root / "pipelines" / pipeline_name
    stamp = datetime.now(UTC).strftime("%Y-%m-%d_%H%M%S")
    archive_dir = root / "archive" / "self_sync" / stamp / "pipelines" / pipeline_name

    try:
        sources = _list_bundle_scripts(bundle_path)
        written = [
            target_root / source.relative_to(bundle_path)
            for source in sources
            if _sync_script(source, target_root, archive_dir, source.relative_to(bundle_path))
        ]
    except Exception as e:
        current_run.log_error(f"[ERROR] Could not sync the pipeline scripts of {pipeline_name}: {e}")
        raise

    current_run.log_info(
        f"Pipeline scripts synced from the pipeline version: {len(written)} written, "
        f"{len(sources) - len(written)} unchanged."
    )
    return written


def resolve_notebook(
    nb_path: Path, country_code: str | None = None, use_custom_notebooks: bool = False
) -> tuple[Path, str]:
    """Decide which file runs for a pipeline's generic notebook.

    With `use_custom_notebooks`: `custom/<name>` beside the notebook's folder if it exists, else the
    country variant, else the generic notebook. Without it: the country variant
    (`{stem}_{country_code}{suffix}` in the same folder) if it exists, else the generic notebook.

    Args:
        nb_path (Path): Generic notebook path, e.g. `<workspace>/pipelines/<name>/code/<name>.ipynb`.
        country_code (str | None): Country code for selecting a country variant. Defaults to None.
        use_custom_notebooks (bool): Whether a notebook in `custom/` replaces the generic one.
            Defaults to False.

    Returns:
        tuple[Path, str]: The path to execute and its source: "custom", "country_variant" or "repo".
    """
    if use_custom_notebooks:
        custom_path = nb_path.parent.parent / CUSTOM_NOTEBOOKS_FOLDER / nb_path.name
        if custom_path.exists():
            return custom_path, "custom"
    if country_code:
        country_specific_path = nb_path.with_name(f"{nb_path.stem}_{country_code}{nb_path.suffix}")
        if country_specific_path.exists():
            return country_specific_path, "country_variant"
    return nb_path, "repo"


def get_notebook_overrides(
    nb_paths: list[Path], country_code: str | None = None, use_custom_notebooks: bool = False
) -> list[dict[str, str]]:
    """List the pipeline notebooks that will not run from the repository copy, for provenance.

    Call it once the country code is known and before any notebook runs, and pass the result to
    `save_pipeline_parameters(extra_metadata={"NOTEBOOK_OVERRIDES": ...})`. With `use_custom_notebooks`
    and no custom notebook for any of `nb_paths`, logs an `[ERROR]` and raises.

    Args:
        nb_paths (list[Path]): Every generic notebook the pipeline may run (code and reporting).
        country_code (str | None): Country code for selecting a country variant. Defaults to None.
        use_custom_notebooks (bool): Whether notebooks in `custom/` replace the generic ones.
            Defaults to False.

    Returns:
        list[dict[str, str]]: One entry per overridden notebook, with keys `NOTEBOOK` (generic file
            name), `SOURCE` ("custom" or "country_variant"), `PATH` and `SHA256`. Empty when every
            notebook is the repository copy.
    """
    overrides = []
    for nb_path in nb_paths:
        path, source = resolve_notebook(nb_path, country_code, use_custom_notebooks)
        if source != "repo":
            overrides.append({"NOTEBOOK": nb_path.name, "SOURCE": source, "PATH": str(path), "SHA256": _sha256(path)})

    if use_custom_notebooks and not any(o["SOURCE"] == "custom" for o in overrides):
        custom_folder = nb_paths[0].parent.parent / CUSTOM_NOTEBOOKS_FOLDER if nb_paths else CUSTOM_NOTEBOOKS_FOLDER
        names = ", ".join(nb_path.name for nb_path in nb_paths)
        message = (
            f"[ERROR] 'Use custom notebooks' is on, but {custom_folder} holds none of this pipeline's "
            f"notebooks ({names}). Put the edited notebook there under the same file name, or switch it off."
        )
        current_run.log_error(message)
        raise FileNotFoundError(message)

    return overrides


def _log_notebook_override(nb_path: Path, path: Path, source: str) -> None:
    """Log a [WARNING] when a notebook other than the repository copy runs.

    Args:
        nb_path (Path): Generic notebook path.
        path (Path): Path actually executed.
        source (str): Source returned by `resolve_notebook`.
    """
    if source == "custom":
        current_run.log_warning(f"[WARNING] Running the custom notebook {path} instead of {nb_path.name}.")
    elif source == "country_variant":
        current_run.log_warning(f"[WARNING] Running the country-specific notebook {path} instead of {nb_path.name}.")


def run_notebook(
    nb_path: Path,
    out_nb_path: Path,
    parameters: dict,
    error_label_severity_map: dict | None = None,
    kernel_name: str = "ir",
    country_code: str | None = None,
    use_custom_notebooks: bool = False,
):
    """Execute a Jupyter notebook using Papermill.

    The file executed is chosen by `resolve_notebook`: with `use_custom_notebooks`, `custom/<name>` beside
    the notebook's folder if it exists; otherwise, if `country_code` is provided and a notebook named
    {stem}_{country_code}{suffix} exists in the same folder as `nb_path` (e.g. pipeline_NER.ipynb for
    country_code="NER"), that one; otherwise `nb_path`. Any file other than `nb_path` is logged as a
    `[WARNING]`, and a custom notebook's output is named `{stem}_CUSTOM_OUTPUT_{timestamp}`.

    Args:
        nb_path (Path): Path to the default notebook to execute.
        out_nb_path (Path): Directory where the output notebook will be saved.
        parameters (dict): Parameters passed to the notebook.
        error_label_severity_map (dict | None): Map of error labels to severity
            (e.g. {"[ERROR]": "error", "[WARNING]": "warning"}). Defaults to None.
        kernel_name (str): Jupyter kernel name. Defaults to "ir" (R).
        country_code (str | None): Country code for selecting a country-specific notebook (e.g. "NER").
            Defaults to None.
        use_custom_notebooks (bool): Whether a notebook in `custom/` replaces `nb_path`. Defaults to False.
    """
    if error_label_severity_map is None:
        error_label_severity_map = {}

    nb_to_execute, source = resolve_notebook(nb_path, country_code, use_custom_notebooks)
    _log_notebook_override(nb_path, nb_to_execute, source)

    current_run.log_info(f"Executing notebook: {nb_to_execute}")
    file_stem = f"{nb_to_execute.stem}_CUSTOM" if source == "custom" else nb_to_execute.stem
    extension = nb_to_execute.suffix
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_nb_full_path = out_nb_path / f"{file_stem}_OUTPUT_{execution_timestamp}{extension}"
    out_nb_path.mkdir(parents=True, exist_ok=True)

    try:
        pm.execute_notebook(
            input_path=nb_to_execute,
            output_path=out_nb_full_path,
            parameters=parameters,
            kernel_name=kernel_name,
            request_save_on_cell_execute=False,
            progress_bar=False,
        )
    except PapermillExecutionError as e:
        handle_rkernel_error_with_labels(e, error_label_severity_map)
    except Exception as e:
        raise RuntimeError(f"Error executing notebook {nb_to_execute}") from e


def run_report_notebook(
    nb_file: Path,
    nb_output_path: Path,
    error_label_severity_map: dict | None = None,
    kernel_name: str = "ir",
    ready: bool = True,
    country_code: str | None = None,
    use_custom_notebooks: bool = False,
) -> None:
    """Execute a report notebook using Papermill and convert the output to HTML.

    The HTML report is not generated if the notebook raised a labelled warning. The file executed is
    chosen as in `run_notebook` (custom, then country variant, then `nb_file`).

    Args:
        nb_file (Path): Full file path to the notebook.
        nb_output_path (Path): Directory where the output notebook will be saved.
        error_label_severity_map (dict | None): Map of error labels to severity ('warning' or 'error'),
            e.g. {'LABEL': 'error', 'ANOTHER_LABEL': 'warning'}. Defaults to None.
        kernel_name (str): Jupyter kernel name. Defaults to "ir" (R).
        ready (bool): Whether the notebook should be executed (can be used as an OpenHexa @task signal).
            Defaults to True.
        country_code (str | None): Country code for selecting a country-specific notebook (e.g. "NER").
            Defaults to None.
        use_custom_notebooks (bool): Whether a notebook in `custom/` replaces `nb_file`. Defaults to False.
    """
    if not ready:
        current_run.log_info("Reporting execution skipped.")
        return

    if error_label_severity_map is None:
        error_label_severity_map = {}

    nb_to_execute, source = resolve_notebook(nb_file, country_code, use_custom_notebooks)
    _log_notebook_override(nb_file, nb_to_execute, source)

    current_run.log_info(f"Executing report notebook: {nb_to_execute}")
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_stem = f"{nb_to_execute.stem}_CUSTOM" if source == "custom" else nb_to_execute.stem
    nb_output_full_path = nb_output_path / f"{file_stem}_OUTPUT_{execution_timestamp}.ipynb"
    nb_output_path.mkdir(parents=True, exist_ok=True)
    warning_raised = False
    try:
        pm.execute_notebook(
            input_path=nb_to_execute,
            output_path=nb_output_full_path,
            kernel_name=kernel_name,
            request_save_on_cell_execute=False,
            progress_bar=False,
        )
    except CellTimeoutError as e:
        raise CellTimeoutError(f"Notebook execution timed out: {e}") from e
    except PapermillExecutionError as e:
        handle_rkernel_error_with_labels(e, error_label_severity_map)  # for labeled R kernel errors
        warning_raised = True
    except Exception as e:
        raise RuntimeError(f"Error executing notebook {nb_to_execute}") from e

    if not warning_raised:
        generate_html_report(nb_output_full_path)


def generate_html_report(output_notebook_path: Path, out_format: str = "html") -> None:
    """Generate an HTML report from a Jupyter notebook and register it as a run output.

    Args:
        output_notebook_path (Path): Path to the executed notebook file.
        out_format (str): nbconvert output format. Defaults to "html".

    Raises:
        RuntimeError: If the path is not an existing .ipynb file.
        CalledProcessError: If nbconvert fails.
    """
    if not output_notebook_path.is_file() or output_notebook_path.suffix.lower() != ".ipynb":
        raise RuntimeError(f"Invalid notebook path: {output_notebook_path}")

    report_path = output_notebook_path.with_suffix(".html")
    current_run.log_info(f"Generating HTML report {report_path}")
    cmd = [
        "jupyter",
        "nbconvert",
        f"--to={out_format}",
        "--no-input",
        str(output_notebook_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except CalledProcessError as e:
        raise CalledProcessError(e.returncode, e.cmd, output=e.output, stderr=e.stderr) from e

    current_run.add_file_output(report_path.as_posix())


def handle_rkernel_error_with_labels(error: Exception, error_labels: dict | None = None):
    """Handle errors from the R kernel and log them with appropriate labels.

    Error severity levels handled:
    - warning: Logs as a warning message.
    - error: Raises a RuntimeError with the message (and details, if any).
    - any other severity: Raises a RuntimeError flagging the unknown severity.
    Errors not matching any label are re-raised as RuntimeError.
    The optional label [ERROR DETAILS] at the end of the message carries additional details, e.g.:
    "Error: [LABEL] Some error message to the user [ERROR DETAILS] Additional error details here."

    Args:
        error (Exception): The error object raised by the R kernel.
        error_labels (dict | None): Map of error labels to severity levels ('warning' or 'error'),
            e.g. {'LABEL': 'error', 'ANOTHER_LABEL': 'warning'}. Defaults to None.

    Raises:
        RuntimeError: If the matched label has 'error' (or unknown) severity, or no label matches.
    """
    if error_labels is None:
        error_labels = {}

    error_msg = getattr(error, "evalue", str(error))
    current_run.log_debug(f"Error message captured from R: {error_msg}")
    matched = False

    for label, severity in error_labels.items():
        pattern = rf".*?{re.escape(label)}\s*(.*?)(?:\s*\[ERROR DETAILS\]\s*(.*))?$"
        match = re.search(pattern, error_msg, flags=re.DOTALL | re.IGNORECASE)
        if match:
            message_main = match.group(1).strip()
            message_details = match.group(2).strip() if match.group(2) else ""
            matched = True
            if severity == "warning":
                current_run.log_debug(f"Warning catched: {message_main}")
                current_run.log_warning(f"{message_main}")
            elif severity == "error":
                current_run.log_debug(f"Error catched: {message_main}")
                raise RuntimeError(f"{message_main} {message_details}")
            else:
                raise RuntimeError(f"{label} {message_main}. Unknown severity '{severity}'")
            break

    if not matched:
        raise RuntimeError(str(error))


SNT_CONFIG_DATASET_ID = "snt-configuration"
SNT_CONFIG_FILENAME = "SNT_config.json"
SNT_CONFIG_HOWTO_URL = (
    "https://github.com/BLSQ/snt-stratification/blob/main/contracts/snt_config/README.md#snt-configuration-dataset"
)


def load_configuration_snt_from_dataset(dataset_id: str = SNT_CONFIG_DATASET_ID) -> tuple[dict, DatasetVersion]:
    """Load the SNT configuration from the latest version of the SNT_CONFIGURATION dataset.

    Reads `SNT_config.json` from the latest version of the dataset, and logs the version's name and id.
    On any failure, logs an `[ERROR]` that says what is missing and how to provide the file, then raises.

    Args:
        dataset_id (str): Slug of the dataset. Defaults to "snt-configuration", fixed by convention.

    Returns:
        tuple[dict, DatasetVersion]: The configuration, and the dataset version it was read from. Record
            the version's `name` and `id` in `{CC}_parameters.json` through `save_pipeline_parameters`.

    Raises:
        ValueError: If the dataset, its latest version or the file is missing, the download fails, or the
            file is not valid JSON.
    """
    source = f"dataset SNT_CONFIGURATION ({dataset_id})"
    advice = f"Provide a valid {SNT_CONFIG_FILENAME} file. See {SNT_CONFIG_HOWTO_URL}"

    def fail(reason: str, cause: Exception | None = None) -> None:
        message = f"[ERROR] No usable {SNT_CONFIG_FILENAME} in the latest version of {source}: {reason}. {advice}"
        current_run.log_error(message)
        raise ValueError(message) from cause

    try:
        dataset = workspace.get_dataset(dataset_id)
    except ValueError as e:  # the SDK raises when the dataset does not exist
        fail("the dataset does not exist", e)
    if not dataset:  # older SDKs return None instead
        fail("the dataset does not exist")

    version = dataset.latest_version
    if not version:
        fail("the dataset has no version")

    try:
        dataset_file = version.get_file(SNT_CONFIG_FILENAME)
    except FileExistsError as e:  # the SDK raises this when the file is not in the version
        fail(f"version '{version.name}' has no file {SNT_CONFIG_FILENAME}", e)
    if not dataset_file:
        fail(f"version '{version.name}' has no file {SNT_CONFIG_FILENAME}")

    response = requests.get(dataset_file.download_url)
    if response.status_code != 200:
        fail(f"downloading it from version '{version.name}' failed with HTTP {response.status_code}")

    try:
        config_json = json.loads(response.content)
    except json.JSONDecodeError as e:
        fail(f"the file in version '{version.name}' is not valid JSON ({e})", e)

    current_run.log_info(f"SNT configuration loaded from {source}, version '{version.name}' (id {version.id}).")
    return config_json, version


def validate_config(config: dict) -> None:
    """Validate that the critical configuration values are set properly.

    All validation errors are collected and reported together.

    Args:
        config (dict): The SNT configuration, as returned by `load_configuration_snt_from_dataset`.

    Raises:
        KeyError: If a required top-level key is missing.
        ValueError: If any required value is missing, empty or malformed.
    """
    missing_top_level = [
        k for k in ("SNT_CONFIG", "SNT_DATASET_IDENTIFIERS", "DHIS2_DATA_DEFINITIONS") if k not in config
    ]
    if missing_top_level:
        raise KeyError(f"Missing top-level key(s) in config: {', '.join(missing_top_level)}")

    snt_config = config["SNT_CONFIG"]
    dataset_ids = config["SNT_DATASET_IDENTIFIERS"]
    definitions = config["DHIS2_DATA_DEFINITIONS"]

    errors = []

    # Required keys in SNT_CONFIG
    required_snt_keys = [
        "COUNTRY_CODE",
        "DHIS2_ADMINISTRATION_1",
        "DHIS2_ADMINISTRATION_2",
        "ANALYTICS_ORG_UNITS_LEVEL",
    ]
    for key in required_snt_keys:
        val = snt_config.get(key)
        if val is None or not str(val).strip():
            errors.append(f"SNT_CONFIG.{key} is missing or empty")

    # Required dataset identifiers
    required_dataset_keys = [
        "DHIS2_DATASET_EXTRACTS",
        "DHIS2_DATASET_FORMATTED",
        "DHIS2_POPULATION_TRANSFORMATION",
        "DHIS2_REPORTING_RATE",
        "DHIS2_OUTLIERS_IMPUTATION",
        "DHIS2_INCIDENCE",
        "DHS_INDICATORS",
        "WORLDPOP_DATASET_EXTRACT",
        "SNT_HEALTHCARE_ACCESS",
        "ERA5_DATASET_CLIMATE",
        "SNT_SEASONALITY_RAINFALL",
        "SNT_SEASONALITY_CASES",
        "SNT_MAP_EXTRACTS",
        "DHIS2_QUALITY_OF_CARE",
        "SNT_POPULATION_USER_PROVIDED",
    ]
    for key in required_dataset_keys:
        val = dataset_ids.get(key)
        if val is None or not str(val).strip():
            errors.append(f"SNT_DATASET_IDENTIFIERS.{key} is missing or empty")

    # Check population indicators
    pop_indicators = definitions.get("POPULATION_INDICATOR_DEFINITIONS", {})
    if not pop_indicators:
        errors.append("DHIS2_DATA_DEFINITIONS.POPULATION_INDICATOR_DEFINITIONS is missing or empty")

    elif pop_indicators.get("POPULATION") is None:
        errors.append(
            "DHIS2_DATA_DEFINITIONS.POPULATION_INDICATOR_DEFINITIONS.POPULATION is not configured"
            " (e.g.: 'POPULATION': {'ids': ['dhis2id'], 'type': 'dataElement'})"
        )

    # Check at least one indicator under DHIS2_INDICATOR_DEFINITIONS
    indicator_defs = definitions.get("DHIS2_INDICATOR_DEFINITIONS", {})
    try:
        flat_indicators = [val for sublist in indicator_defs.values() for val in sublist]
    except TypeError:
        errors.append("DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS has invalid structure (values must be lists)")
    else:
        if not flat_indicators:
            errors.append("DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS has no indicators defined")

    if errors:
        error_list = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"Configuration validation failed with {len(errors)} error(s):\n{error_list}")


def _write_file_to_tmp(src: Path) -> str:
    """Read a supported file and write a copy of it to a named temporary file.

    Args:
        src (Path): Source file (.parquet, .csv, .geojson or .json).

    Returns:
        str: The path to the temporary file created.

    Raises:
        ValueError: If the file format is not supported.
    """
    ext = src.suffix.lower()
    if ext == ".parquet":
        # # Convert NaN to null so downstream readers (pandas or polars) see consistent null values.
        data = pl.read_parquet(src).with_columns(pl.col(pl.Float32, pl.Float64).fill_nan(None))
    elif ext == ".csv":
        data = pd.read_csv(src)
    elif ext == ".geojson":
        data = gpd.read_file(src)
    elif ext == ".json":
        with src.open(encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported file format: {src.name}")

    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp_path = tmp.name

    if ext == ".parquet":
        data.write_parquet(tmp_path)
    elif ext == ".csv":
        data.to_csv(tmp_path, index=False)
    elif ext == ".geojson":
        data.to_file(tmp_path, driver="GeoJSON")
    elif ext == ".json":
        with Path(tmp_path).open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    return tmp_path


def add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    file_paths: list[Path],
    ds_version_prefix: str = "SNT",
) -> bool:
    """Add files to a new dataset version.

    The version is only created once the first file is ready; missing or unsupported files are skipped.

    Args:
        dataset_id (str): ID of the dataset to which files will be added.
        country_code (str): Country code used for naming the dataset version.
        file_paths (list[Path]): Files to add to the dataset.
        ds_version_prefix (str): Prefix for the dataset version name. Defaults to "SNT".

    Returns:
        bool: True if at least one file was added successfully, False otherwise.

    Raises:
        ValueError: If the dataset ID is not specified in the configuration.
    """
    if dataset_id is None:
        raise ValueError("Dataset ID is not specified in the configuration.")

    added_any = False
    new_version = None

    for src in file_paths:
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        try:  # ruff:ignore[too-many-statements-in-try-clause]
            tmp_path = _write_file_to_tmp(src)
            if not added_any:
                new_version = get_new_dataset_version(ds_id=dataset_id, prefix=f"{ds_version_prefix}_{country_code}")
                current_run.log_info(f"New dataset version created : {new_version.name}")
                added_any = True
            new_version.add_file(tmp_path, filename=src.name)
            current_run.log_info(f"File {src.name} added to dataset version : {new_version.name}")
        except ValueError:
            current_run.log_warning(f"Unsupported file format: {src.name}")
            continue
        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added : {e}")
            continue

    if not added_any:
        current_run.log_warning("No valid files found. Dataset version was not created.")
        return False

    return True


def save_pipeline_parameters(
    pipeline_name: str,
    parameters: dict[str, Any],
    output_path: Path,
    country_code: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Save pipeline execution parameters to JSON for provenance tracking.

    Creates a JSON file mapping parameter names to values, one entry per parameter.
    The execution timestamp is included as an entry with key "EXECUTION_TIMESTAMP".

    Args:
        pipeline_name (str): Name of the pipeline being executed (e.g. "snt_dhis2_incidence").
        parameters (dict[str, Any]): Parameters used in this pipeline run.
        output_path (Path): Directory where the parameters file will be saved.
        country_code (str): Country code for file naming (e.g. "COD", "NER").
        extra_metadata (dict[str, Any] | None): Additional metadata to include (e.g. input file names,
            source dataset versions). Defaults to None.

    Returns:
        Path: Path to the created parameters JSON file. Add to file_paths when calling add_files_to_dataset.

    Examples:
    >>> params_file = save_pipeline_parameters(
    ...     pipeline_name="snt_dhis2_incidence",
    ...     parameters={"n1_method": "PRES", "routine_data_choice": "imputed"},
    ...     output_path=data_path,
    ...     country_code="COD",
    ...     extra_metadata={"input_file": "COD_routine_imputed.parquet"},
    ... )
    >>> add_files_to_dataset(..., file_paths=[params_file, ...])
    """
    output_path.mkdir(parents=True, exist_ok=True)

    execution_timestamp = datetime.now(UTC).isoformat()
    normalized_parameters = {str(key).upper(): value for key, value in parameters.items()}

    normalized_extra_metadata: dict[str, Any] = {}
    if extra_metadata:
        normalized_extra_metadata = {str(key).upper(): value for key, value in extra_metadata.items()}

    all_params = {
        "EXECUTION_TIMESTAMP": execution_timestamp,
        "PIPELINE_NAME": pipeline_name,
        "COUNTRY_CODE": country_code,
        **normalized_parameters,
    }

    if normalized_extra_metadata:
        all_params.update(normalized_extra_metadata)

    json_path = output_path / f"{country_code}_parameters.json"

    with Path.open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_params, f, indent=2, default=str)

    return json_path


def get_new_dataset_version(ds_id: str, prefix: str = "ds", ds_desc: str = "SNT Process dataset") -> DatasetVersion:
    """Create and return a new dataset version, creating the dataset first if it does not exist.

    Args:
        ds_id (str): ID of the dataset for which a new version will be created.
        prefix (str): Prefix for the dataset version name. Defaults to "ds".
        ds_desc (str): Description used if the dataset has to be created. Defaults to "SNT Process dataset".

    Returns:
        DatasetVersion: The newly created dataset version.

    Raises:
        Exception: If an error occurs while creating the new dataset version.
    """
    # Use get_dataset first so we reuse existing dataset with this exact slug (avoids duplicates)
    try:
        dataset = workspace.get_dataset(ds_id)
    except Exception as e:
        current_run.log_warning(f"Error retrieving dataset: {ds_id}")
        current_run.log_debug(f"Error retrieving dataset: {ds_id}: {e}")
        dataset = None

    if dataset is None:
        current_run.log_warning(f"Creating new Dataset with ID : {ds_id}")
        dataset = workspace.create_dataset(name=ds_id.replace("-", "_").upper(), description=ds_desc)

    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        current_run.log_debug(f"An error occurred while creating the new dataset version: {e}")
        raise Exception(f"An error occurred while creating the new dataset version: {e}") from e

    return new_version


def delete_raw_files(directory: Path, pattern: str) -> None:
    """Delete all files matching a glob pattern in the specified directory.

    Args:
        directory (Path): Directory in which to search for files to delete.
        pattern (str): Glob pattern matching the files to delete (e.g. "*_routine_data_*.parquet").

    Raises:
        Exception: If a matching file cannot be deleted.
    """
    files_to_delete = list(directory.glob(pattern))

    for file in files_to_delete:
        try:
            file.unlink()
        except Exception as e:
            raise Exception(f"Failed to delete {file}: {e}") from e


def get_file_from_dataset(dataset_id: str, filename: str) -> pd.DataFrame | gpd.GeoDataFrame | dict:
    """Download a file from the latest version of a dataset and load it.

    Args:
        dataset_id (str): ID of the dataset.
        filename (str): Name of the file to retrieve (.csv, .parquet, .json, .geojson, .gpkg or .shp).

    Returns:
        pd.DataFrame | gpd.GeoDataFrame | dict: The DataFrame, GeoDataFrame or dict containing the data.

    Raises:
        ValueError: If the dataset, version or file is not found, the download fails, or the file type
            is not supported.
    """
    dataset = workspace.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset with ID {dataset_id} not found.")

    version = dataset.latest_version
    if not version:
        raise ValueError(f"No versions found for dataset {dataset_id}.")

    file_path = version.get_file(filename)
    if not file_path:
        raise ValueError(f"File {filename} not found in dataset {dataset_id}.")

    suffix = Path(filename).suffix.lower()
    url = file_path.download_url
    r = requests.get(url)

    if r.status_code != 200:
        raise ValueError(f"Failed to download file: {r.status_code} - {r.text}")

    if len(r.content) < 100:
        raise ValueError(f"Downloaded file is suspiciously small ({len(r.content)} bytes)")

    if suffix in [".csv", ".parquet", ".geojson", ".gpkg", ".shp", ".json"]:
        tfile_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tfile:
                tfile_path = tfile.name
                tfile.write(r.content)
                tfile.flush()
            if suffix == ".csv":
                return pd.read_csv(tfile_path)
            if suffix == ".parquet":
                return pd.read_parquet(tfile_path)
            if suffix == ".json":
                with Path(tfile_path).open(encoding="utf-8") as f:
                    return json.load(f)
            return gpd.read_file(tfile_path)
        finally:
            if tfile_path and Path(tfile_path).exists():
                Path(tfile_path).unlink()

    raise ValueError(f"Unsupported file type: {suffix}")


def dataset_file_exists(ds_id: str, filename: str) -> bool:
    """Check if a file exists in the latest version of a dataset.

    Args:
        ds_id (str): ID of the dataset to check.
        filename (str): Name of the file to check for.

    Returns:
        bool: True if the file exists, False otherwise (including when the dataset cannot be retrieved).
    """
    try:
        dataset = workspace.get_dataset(ds_id)
        if dataset.latest_version is not None and hasattr(dataset.latest_version, "files"):
            return any(file.filename == filename for file in dataset.latest_version.files)
        return False
    except Exception:
        return False


def push_data_to_db_table(
    table_name: str,
    dataframe: pd.DataFrame | None = None,
    file_path: Path | None = None,
    db_url: str | None = None,
) -> None:
    """Push data to a database table, replacing the table if it already exists.

    Args:
        table_name (str): Name of the table to create or replace.
        dataframe (pd.DataFrame | None): Data to push. Ignored if `file_path` is given. Defaults to None.
        file_path (Path | None): Parquet file containing the data to push. Takes precedence over
            `dataframe`. Defaults to None.
        db_url (str | None): Database URL to connect to. Defaults to None (workspace database URL).

    Raises:
        ValueError: If `table_name` is empty, no data source is given, or the data is empty.
        FileNotFoundError: If `file_path` does not exist.
        Exception: If writing the table fails.
    """
    current_run.log_info(f"Pushing data to table : {table_name}")

    if table_name is None or not table_name:
        raise ValueError("Parameter 'table_name' cannot be empty.")

    if dataframe is None and file_path is None:
        raise ValueError("You must provide either a dataframe (pandas) or a file_path")

    if file_path is not None:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")
        df = pd.read_parquet(file_path)
    else:
        df = dataframe.copy()

    if df.empty:
        raise ValueError(f"DataFrame is empty, cannot create DB table '{table_name}'")

    if db_url:
        database_url = db_url
    else:
        # Use the workspace database URL if not provided
        database_url = workspace.database_url

    try:
        # Create engine
        dbengine = create_engine(database_url)
        df.to_sql(table_name, dbengine, index=False, if_exists="replace", chunksize=4096)
    except Exception as e:
        raise Exception(f"Error creating table '{table_name}' with file {file_path}: {e}") from e


def check_outputs_generated(file_paths: list[Path], run_start_ts: float) -> None:
    """Raise if any expected output was not written during the current run.

    Guards against publishing stale files: all outliers imputation pipelines write the same
    filenames, so a leftover file may come from a previous run of another method.

    Args:
        file_paths (list[Path]): Output files the notebook is expected to produce.
        run_start_ts (float): Timestamp taken just before the notebook ran; files modified earlier are stale.

    Raises:
        RuntimeError: If a file is missing or was last modified before `run_start_ts`.
    """
    missing = [p.name for p in file_paths if not p.exists() or p.stat().st_mtime < run_start_ts]
    if missing:
        msg = f"Expected output files were not generated during this run: {', '.join(missing)}"
        current_run.log_error(msg)
        raise RuntimeError(msg)
