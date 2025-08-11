import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import geopandas as gpd
import tempfile
import papermill as pm
from openhexa.sdk import current_run, workspace
import subprocess
from subprocess import CalledProcessError
from nbclient.exceptions import CellTimeoutError
from openhexa.sdk.datasets.dataset import DatasetVersion
import requests
import shutil
import stat
from git import Repo
from papermill.exceptions import PapermillExecutionError
import re
import fnmatch
from sqlalchemy import create_engine


def pull_scripts_from_repository(
    pipeline_name: str,
    report_scripts: list[str],
    code_scripts: list[str],
    repo_path: Path = Path("/tmp"),
    repo_name: str = "snt_development",
    pipeline_parent_folder: Path = Path(workspace.files_path, "pipelines"),
) -> None:
    """Pull the latest pipeline scripts from the SNT repository and update the local workspace.

    Parameters
    ----------
    pipeline_name : str
        The name of the pipeline for which scripts are being updated.
    report_scripts : list[str]
        List of reporting script names to be updated.
    code_scripts : list[str]
        List of code script names to be updated.
    repo_path : Path, optional
        The path to the repository where the scripts are stored (default is "/tmp").
    repo_name : str, optional
        The name of the repository from which to pull the scripts (default is "snt_development").
        It also corresponds to the folder where the repo is stored.
    pipeline_parent_folder : Path, optional
        The path to the pipeline location (not the full path!) in the workspace where
        the scripts will be replaced (default is "pipelines" in the SNT workspaces files path).

    This function attempts to update reporting scripts and logs errors or warnings if the update fails.
    """
    # Paths Repository -> Workspace
    repository_source = repo_path / repo_name / "pipelines" / pipeline_name
    pipeline_target = pipeline_parent_folder / pipeline_name

    # Create the mapping of script paths
    reporting_paths = {
        (repository_source / "reporting" / r): (pipeline_target / "reporting" / r)
        for r in report_scripts
    }
    code_paths = {
        (repository_source / "code" / c): (pipeline_target / "code" / c)
        for c in code_scripts
    }

    current_run.log_info(
        f"Updating scripts {', '.join(report_scripts + code_scripts)} from repository '{repo_name}'"
    )

    try:
        # Pull scripts from the SNT repository (replace local)
        load_scripts_for_pipeline(
            snt_script_paths=reporting_paths | code_paths,
            repository_path=repo_path,
            repository_name=repo_name,
        )
    except Exception as e:
        current_run.log_error(f"Error: {e}")
        current_run.log_warning("Continuing without scripts update.")


def load_scripts_for_pipeline(
    snt_script_paths: dict[Path],
    repository_path: Path = Path("/tmp"),
    repository_name: str = "snt_development",
) -> None:
    """Load all scripts required for the SNT pipeline.

    Parameters
    ----------
    snt_script_paths : dict[Path]
        A dictionary where keys are source paths in the repository and values are target paths in the OpenHexa workspace.
        Example: {'pipelines/[pipeline name]/snt_pipeline_utils.py': '/home/hexa/workspace/pipelines/[pipeline name]/snt_pipeline_utils.py'}
        WARNINGS: This function will overwrite existing scripts in the pipeline folder.
    repository_path : Path, optional
        The local path where the repository will be cloned. Defaults to '/tmp' (temporary OH directory).
    repository_name : str, optional
        The name of the repository to clone. Defaults to 'snt_development'.
    """
    try:
        get_repository(local_repo_path=repository_path, repo_name=repository_name)
    except Exception as e:
        raise Exception(f"Error while loading repository: {e}") from e

    for source_path, target_path in snt_script_paths.items():
        script_source = repository_path / repository_name / source_path
        if script_source.exists():
            current_run.log_debug(f"Loading pipeline script: {script_source}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(script_source, target_path)
        else:
            current_run.log_warning(f"Pipeline scripts : {script_source} not found")
    current_run.log_info(
        f"Pipeline scripts loaded successfully from https://github.com/BLSQ/{repository_name}.git"
    )


def force_remove_readonly(func: callable, path: Path, exc_info: tuple) -> None:
    """Error handler for shutil.rmtree that makes read-only files writable and retries."""
    try:
        Path.chmod(path, stat.S_IWRITE)  # Make the file writable
        func(path)
    except Exception as e:
        raise Exception(
            f"Failed to remove {path} after changing permissions: {e}"
        ) from e


def safe_rmtree(path: Path) -> None:
    """Removes a directory tree, handling read-only files."""
    if path.exists():
        shutil.rmtree(path, onerror=force_remove_readonly)


def clone_repository(
    repo_owner: str,
    repo_name: str,
    dest_path: Path,
    token: str | None = None,
    depth: int = 1,
) -> None:
    """Clone a private GitHub repository using a token, or public without a token.

    Args:
        token (str | None): GitHub personal access token.
        repo_owner (str): Owner of the repository.
        repo_name (str): Name of the repository.
        dest_path (Path): Destination path to clone the repository into.
        depth (int, optional): Depth for shallow clone. Defaults to 1.
    """
    if token:
        url = f"https://{token}:x-oauth-basic@github.com/{repo_owner}/{repo_name}.git"
    else:
        url = f"https://github.com/{repo_owner}/{repo_name}.git"  # Public
    Repo.clone_from(url=url, to_path=dest_path, depth=depth)


def get_repository(
    local_repo_path: Path,
    repo_name: str = "snt_development",
    repo_owner: str = "BLSQ",
    token: str | None = None,
) -> None:
    """Clone a GitHub repo, copy a specific folder from it, and delete the rest.

    Args:
        repo_name (str): Name of the GitHub repository.
        repo_path (Path): Folder to clone the repo into. A new folder with the name of the repository will be created.
        target_folder_in_repo (str): Folder path inside repo to extract (e.g. "snt_pipeline_utils")
        output_path (Path): Where to save the extracted folder
        repo_owner (str): Owner of the repository, defaults to "BLSQR".
        token (str | None): GitHub personal access token, if needed for private repos.
    """
    current_run.log_debug(f"Cloning repository: {repo_name}")

    # Ensure the local_repo_path is clean before cloning
    temp_repository = local_repo_path / repo_name
    safe_rmtree(temp_repository)

    try:
        clone_repository(
            repo_owner=repo_owner,
            repo_name=repo_name,
            dest_path=temp_repository,
            token=token,
        )
    except Exception as e:
        raise Exception(f"Failed to clone repository {repo_name}: {e}") from e

    current_run.log_debug(f"Extracted repository to '{temp_repository}'")


def run_notebook(
    nb_path: Path,
    out_nb_path: Path,
    parameters: dict,
    error_label_severity_map: dict = {},
    kernel_name: str = "ir",
):
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_name : str
        The name of the notebook to execute (without the .ipynb extension).
    nb_path : Path
        The path to the directory containing the notebook.
    out_nb_path : Path
        The path to the directory where the output notebook will be saved.
    parameters : dict
        A dictionary of parameters to pass to the notebook.
    error_label_severity_map : dict
        A dictionary mapping error labels to their severity levels.
        Example: {'LABEL': 'error', 'ANOTHER_LABEL': 'warning', ...}
    kernel_name : str, optional
        The name of the kernel to use for execution (default is "ir" for R, "python3" for Python).
    """
    current_run.log_info(f"Executing notebook: {nb_path}")
    file_stem = nb_path.stem
    extension = nb_path.suffix
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    out_nb_full_path = (
        out_nb_path / f"{file_stem}_OUTPUT_{execution_timestamp}{extension}"
    )
    out_nb_path.mkdir(parents=True, exist_ok=True)

    try:
        pm.execute_notebook(
            input_path=nb_path,
            output_path=out_nb_full_path,
            parameters=parameters,
            kernel_name=kernel_name,
            request_save_on_cell_execute=False,
        )
    except PapermillExecutionError as e:
        handle_rkernel_error_with_labels(
            e, error_label_severity_map
        )  # for labeled R kernel errors
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e


def run_report_notebook(
    nb_file: Path,
    nb_output_path: Path,
    nb_parameters: dict | None = None,
    error_label_severity_map: dict = {},
    kernel_name: str = "ir",
    ready: bool = True,
) -> None:
    """Execute a Jupyter notebook using Papermill.

    Parameters
    ----------
    nb_file : Path
        The full file path to the notebook.
    nb_output_path : Path
        The path to the directory where the output notebook will be saved.
    nb_parameters : dict | None, optional
        A dictionary of parameters to pass to the notebook (default is None).
    error_label_severity_map : dict
        A dictionary mapping error labels to their severity levels.
        Levels can be 'warning', 'error', or 'critical'.
        Example: {'LABEL': 'error', 'ANOTHER_LABEL': 'warning', ...}
    ready : bool, optional
        Whether the notebook should be executed (default is True) (can be used as openHexa @task signal).
    """
    if not ready:
        current_run.log_info("Reporting execution skipped.")
        return

    current_run.log_info(f"Executing report notebook: {nb_file}")
    execution_timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    nb_output_full_path = (
        nb_output_path / f"{nb_file.stem}_OUTPUT_{execution_timestamp}.ipynb"
    )
    nb_output_path.mkdir(parents=True, exist_ok=True)

    try:
        pm.execute_notebook(
            input_path=nb_file,
            output_path=nb_output_full_path,
            parameters=nb_parameters,
            kernel_name=kernel_name,
            request_save_on_cell_execute=False,
        )
    except CellTimeoutError as e:
        raise CellTimeoutError(f"Notebook execution timed out: {e}") from e
    except PapermillExecutionError as e:
        handle_rkernel_error_with_labels(
            e, error_label_severity_map
        )  # for labeled R kernel errors
    except Exception as e:
        raise Exception(f"Error executing the notebook {type(e)}: {e}") from e
    generate_html_report(nb_output_full_path)


def get_matching_filename_from_dataset_last_version(
    dataset_id: str, filename_pattern: str
) -> str:
    """Get the filename from openhexa dataset last version that matches the pattern.

    Returns
    -------
    str
        The filename that matches the pattern, if not found returns None.
    """
    dataset = workspace.get_dataset(dataset_id)
    if not dataset:
        raise ValueError(f"Dataset with ID {dataset_id} not found.")

    version = dataset.latest_version
    if not version:
        raise ValueError(f"No versions found for dataset {dataset_id}.")

    for file in version.files:
        current_run.log_debug(f"DS file: {file.filename}")
        if fnmatch.fnmatch(file.filename, filename_pattern):
            current_run.log_debug(f"Found file matching pattern: {file.filename}")
            return file.filename

    raise ValueError(
        f"File with pattern {filename_pattern} not found in dataset {dataset_id}."
    )


def generate_html_report(output_notebook_path: Path, out_format: str = "html") -> None:
    """Generate an HTML report from a Jupyter notebook.

    Parameters
    ----------
    output_notebook_path : Path
        Path to the output notebook file.
    out_format : str
        output extension

    Raises
    ------
    RuntimeError
        If an error occurs during the conversion process.
    """
    if (
        not output_notebook_path.is_file()
        or output_notebook_path.suffix.lower() != ".ipynb"
    ):
        raise RuntimeError(f"Invalid notebook path: {output_notebook_path}")

    report_path = output_notebook_path.with_suffix(".html")
    current_run.log_info(f"Generating HTML report {report_path}")
    cmd = [
        "jupyter",
        "nbconvert",
        f"--to={out_format}",
        str(output_notebook_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except CalledProcessError as e:
        raise CalledProcessError(
            e.returncode, e.cmd, output=e.output, stderr=e.stderr
        ) from e

    current_run.add_file_output(report_path.as_posix())


def handle_rkernel_error_with_labels(
    error: Exception, error_labels: dict | None = None
):
    """Handle errors from the R kernel and log them with appropriate labels.
    Error severity levels handled:
    - warning: Logs as a warning message.
    - error: Logs as an error message and raises a RuntimeError.
    - critical: Logs as a critical message and raises a RuntimeError.
    (!) Attention: Label [ERROR DETAILS] can be used to specify detailed information from the error message.
    This label can optionally added at the end of the error message.

    Example error message:
    "Error: [LABEL] Some error message to the user [ERROR DETAILS] Additional error details here."

    Parameters
    ----------
    error : Exception
        The error object raised by the R kernel.
    error_labels : dict
        A dictionary mapping error labels to their severity levels.
        Levels can be 'warning', 'error', or 'critical'.
        Example: {'LABEL': 'error', 'ANOTHER_LABEL': 'warning', ...}
    """
    if error_labels is None:
        error_labels = {}

    error_msg = getattr(error, "evalue", str(error))
    current_run.log_debug(f"Error message captured from R: {error_msg}")
    matched = False

    for label, severity in error_labels.items():
        if label in error_msg:
            pattern = rf"\s*{re.escape(label)}\s*(.*?)(?:\s*\[ERROR DETAILS\]\s*(.*))?$"
            match = re.search(pattern, error_msg)

            if match:
                message_main = match.group(1).strip()
                message_details = match.group(2).strip() if match.group(2) else ""
                matched = True
                if severity == "warning":
                    current_run.log_warning(f"{message_main}")
                elif severity == "error":
                    raise RuntimeError(f"{message_main} {message_details}")
                else:
                    raise RuntimeError(
                        f"{label} {message_main}. Unknown severity '{severity}'"
                    )

                break

    if not matched:
        raise RuntimeError(str(error))


def load_configuration_snt(config_path: Path) -> dict:
    """Load the SNT configuration from a JSON file.

    Parameters
    ----------
    config_path : str
        Path to the configuration JSON file.

    Returns
    -------
    dict
        The loaded configuration as a dictionary.

    Raises
    ------
    FileNotFoundError
        If the configuration file is not found.
    ValueError
        If the configuration file contains invalid JSON.
    Exception
        For any other unexpected errors.
    """
    try:
        # Load the JSON file
        with config_path.open("r") as file:
            config_json = json.load(file)
        current_run.log_info(f"SNT configuration loaded: {config_path}")

    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: The file {config_path} was not found.") from e
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: The file contains invalid JSON {e}") from e
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}") from e

    return config_json


def validate_config(config: dict) -> None:
    """Validate that the critical configuration values are set properly."""
    try:
        snt_config = config["SNT_CONFIG"]
        dataset_ids = config["SNT_DATASET_IDENTIFIERS"]
        definitions = config["DHIS2_DATA_DEFINITIONS"]
    except KeyError as e:
        raise KeyError(f"Missing top-level key in config: {e}") from e

    # Required keys in SNT_CONFIG
    required_snt_keys = [
        "COUNTRY_CODE",
        "DHIS2_ADMINISTRATION_1",
        "DHIS2_ADMINISTRATION_2",
        "ANALYTICS_ORG_UNITS_LEVEL",
        "POPULATION_ORG_UNITS_LEVEL",
    ]
    for key in required_snt_keys:
        if key not in snt_config or snt_config[key] in [None, ""]:
            raise ValueError(f"Missing or empty configuration for: SNT_CONFIG.{key}")

    # Required dataset identifiers
    required_dataset_keys = [
        "DHIS2_DATASET_EXTRACTS",
        "DHIS2_DATASET_FORMATTED",
        "DHIS2_REPORTING_RATE",
        "DHIS2_INCIDENCE",
        "DHIS2_OUTLIERS_DETECTION",
        "DHIS2_OUTLIERS_REMOVAL_IMPUTATION",
        "DHS_INDICATORS",
        "WORLDPOP_DATASET_EXTRACT",
        "ERA5_DATASET_CLIMATE",
        "SNT_SEASONALITY",
        "SNT_MAP_EXTRACT",
        "SNT_RESULTS",
    ]
    for key in required_dataset_keys:
        if key not in dataset_ids or dataset_ids[key] in [None, ""]:
            raise ValueError(
                f"Missing or empty configuration for: SNT_DATASET_IDENTIFIERS.{key}"
            )

    # Check population indicator
    pop_indicators = definitions.get("POPULATION_INDICATOR_DEFINITIONS", {})
    tot_population = pop_indicators.get("POPULATION", [])
    if not tot_population:
        raise ValueError("Missing or empty POPULATION indicator definition.")

    # Check at least one indicator under DHIS2_INDICATOR_DEFINITIONS
    indicator_defs = definitions.get("DHIS2_INDICATOR_DEFINITIONS", {})
    flat_indicators = [val for sublist in indicator_defs.values() for val in sublist]
    if not flat_indicators:
        raise ValueError("No indicators defined under DHIS2_INDICATOR_DEFINITIONS.")


def add_files_to_dataset(
    dataset_id: str,
    country_code: str,
    file_paths: list[Path],
    ds_version_prefix: str = "SNT",
) -> bool:
    """Add files to a new dataset version.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset to which files will be added.
    country_code : str
        The country code used for naming the dataset version.
    file_paths : list[Path]
        A list of file paths to be added to the dataset.

    Raises
    ------
    ValueError
        If the dataset ID is not specified in the configuration.

    Returns
    -------
    bool
        True if at least one file was added successfully, False otherwise.
    """
    if dataset_id is None:
        raise ValueError("Dataset ID is not specified in the configuration.")

    added_any = False

    for src in file_paths:
        if not src.exists():
            current_run.log_warning(f"File not found: {src}")
            continue

        try:
            # Determine file extension
            ext = src.suffix.lower()
            if ext == ".parquet":
                df = pd.read_parquet(src)
                tmp_suffix = ".parquet"
            elif ext == ".csv":
                df = pd.read_csv(src)
                tmp_suffix = ".csv"
            elif ext == ".geojson":
                gdf = gpd.read_file(src)
                tmp_suffix = ".geojson"
            else:
                current_run.log_warning(f"Unsupported file format: {src.name}")
                continue

            with tempfile.NamedTemporaryFile(suffix=tmp_suffix) as tmp:
                if ext == ".parquet":
                    df.to_parquet(tmp.name)
                elif ext == ".csv":
                    df.to_csv(tmp.name, index=False)
                elif ext == ".geojson":
                    gdf.to_file(tmp.name, driver="GeoJSON")

                if not added_any:
                    new_version = get_new_dataset_version(
                        ds_id=dataset_id, prefix=f"{ds_version_prefix}_{country_code}"
                    )
                    current_run.log_info(
                        f"New dataset version created : {new_version.name}"
                    )
                    added_any = True
                new_version.add_file(tmp.name, filename=src.name)
                current_run.log_info(
                    f"File {src.name} added to dataset version : {new_version.name}"
                )
        except Exception as e:
            current_run.log_warning(f"File {src.name} cannot be added : {e}")
            continue

    if not added_any:
        current_run.log_warning(
            "No valid files found. Dataset version was not created."
        )
        return False

    return True


def get_new_dataset_version(
    ds_id: str, prefix: str = "ds", ds_desc: str = "SNT Process dataset"
) -> DatasetVersion:
    """Create and return a new dataset version.

    Parameters
    ----------
    ds_id : str
        The ID of the dataset for which a new version will be created.
    prefix : str, optional
        Prefix for the dataset version name (default is "ds").

    Returns
    -------
    DatasetVersion
        The newly created dataset version.

    Raises
    ------
    Exception
        If an error occurs while creating the new dataset version.
    """
    existing_datasets = workspace.list_datasets()
    if ds_id in [eds.slug for eds in existing_datasets]:
        dataset = workspace.get_dataset(ds_id)
    else:
        current_run.log_warning(
            f"Dataset with ID {ds_id} not found, creating a new one."
        )
        dataset = workspace.create_dataset(
            name=ds_id.replace("-", "_").upper(), description=ds_desc
        )

    version_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    try:
        new_version = dataset.create_version(version_name)
    except Exception as e:
        raise Exception(
            f"An error occurred while creating the new dataset version: {e}"
        ) from e

    return new_version


def remove_all_files(folder_path: str) -> None:
    """Remove all files from the specified folder.

    Parameters
    ----------
    folder_path : str
        Path to the folder from which all files will be removed.

    Raises
    ------
    ValueError
        If the provided path is not a valid directory.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        raise ValueError(f'"{folder_path}" is not a valid directory')

    for item in folder.iterdir():
        if item.is_file():
            item.unlink()


def delete_raw_files(directory: Path, pattern: str) -> None:
    """Delete raw parquet files for a given country code in the specified directory.

    Parameters
    ----------
    directory : Path
        The directory in which to search for files to delete.
    pattern : str
        The pattern to match files for deletion.

    This function deletes all files matching the pattern in the given directory.
    """
    files_to_delete = list(directory.glob(pattern))

    for file in files_to_delete:
        try:
            file.unlink()
        except Exception as e:
            raise Exception(f"Failed to delete {file}: {e}") from e


def get_file_from_dataset(
    dataset_id: str, filename: str
) -> pd.DataFrame | gpd.GeoDataFrame:
    """Get a file from a dataset.

    Parameters
    ----------
    dataset_id : str
        The ID of the dataset.
    filename : str
        The name of the file to retrieve.

    Returns
    -------
    pd.DataFrame | gpd.GeoDataFrame
        The DataFrame or GeoDataFrame containing the data.
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
        raise ValueError(
            f"Downloaded file is suspiciously small ({len(r.content)} bytes)"
        )

    if suffix in [".csv", ".parquet", ".geojson", ".gpkg", ".shp"]:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tfile:
            tfile.write(r.content)
            tfile.flush()
            if suffix == ".csv":
                return pd.read_csv(tfile.name)
            elif suffix == ".parquet":
                return pd.read_parquet(tfile.name)
            else:
                return gpd.read_file(tfile.name)

    raise ValueError(f"Unsupported file type: {suffix}")


def copy_json_file(
    source_folder: Path, destination_folder: Path, json_filename: str
) -> None:
    """Copies a JSON file directly from a source folder to a destination folder using pathlib.

    This method does not read or modify the file's content in Python.

    Args:
        source_folder (str or Path): The path to the folder containing the source JSON file.
        destination_folder (str or Path): The path to the folder where the JSON file will be copied.
        json_filename (str): The name of the JSON file (e.g., "my_data.json").
    """
    source_path = source_folder / json_filename
    destination_path = destination_folder / json_filename

    try:
        # Ensure the destination folder exists (parents=True creates parent directories if needed)
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy(source_path, destination_path)
        current_run.log_debug(
            f"Successfully copied '{json_filename}' from '{source_folder}' to '{destination_folder}'."
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Error: Source file '{source_path}' not found.") from e
    except Exception as e:
        raise Exception(
            f"An error occurred while copying '{json_filename}': {e}"
        ) from e


def dataset_file_exists(ds_id: str, filename: str) -> bool:
    """Check if a file exists in a dataset.

    Parameters
    ----------
    ds_id : str
        The ID of the dataset to check.
    filename : str
        The name of the file to check for.

    Returns
    -------
    bool
        True if the file exists, False otherwise.
    """
    try:
        dataset = workspace.get_dataset(ds_id)
        if dataset.latest_version is not None and hasattr(
            dataset.latest_version, "files"
        ):
            return any(
                file.filename == filename for file in dataset.latest_version.files
            )
        return False
    except Exception:
        return False


def push_data_to_db_table(
    table_name: str,
    dataframe: pd.DataFrame | None = None,
    file_path: Path | None = None,
    db_url: str | None = None,
) -> None:
    """Push data to a database table.

    Parameters
    ----------
    table_name : str
        The name of the table to update or create.
    dataframe : pd.DataFrame | None
        The DataFrame containing the data to push to the table. If None, data will be read from file_path.
    file_path : Path | None
        The path to the file containing the data to push to the table. If None, data will be taken from the 'data' parameter.
    db_url : str | None
        The database URL to connect to. If None, the workspace database URL will be used.
    """
    current_run.log_info(f"Pushing data to table : {table_name}")

    if table_name is None or table_name == "":
        raise ValueError("Table name cannot be None")

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
        df.to_sql(
            table_name, dbengine, index=False, if_exists="replace", chunksize=4096
        )
    except Exception as e:
        raise Exception(
            f"Error creating table '{table_name}' with file {file_path}: {e}"
        ) from e
