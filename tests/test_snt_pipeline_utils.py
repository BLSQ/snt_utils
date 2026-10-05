import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import nbformat
import pandas as pd
import papermill as pm
import pytest
from sqlalchemy import create_engine

from snt_lib import snt_pipeline_utils
from snt_lib.snt_pipeline_utils import (
    add_files_to_dataset,
    check_outputs_generated,
    clone_repository,
    copy_file,
    dataset_file_exists,
    delete_raw_files,
    force_remove_readonly,
    generate_html_report,
    get_file_from_dataset,
    get_matching_filename_from_dataset_last_version,
    get_new_dataset_version,
    get_repository,
    handle_rkernel_error_with_labels,
    load_configuration_snt,
    load_scripts_for_pipeline,
    pull_scripts_from_repository,
    push_data_to_db_table,
    remove_all_files,
    run_notebook,
    run_report_notebook,
    save_pipeline_parameters,
    validate_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_configuration_snt_valid():
    """Test that loading a valid configuration file returns a dictionary with expected keys."""
    config = load_configuration_snt(FIXTURES / "SNT_config_valid.json")
    assert isinstance(config, dict)
    assert "SNT_CONFIG" in config


def test_load_configuration_snt_missing_file():
    """Test that loading a non-existent configuration file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_configuration_snt(FIXTURES / "nonexistent.json")


def test_validate_config_valid():
    """Test that a valid configuration passes validation without raising an error."""
    config = load_configuration_snt(FIXTURES / "SNT_config_valid.json")
    validate_config(config)  # should not raise


def test_validate_config_missing_top_level_key():
    """Test that missing top-level keys in the configuration raise a KeyError."""
    with pytest.raises(KeyError):
        validate_config({"SNT_CONFIG": {}, "SNT_DATASET_IDENTIFIERS": {}})


def test_validate_config_pop_definition_missing():
    """Test error is raised when POPULATION_INDICATOR_DEFINITIONS is present but missing POPULATION definition."""
    config = load_configuration_snt(FIXTURES / "SNT_config_pop_def_missing.json")
    with pytest.raises(ValueError, match=r"POPULATION_INDICATOR_DEFINITIONS\.POPULATION is not configured"):
        validate_config(config)


def test_validate_config_all_errors():
    """Test that all validation error categories are reported together in a single ValueError."""
    config = {
        "SNT_CONFIG": {},
        "SNT_DATASET_IDENTIFIERS": {},
        "DHIS2_DATA_DEFINITIONS": {
            "POPULATION_INDICATOR_DEFINITIONS": {},
            "DHIS2_INDICATOR_DEFINITIONS": {},
        },
    }

    expected_errors = [
        "SNT_CONFIG.COUNTRY_CODE is missing or empty",
        "SNT_CONFIG.DHIS2_ADMINISTRATION_1 is missing or empty",
        "SNT_CONFIG.DHIS2_ADMINISTRATION_2 is missing or empty",
        "SNT_CONFIG.ANALYTICS_ORG_UNITS_LEVEL is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_EXTRACTS is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_DATASET_FORMATTED is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_POPULATION_TRANSFORMATION is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_REPORTING_RATE is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_OUTLIERS_IMPUTATION is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_INCIDENCE is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHS_INDICATORS is missing or empty",
        "SNT_DATASET_IDENTIFIERS.WORLDPOP_DATASET_EXTRACT is missing or empty",
        "SNT_DATASET_IDENTIFIERS.SNT_HEALTHCARE_ACCESS is missing or empty",
        "SNT_DATASET_IDENTIFIERS.ERA5_DATASET_CLIMATE is missing or empty",
        "SNT_DATASET_IDENTIFIERS.SNT_SEASONALITY_RAINFALL is missing or empty",
        "SNT_DATASET_IDENTIFIERS.SNT_SEASONALITY_CASES is missing or empty",
        "SNT_DATASET_IDENTIFIERS.SNT_MAP_EXTRACTS is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_QUALITY_OF_CARE is missing or empty",
        "SNT_DATASET_IDENTIFIERS.SNT_POPULATION_USER_PROVIDED is missing or empty",
        "DHIS2_DATA_DEFINITIONS.POPULATION_INDICATOR_DEFINITIONS is missing or empty",
        "DHIS2_DATA_DEFINITIONS.DHIS2_INDICATOR_DEFINITIONS has no indicators defined",
    ]

    with pytest.raises(ValueError, match="Configuration validation failed") as exc_info:
        validate_config(config)

    error_message = str(exc_info.value)
    assert f"Configuration validation failed with {len(expected_errors)} error(s)" in error_message
    for expected in expected_errors:
        assert expected in error_message, f"Missing expected error: {expected}"


def test_delete_raw_files(tmp_path: Path):
    """Test that only files matching the pattern are deleted."""
    to_delete = [
        tmp_path / "COD_routine_data_202501.parquet",
        tmp_path / "COD_routine_data_202502.parquet",
        tmp_path / "COD_routine_data_202503.parquet",
    ]
    to_keep = [
        tmp_path / "COD_routine_data_202501.csv",
        tmp_path / "COD_summary_202501.parquet",
    ]
    for f in to_delete + to_keep:
        f.touch()

    delete_raw_files(tmp_path, "*_routine_data_*.parquet")

    assert all(not f.exists() for f in to_delete)
    assert all(f.exists() for f in to_keep)


def _make_version(filenames: list[str]) -> SimpleNamespace:
    files = [SimpleNamespace(filename=f) for f in filenames]
    return SimpleNamespace(files=files)


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_matching_filename_returns_all_matches(mock_workspace: MagicMock) -> None:
    """Returns every filename that matches the glob pattern."""
    mock_workspace.get_dataset.return_value.latest_version = _make_version(
        ["COD_data_202501.parquet", "COD_data_202502.parquet", "COD_summary.parquet"]
    )
    result = get_matching_filename_from_dataset_last_version("ds-id", "COD_data_*.parquet")
    assert result == ["COD_data_202501.parquet", "COD_data_202502.parquet"]


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_matching_filename_no_match_returns_empty(mock_workspace: MagicMock) -> None:
    """Returns an empty list when no files match the pattern."""
    mock_workspace.get_dataset.return_value.latest_version = _make_version(["other_file.csv"])
    result = get_matching_filename_from_dataset_last_version("ds-id", "COD_*.parquet")
    assert result == []


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_matching_filename_dataset_not_found(mock_workspace: MagicMock) -> None:
    """Raises ValueError when the dataset does not exist."""
    mock_workspace.get_dataset.return_value = None
    with pytest.raises(ValueError, match="not found"):
        get_matching_filename_from_dataset_last_version("bad-id", "*.parquet")


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_matching_filename_no_version(mock_workspace: MagicMock) -> None:
    """Raises ValueError when the dataset has no versions."""
    mock_workspace.get_dataset.return_value.latest_version = None
    with pytest.raises(ValueError, match="No versions found"):
        get_matching_filename_from_dataset_last_version("ds-id", "*.parquet")


def _make_r_notebook(language_name: str) -> nbformat.NotebookNode:
    """Build a minimal R notebook whose language_info.name is either 'r' or 'R'.

    Returns:
        nbformat.NotebookNode: A notebook with a single parameters cell tagged with 'parameters'.
    """
    nb = nbformat.v4.new_notebook()
    nb.metadata["language_info"] = {"name": language_name}
    nb.metadata["kernelspec"] = {"name": "ir", "language": language_name, "display_name": "R"}
    parameters_cell = nbformat.v4.new_code_cell(source="x <- 1")
    parameters_cell.metadata["tags"] = ["parameters"]
    nb.cells = [parameters_cell]
    return nb


@pytest.mark.parametrize("language_name", ["r", "R"])
def test_ir_kernel_translator_registered_for_lowercase_and_uppercase_r(tmp_path: Path, language_name: str) -> None:
    """Test that notebooks reporting language 'r' or 'R' can be parameterized with kernel_name="ir".

    Papermill only ships a built-in translator for language 'R' (capital), but run_notebook and
    run_report_notebook default to kernel_name="ir". Without registering "ir" as a translator
    alias, parameterizing a notebook whose language_info.name is lowercase 'r' raises
    PapermillException("No parameter translator functions specified for kernel 'ir' or language 'r'").
    """
    nb_path = tmp_path / f"notebook_{language_name}.ipynb"
    out_path = tmp_path / f"output_{language_name}.ipynb"
    nbformat.write(_make_r_notebook(language_name), nb_path)

    pm.execute_notebook(
        input_path=nb_path,
        output_path=out_path,
        parameters={"x": 2},
        kernel_name="ir",
        prepare_only=True,
        progress_bar=False,
    )

    executed = nbformat.read(out_path, as_version=4)
    injected = [c for c in executed.cells if "injected-parameters" in c.metadata.get("tags", [])]
    assert injected, "expected an injected-parameters cell"
    assert "x = 2" in injected[0].source


def test_save_pipeline_parameters(tmp_path: Path) -> None:
    """Test that save_pipeline_parameters returns a single Path to an existing JSON file."""
    result = save_pipeline_parameters(
        pipeline_name="snt_dhis2_incidence",
        parameters={"n1_method": "PRES", "routine_data_choice": "imputed"},
        output_path=tmp_path,
        country_code="COD",
        extra_metadata={"input_file": "COD_routine_imputed.parquet"},
    )

    assert isinstance(result, Path)
    assert result.exists()
    assert result == tmp_path / "COD_parameters.json"


@pytest.mark.parametrize(
    ("code_scripts", "report_scripts", "expected_utils"),
    [
        (None, None, []),
        (["my_pipeline.ipynb"], None, ["my_pipeline.r"]),
        (None, ["my_pipeline_report.ipynb"], ["my_pipeline_report.r"]),
        (["my_pipeline.ipynb"], ["my_pipeline_report.ipynb"], ["my_pipeline.r", "my_pipeline_report.r"]),
    ],
)
@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.load_scripts_for_pipeline")
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_pull_scripts_from_repository_utils_follow_scripts(
    mock_workspace: MagicMock,
    mock_load_scripts: MagicMock,
    tmp_path: Path,
    code_scripts: list[str] | None,
    report_scripts: list[str] | None,
    expected_utils: list[str],
) -> None:
    """Each pipeline util file is pulled only when its matching code or report scripts are requested."""
    mock_workspace.files_path = str(tmp_path)
    repo_path = tmp_path / "repo"
    pipeline_parent_folder = tmp_path / "pipelines"

    pull_scripts_from_repository(
        "my_pipeline",
        report_scripts=report_scripts,
        code_scripts=code_scripts,
        repo_path=repo_path,
        pipeline_parent_folder=pipeline_parent_folder,
    )

    snt_script_paths = mock_load_scripts.call_args.kwargs["snt_script_paths"]
    source = repo_path / "snt_development"
    pipeline_source = source / "pipelines" / "my_pipeline"
    target = pipeline_parent_folder / "my_pipeline"
    expected = {pipeline_source / "code" / c: target / "code" / c for c in code_scripts or []}
    expected |= {pipeline_source / "reporting" / r: target / "reporting" / r for r in report_scripts or []}
    expected |= {pipeline_source / "utils" / u: target / "utils" / u for u in expected_utils}
    expected |= {source / "code" / f: tmp_path / "code" / f for f in ("snt_palettes.r", "snt_report.r", "snt_utils.r")}
    assert snt_script_paths == expected


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
def test_check_outputs_generated(tmp_path: Path) -> None:
    """Fresh files pass; missing or stale (modified before the run started) files raise RuntimeError."""
    run_start_ts = time.time()
    fresh = tmp_path / "fresh.parquet"
    stale = tmp_path / "stale.parquet"
    missing = tmp_path / "missing.parquet"
    fresh.touch()
    stale.touch()
    # Set mtimes explicitly: filesystem timestamp granularity can make a just-touched file look older than time.time()
    os.utime(fresh, (run_start_ts + 1, run_start_ts + 1))
    os.utime(stale, (run_start_ts - 60, run_start_ts - 60))

    check_outputs_generated([fresh], run_start_ts)  # should not raise

    with pytest.raises(RuntimeError, match=r"stale\.parquet, missing\.parquet") as exc_info:
        check_outputs_generated([fresh, stale, missing], run_start_ts)
    assert "fresh.parquet" not in str(exc_info.value)


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.get_repository")
def test_load_scripts_for_pipeline(mock_get_repository: MagicMock, tmp_path: Path) -> None:
    """Existing scripts are copied to their targets; missing scripts are skipped."""
    source = tmp_path / "snt_development" / "code"
    source.mkdir(parents=True)
    (source / "a.r").write_text("x <- 1")
    target = tmp_path / "workspace" / "code" / "a.r"
    missing_target = tmp_path / "workspace" / "code" / "missing.r"

    load_scripts_for_pipeline(
        {Path("code/a.r"): target, Path("code/missing.r"): missing_target},
        repository_path=tmp_path,
    )

    mock_get_repository.assert_called_once_with(local_repo_path=tmp_path, repo_name="snt_development")
    assert target.read_text() == "x <- 1"
    assert not missing_target.exists()


def test_force_remove_readonly(tmp_path: Path) -> None:
    """A read-only file is made writable and removed by the retried function."""
    file = tmp_path / "readonly.txt"
    file.touch()
    file.chmod(0o444)

    force_remove_readonly(os.unlink, file, None)

    assert not file.exists()


def test_safe_rmtree(tmp_path: Path) -> None:
    """The directory tree is removed, and a non-existent path is ignored."""
    tree = tmp_path / "tree"
    (tree / "sub").mkdir(parents=True)
    (tree / "sub" / "file.txt").touch()

    snt_pipeline_utils._safe_rmtree(tree)
    snt_pipeline_utils._safe_rmtree(tmp_path / "does_not_exist")  # should not raise

    assert not tree.exists()


@pytest.mark.parametrize(
    ("token", "expected_url"),
    [
        (None, "https://github.com/BLSQ/my_repo.git"),
        ("abc", "https://abc:x-oauth-basic@github.com/BLSQ/my_repo.git"),
    ],
)
@patch("snt_lib.snt_pipeline_utils.Repo")
def test_clone_repository(mock_repo: MagicMock, tmp_path: Path, token: str | None, expected_url: str) -> None:
    """The clone URL includes the token only when one is given."""
    clone_repository("BLSQ", "my_repo", tmp_path, token=token)
    mock_repo.clone_from.assert_called_once_with(url=expected_url, to_path=tmp_path, depth=1)


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.clone_repository")
def test_get_repository_replaces_previous_clone(mock_clone: MagicMock, tmp_path: Path) -> None:
    """A previous clone is removed before cloning into local_repo_path / repo_name."""
    previous_clone = tmp_path / "my_repo"
    previous_clone.mkdir()
    (previous_clone / "old.txt").touch()

    get_repository(tmp_path, repo_name="my_repo")

    assert not previous_clone.exists()
    mock_clone.assert_called_once_with(repo_owner="BLSQ", repo_name="my_repo", dest_path=previous_clone, token=None)


@pytest.mark.parametrize("country_notebook_exists", [True, False])
@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.pm")
def test_run_notebook_selects_country_notebook(
    mock_pm: MagicMock, tmp_path: Path, country_notebook_exists: bool
) -> None:
    """The country-specific notebook is executed when it exists, otherwise the default one."""
    nb_path = tmp_path / "pipeline.ipynb"
    nb_path.touch()
    country_nb_path = tmp_path / "pipeline_NER.ipynb"
    if country_notebook_exists:
        country_nb_path.touch()
    out_path = tmp_path / "output"

    run_notebook(nb_path, out_path, parameters={"x": 1}, country_code="NER")

    expected = country_nb_path if country_notebook_exists else nb_path
    assert mock_pm.execute_notebook.call_args.kwargs["input_path"] == expected
    assert mock_pm.execute_notebook.call_args.kwargs["parameters"] == {"x": 1}
    assert out_path.is_dir()


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.generate_html_report")
@patch("snt_lib.snt_pipeline_utils.pm")
def test_run_report_notebook(mock_pm: MagicMock, mock_html: MagicMock, tmp_path: Path) -> None:
    """The notebook is executed and converted to HTML; nothing runs when ready is False."""
    nb_file = tmp_path / "report.ipynb"
    nb_file.touch()

    run_report_notebook(nb_file, tmp_path / "output", ready=False)
    mock_pm.execute_notebook.assert_not_called()

    run_report_notebook(nb_file, tmp_path / "output")
    output_path = mock_pm.execute_notebook.call_args.kwargs["output_path"]
    mock_html.assert_called_once_with(output_path)


@patch("snt_lib.snt_pipeline_utils.current_run")
@patch("snt_lib.snt_pipeline_utils.subprocess")
def test_generate_html_report(mock_subprocess: MagicMock, mock_current_run: MagicMock, tmp_path: Path) -> None:
    """Calls nbconvert on the notebook and registers the HTML file as a run output."""
    notebook = tmp_path / "report.ipynb"
    notebook.touch()

    generate_html_report(notebook)

    cmd = mock_subprocess.run.call_args.args[0]
    assert cmd[:2] == ["jupyter", "nbconvert"]
    assert cmd[-1] == str(notebook)
    mock_current_run.add_file_output.assert_called_once_with((tmp_path / "report.html").as_posix())


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
def test_generate_html_report_invalid_path(tmp_path: Path) -> None:
    """A path that is not an existing .ipynb file raises RuntimeError."""
    with pytest.raises(RuntimeError, match="Invalid notebook path"):
        generate_html_report(tmp_path / "missing.ipynb")


@patch("snt_lib.snt_pipeline_utils.current_run")
def test_handle_rkernel_error_with_labels(mock_current_run: MagicMock) -> None:
    """Warning labels are logged, error labels and unlabelled errors raise RuntimeError."""
    labels = {"[WARNING]": "warning", "[ERROR]": "error"}

    handle_rkernel_error_with_labels(Exception("Error: [WARNING] Low data"), labels)
    mock_current_run.log_warning.assert_called_once_with("Low data")

    with pytest.raises(RuntimeError, match="Bad data details here"):
        handle_rkernel_error_with_labels(Exception("Error: [ERROR] Bad data [ERROR DETAILS] details here"), labels)

    with pytest.raises(RuntimeError, match="Unexpected failure"):
        handle_rkernel_error_with_labels(Exception("Unexpected failure"), labels)


def test_write_file_to_tmp(tmp_path: Path) -> None:
    """A supported file is copied to a temp file with the same extension and content."""
    src = tmp_path / "data.csv"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_csv(src, index=False)

    tmp_file = Path(snt_pipeline_utils._write_file_to_tmp(src))
    try:
        assert tmp_file.suffix == ".csv"
        pd.testing.assert_frame_equal(pd.read_csv(tmp_file), pd.read_csv(src))
    finally:
        tmp_file.unlink()


def test_write_file_to_tmp_unsupported_format(tmp_path: Path) -> None:
    """An unsupported file extension raises ValueError."""
    src = tmp_path / "data.txt"
    src.touch()
    with pytest.raises(ValueError, match="Unsupported file format"):
        snt_pipeline_utils._write_file_to_tmp(src)


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.get_new_dataset_version")
def test_add_files_to_dataset(mock_new_version: MagicMock, tmp_path: Path) -> None:
    """Existing files are added to a single new version; missing files are skipped."""
    src = tmp_path / "COD_data.csv"
    pd.DataFrame({"a": [1]}).to_csv(src, index=False)

    result = add_files_to_dataset("ds-id", "COD", [src, tmp_path / "missing.csv"])

    assert result is True
    mock_new_version.assert_called_once_with(ds_id="ds-id", prefix="SNT_COD")
    add_file = mock_new_version.return_value.add_file
    add_file.assert_called_once()
    assert add_file.call_args.kwargs["filename"] == "COD_data.csv"
    Path(add_file.call_args.args[0]).unlink()  # clean up the temp file


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.get_new_dataset_version")
def test_add_files_to_dataset_no_valid_files(mock_new_version: MagicMock, tmp_path: Path) -> None:
    """No version is created and False is returned when no file can be added."""
    assert add_files_to_dataset("ds-id", "COD", [tmp_path / "missing.csv"]) is False
    mock_new_version.assert_not_called()


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_new_dataset_version(mock_workspace: MagicMock) -> None:
    """A version named after the prefix is created on the existing dataset."""
    dataset = mock_workspace.get_dataset.return_value

    result = get_new_dataset_version("ds-id", prefix="SNT_COD")

    assert result is dataset.create_version.return_value
    assert dataset.create_version.call_args.args[0].startswith("SNT_COD_")
    mock_workspace.create_dataset.assert_not_called()


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_new_dataset_version_creates_missing_dataset(mock_workspace: MagicMock) -> None:
    """The dataset is created when it does not exist yet."""
    mock_workspace.get_dataset.return_value = None

    get_new_dataset_version("my-ds")

    mock_workspace.create_dataset.assert_called_once_with(name="MY_DS", description="SNT Process dataset")


def test_remove_all_files(tmp_path: Path) -> None:
    """Files are removed while subfolders are kept; an invalid directory raises ValueError."""
    (tmp_path / "a.txt").touch()
    (tmp_path / "b.parquet").touch()
    (tmp_path / "sub").mkdir()

    remove_all_files(str(tmp_path))

    assert [p.name for p in tmp_path.iterdir()] == ["sub"]
    with pytest.raises(ValueError, match="not a valid directory"):
        remove_all_files(str(tmp_path / "missing"))


@patch("snt_lib.snt_pipeline_utils.requests")
@patch("snt_lib.snt_pipeline_utils.workspace")
def test_get_file_from_dataset_csv(mock_workspace: MagicMock, mock_requests: MagicMock) -> None:
    """A CSV file is downloaded from the latest dataset version and returned as a DataFrame."""
    expected = pd.DataFrame({"org_unit": [f"ou_{i}" for i in range(20)], "value": range(20)})
    mock_requests.get.return_value = SimpleNamespace(status_code=200, content=expected.to_csv(index=False).encode())

    result = get_file_from_dataset("ds-id", "data.csv")

    mock_workspace.get_dataset.return_value.latest_version.get_file.assert_called_once_with("data.csv")
    pd.testing.assert_frame_equal(result, expected)


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
def test_copy_file(tmp_path: Path) -> None:
    """The file is copied into a newly created destination folder; a missing source raises."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "data.json").write_text('{"a": 1}')
    destination = tmp_path / "destination" / "nested"

    copy_file(source, destination, "data.json")

    assert (destination / "data.json").read_text() == '{"a": 1}'
    with pytest.raises(FileNotFoundError):
        copy_file(source, destination, "missing.json")


@patch("snt_lib.snt_pipeline_utils.workspace")
def test_dataset_file_exists(mock_workspace: MagicMock) -> None:
    """Returns True only for files present in the latest dataset version."""
    mock_workspace.get_dataset.return_value.latest_version = _make_version(["COD_data.parquet"])

    assert dataset_file_exists("ds-id", "COD_data.parquet") is True
    assert dataset_file_exists("ds-id", "other.parquet") is False


@patch("snt_lib.snt_pipeline_utils.current_run", MagicMock())
def test_push_data_to_db_table(tmp_path: Path) -> None:
    """The DataFrame is written to the database table; an empty table name raises ValueError."""
    db_url = f"sqlite:///{tmp_path / 'test.db'}"
    df = pd.DataFrame({"org_unit": ["a", "b"], "value": [1, 2]})

    push_data_to_db_table("my_table", dataframe=df, db_url=db_url)

    pd.testing.assert_frame_equal(pd.read_sql_table("my_table", create_engine(db_url)), df)
    with pytest.raises(ValueError, match="cannot be empty"):
        push_data_to_db_table("", dataframe=df, db_url=db_url)
