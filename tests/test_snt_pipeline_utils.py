from pathlib import Path

import pytest

from snt_lib.snt_pipeline_utils import delete_raw_files, load_configuration_snt, save_pipeline_parameters, validate_config

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
        "SNT_DATASET_IDENTIFIERS.SNT_RESULTS is missing or empty",
        "SNT_DATASET_IDENTIFIERS.DHIS2_QUALITY_OF_CARE is missing or empty",
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
