"""
Tests for the proxy layer (QueryNavigator, StudyProxy, AssayProxy, RunProxy).

Covers:
- Full fluent chain on single-run fixture.
- study() lookup by title and UUID.
- AmbiguousRunError for multi-run assay without run_id.
- StudyNotFoundError on bad study_id.
- AssayNotFoundError on bad assay_id.
- RunNotFoundError on bad run_id.
- overview() and list_*() methods return correct types.
- lifecycle_features() shape.
- missing_values_report() fields.
"""

from __future__ import annotations

import copy
import json
import numpy as np
import pandas as pd
import pytest

from isa_phm.errors import (
    AmbiguousRunError,
    AssayNotFoundError,
    DataFileError,
    StudyNotFoundError,
    RunNotFoundError,
    ValidationError,
)
from isa_phm.integrator import DataIntegrator
from isa_phm.extractor import MetadataExtractor
from isa_phm.parser import ISAParser
from isa_phm.plotter import ISAPlotter
from isa_phm.preprocessor import ISAPreprocessor
from isa_phm.proxy import QueryNavigator, StudyProxy, AssayProxy, RunProxy
from isa_phm.schemas import (
    AssayOverview,
    InvestigationOverview,
    MissingValuesReport,
    RunOverview,
    StudyOverview,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_navigator(isa_file, data_root, cache_maxsize=10):
    raw = ISAParser(strict=False).load(isa_file)
    repaired, _ = ISAPreprocessor(data_root=data_root, auto_fix=True).preprocess(raw)
    inv = MetadataExtractor().extract(repaired)
    integrator = DataIntegrator(data_root=data_root, cache_maxsize=cache_maxsize)
    plotter = ISAPlotter()
    return QueryNavigator(inv, integrator, plotter), inv


# ---------------------------------------------------------------------------
# QueryNavigator
# ---------------------------------------------------------------------------

class TestQueryNavigator:
    def test_investigation_overview_returned(
        self, minimal_single_run_isa_file, tmp_path
    ):
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        ov = nav.investigation_overview()
        assert isinstance(ov, InvestigationOverview)

    def test_investigation_title(self, minimal_single_run_isa_file, tmp_path):
        nav, inv = _build_navigator(minimal_single_run_isa_file, tmp_path)
        ov = nav.investigation_overview()
        assert ov.title == inv.title

    def test_n_studies_correct(self, minimal_single_run_isa_file, tmp_path):
        nav, inv = _build_navigator(minimal_single_run_isa_file, tmp_path)
        ov = nav.investigation_overview()
        assert ov.n_studies == len(inv.studies)

    def test_list_studies_returns_list(self, minimal_single_run_isa_file, tmp_path):
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        summaries = nav.list_studies()
        assert isinstance(summaries, list)
        assert len(summaries) == 1

    def test_study_by_title(self, minimal_single_run_isa_file, tmp_path):
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        proxy = nav.study("Test Study")
        assert isinstance(proxy, StudyProxy)

    def test_study_by_title_case_insensitive(
        self, minimal_single_run_isa_file, tmp_path
    ):
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        proxy = nav.study("test study")
        assert isinstance(proxy, StudyProxy)

    def test_study_by_uuid(self, minimal_single_run_isa_file, tmp_path):
        nav, inv = _build_navigator(minimal_single_run_isa_file, tmp_path)
        uid = inv.studies[0].study_id
        proxy = nav.study(uid)
        assert isinstance(proxy, StudyProxy)

    def test_study_not_found_raises(self, minimal_single_run_isa_file, tmp_path):
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        with pytest.raises(StudyNotFoundError):
            nav.study("nonexistent study title xxxx")

    def test_duplicate_normalized_study_titles_raise_validation_error(
        self, minimal_single_run_isa, tmp_csv, tmp_path
    ):
        isa = minimal_single_run_isa
        for data_file in isa["studies"][0]["assays"][0]["dataFiles"]:
            if data_file.get("type") == "Processed Data File":
                data_file["name"] = str(tmp_csv)

        study_copy = copy.deepcopy(isa["studies"][0])
        study_copy["identifier"] = "st2-uuid"
        study_copy["title"] = "  test study  "
        isa["studies"].append(study_copy)

        p = tmp_path / "i_dup_study_title.json"
        p.write_text(json.dumps(isa), encoding="utf-8")

        with pytest.raises(ValidationError, match="Duplicate normalized study title"):
            _build_navigator(p, tmp_path)


# ---------------------------------------------------------------------------
# StudyProxy
# ---------------------------------------------------------------------------

class TestStudyProxy:
    def _study(self, minimal_single_run_isa_file, tmp_path) -> StudyProxy:
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        return nav.study("Test Study")

    def test_overview_returned(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        ov = study.overview()
        assert isinstance(ov, StudyOverview)

    def test_overview_title(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        ov = study.overview()
        assert ov.title == "Test Study"

    def test_list_assays(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        assays = study.list_assays()
        assert len(assays) == 1
        assert assays[0].assay_id == "a_st01_se01"

    def test_list_factors(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        factors = study.list_factors()
        assert len(factors) == 1
        assert factors[0].factor_name == "Speed"

    def test_test_matrix_pivot_structure(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        df = study.test_matrix()
        # Must have the three fixed columns plus one value column
        assert list(df.columns[:3]) == ["variable", "type", "unit"]
        assert len(df.columns) == 4  # + "Value" (single condition)
        assert len(df) == 1          # one factor in the fixture
        assert df["variable"].iloc[0] == "Speed"

    def test_operating_conditions_filter(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        df = study.operating_conditions()
        # The fixture factor type is "Operating condition" → should appear here
        assert len(df) == 1
        assert df["variable"].iloc[0] == "Speed"

    def test_fault_conditions_filter(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        df = study.fault_conditions()
        # No fault factors in the fixture → empty DataFrame with correct columns
        assert len(df) == 0
        assert "variable" in df.columns

    def test_has_runs_single(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        assert study.has_runs is False  # 1 run → not multi-run

    def test_variable_overview_returns_list_of_dataframes(
        self, minimal_single_run_isa_file, tmp_path
    ):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        result = study.variable_overview()
        assert isinstance(result, list)
        assert len(result) >= 1
        assert list(result[0].columns) == ["variable", "value"]

    def test_assay_by_id(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        assay = study.assay("a_st01_se01")
        assert isinstance(assay, AssayProxy)

    def test_assay_not_found_raises(self, minimal_single_run_isa_file, tmp_path):
        study = self._study(minimal_single_run_isa_file, tmp_path)
        with pytest.raises(AssayNotFoundError):
            study.assay("nonexistent_assay")


# ---------------------------------------------------------------------------
# StudyProxy.get_fault_labels
# ---------------------------------------------------------------------------

class TestStudyProxyGetFaultLabels:
    @staticmethod
    def _study(isa_path, tmp_path):
        nav, _ = _build_navigator(isa_path, tmp_path)
        return nav.study("Test Study")

    def test_empty_when_no_fault_factors(self, minimal_single_run_isa_file, tmp_path):
        df = self._study(minimal_single_run_isa_file, tmp_path).get_fault_labels()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0
        assert list(df.columns) == ["assay_id", "run_id", "run_number"]

    def test_returns_fault_column(self, minimal_fault_isa_file, tmp_path):
        df = self._study(minimal_fault_isa_file, tmp_path).get_fault_labels()
        assert "Fault size" in df.columns
        assert {"assay_id", "run_id", "run_number"}.issubset(df.columns)
        assert len(df) == 3  # 1 assay × 3 runs
        assert df["Fault size"].notna().all()

    def test_filter_by_assay_id(self, minimal_fault_isa_file, tmp_path):
        df = self._study(minimal_fault_isa_file, tmp_path).get_fault_labels(
            assay_id="st01_se01"
        )
        assert list(df["assay_id"].unique()) == ["a_st01_se01"]
        assert len(df) == 3

    def test_run_number_is_1_based(self, minimal_fault_isa_file, tmp_path):
        df = self._study(minimal_fault_isa_file, tmp_path).get_fault_labels()
        assert list(df["run_number"]) == [1, 2, 3]

    def test_unknown_assay_id_returns_empty(self, minimal_fault_isa_file, tmp_path):
        df = self._study(minimal_fault_isa_file, tmp_path).get_fault_labels(
            assay_id="nonexistent"
        )
        assert len(df) == 0
        assert "Fault size" in df.columns


# ---------------------------------------------------------------------------
# AssayProxy — single-run
# ---------------------------------------------------------------------------

class TestAssayProxySingleRun:
    def _assay(self, minimal_single_run_isa_file, tmp_path) -> AssayProxy:
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        return nav.study("Test Study").assay("a_st01_se01")

    def test_overview_returned(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        ov = assay.overview()
        assert isinstance(ov, AssayOverview)

    def test_run_count(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        assert assay.run_count == 1

    def test_list_runs(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        runs = assay.list_runs()
        assert len(runs) == 1

    def test_load_dataframe_no_run_id(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df = assay.load_dataframe()  # No run_id — single run auto-selected.
        assert "time" in df.columns
        assert "value" in df.columns

    def test_dataframe_has_only_time_and_value(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df = assay.load_dataframe()
        assert list(df.columns) == ["time", "value"], (
            f"Expected only [time, value], got {list(df.columns)}"
        )

    def test_load_dataframe_with_meta_auto(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df, meta = assay.load_dataframe_with_meta(file_type="auto")
        assert list(df.columns) == ["time", "value"]
        assert meta.requested_file_type == "auto"
        assert meta.resolved_file_type == "processed"
        assert meta.run_id == "run_01"

    def test_invalid_file_type_raises(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        with pytest.raises(DataFileError, match="Invalid file_type"):
            assay.load_dataframe(file_type="invalid")

    def test_missing_values_report(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        report = assay.missing_values_report()
        assert isinstance(report, MissingValuesReport)
        assert report.n_rows > 0

    def test_run_navigation(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        run = assay.run("run_01")
        assert isinstance(run, RunProxy)

    def test_invalid_run_raises(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        with pytest.raises(RunNotFoundError):
            assay.run("run_99")


# ---------------------------------------------------------------------------
# AssayProxy — multi-run
# ---------------------------------------------------------------------------

class TestAssayProxyMultiRun:
    def _assay(self, minimal_multi_run_isa_file, tmp_csv_dir) -> AssayProxy:
        nav, _ = _build_navigator(minimal_multi_run_isa_file, tmp_csv_dir)
        return nav.study("Test Study").assay("a_st01_se01")

    def test_run_count_three(self, minimal_multi_run_isa_file, tmp_csv_dir):
        assay = self._assay(minimal_multi_run_isa_file, tmp_csv_dir)
        assert assay.run_count == 3

    def test_ambiguous_run_without_id_raises(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        assay = self._assay(minimal_multi_run_isa_file, tmp_csv_dir)
        with pytest.raises(AmbiguousRunError):
            assay.load_dataframe()  # No run_id with multi-run assay.

    def test_explicit_run_id_loads(self, minimal_multi_run_isa_file, tmp_csv_dir):
        assay = self._assay(minimal_multi_run_isa_file, tmp_csv_dir)
        df = assay.load_dataframe(run_id="run_01")
        assert len(df) > 0

    def test_lifecycle_features_shape(self, minimal_multi_run_isa_file, tmp_csv_dir):
        assay = self._assay(minimal_multi_run_isa_file, tmp_csv_dir)
        lc = assay.lifecycle_features()
        assert len(lc) == 3
        assert "rms" in lc.columns


# ---------------------------------------------------------------------------
# RunProxy
# ---------------------------------------------------------------------------

class TestRunProxy:
    def _run(self, minimal_single_run_isa_file, tmp_path) -> RunProxy:
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        return nav.study("Test Study").assay("a_st01_se01").run("run_01")

    def test_run_id(self, minimal_single_run_isa_file, tmp_path):
        run = self._run(minimal_single_run_isa_file, tmp_path)
        assert run.run_id == "run_01"

    def test_run_number(self, minimal_single_run_isa_file, tmp_path):
        run = self._run(minimal_single_run_isa_file, tmp_path)
        assert run.run_number == 1

    def test_overview_returned(self, minimal_single_run_isa_file, tmp_path):
        run = self._run(minimal_single_run_isa_file, tmp_path)
        ov = run.overview()
        assert isinstance(ov, RunOverview)

    def test_factor_values_returned(self, minimal_single_run_isa_file, tmp_path):
        run = self._run(minimal_single_run_isa_file, tmp_path)
        fv = run.factor_values()
        assert isinstance(fv, dict)

    def test_load_dataframe(self, minimal_single_run_isa_file, tmp_path):
        run = self._run(minimal_single_run_isa_file, tmp_path)
        df = run.load_dataframe()
        assert "time" in df.columns
        assert "value" in df.columns

    def test_load_dataframe_with_meta(self, minimal_single_run_isa_file, tmp_path):
        run = self._run(minimal_single_run_isa_file, tmp_path)
        df, meta = run.load_dataframe_with_meta(file_type="auto")
        assert list(df.columns) == ["time", "value"]
        assert meta.requested_file_type == "auto"
        assert meta.resolved_file_type == "processed"
        assert meta.run_id == "run_01"


# ---------------------------------------------------------------------------
# Proxy file_type strict/auto contract
# ---------------------------------------------------------------------------

class TestProxyFileTypeContract:
    @staticmethod
    def _raw_only_isa_file(minimal_single_run_isa, tmp_csv, tmp_path):
        isa = minimal_single_run_isa
        assay = isa["studies"][0]["assays"][0]
        for data_file in assay["dataFiles"]:
            if data_file.get("type") == "Raw Data File":
                data_file["name"] = str(tmp_csv)
            elif data_file.get("type") == "Processed Data File":
                data_file["name"] = ""
        p = tmp_path / "i_raw_only_proxy.json"
        p.write_text(json.dumps(isa), encoding="utf-8")
        return p

    def test_auto_falls_back_to_raw_via_proxy(
        self, minimal_single_run_isa, tmp_csv, tmp_path
    ):
        isa_file = self._raw_only_isa_file(minimal_single_run_isa, tmp_csv, tmp_path)
        nav, _ = _build_navigator(isa_file, tmp_path)
        assay = nav.study("Test Study").assay("a_st01_se01")

        _, meta = assay.load_dataframe_with_meta(file_type="auto")
        assert meta.requested_file_type == "auto"
        assert meta.resolved_file_type == "raw"
        assert meta.file_path == str(tmp_csv)

    def test_processed_missing_raises_via_proxy(
        self, minimal_single_run_isa, tmp_csv, tmp_path
    ):
        isa_file = self._raw_only_isa_file(minimal_single_run_isa, tmp_csv, tmp_path)
        nav, _ = _build_navigator(isa_file, tmp_path)
        assay = nav.study("Test Study").assay("a_st01_se01")

        with pytest.raises(DataFileError, match="no 'processed' data file"):
            assay.load_dataframe(file_type="processed")


# ---------------------------------------------------------------------------
# AssayProxy — fix_outliers strategies
# ---------------------------------------------------------------------------

class TestAssayProxyFixOutliersStrategies:
    """Test the interpolate, ffill, and bfill fix strategies."""

    @staticmethod
    def _outlier_df(outlier_indices: list[int], n: int = 100):
        """Return a DataFrame with obvious outliers at the given indices."""
        t = np.linspace(0, 1, n)
        v = np.ones(n, dtype=float)
        for i in outlier_indices:
            v[i] = 1_000_000.0  # extreme spike
        return pd.DataFrame({"time": t, "value": v})

    def _assay(self, minimal_single_run_isa_file, tmp_path) -> AssayProxy:
        nav, _ = _build_navigator(minimal_single_run_isa_file, tmp_path)
        return nav.study("Test Study").assay("a_st01_se01")

    def test_interpolate_no_nan_remain(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df_in = self._outlier_df(outlier_indices=[10, 50])
        result = assay.fix_outliers(df_in, method="iqr", strategy="interpolate")
        assert result["value"].isna().sum() == 0, "interpolate must not leave NaNs"

    def test_interpolate_reduces_spike(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df_in = self._outlier_df(outlier_indices=[10])
        result = assay.fix_outliers(df_in, method="iqr", strategy="interpolate")
        assert abs(result.at[10, "value"]) < 1_000, (
            "Interpolated value should be close to neighbouring values"
        )

    def test_ffill_no_nan_remain(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df_in = self._outlier_df(outlier_indices=[20, 80])
        result = assay.fix_outliers(df_in, method="iqr", strategy="ffill")
        assert result["value"].isna().sum() == 0, "ffill must not leave NaNs"

    def test_bfill_no_nan_remain(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df_in = self._outlier_df(outlier_indices=[30])
        result = assay.fix_outliers(df_in, method="iqr", strategy="bfill")
        assert result["value"].isna().sum() == 0, "bfill must not leave NaNs"

    def test_unknown_strategy_raises(self, minimal_single_run_isa_file, tmp_path):
        assay = self._assay(minimal_single_run_isa_file, tmp_path)
        df_in = self._outlier_df(outlier_indices=[5])
        with pytest.raises(ValueError, match="Unknown fix strategy"):
            assay.fix_outliers(df_in, strategy="bogus")


# ---------------------------------------------------------------------------
# AssayProxy — list_measurement_params / list_processing_params
# ---------------------------------------------------------------------------

class TestAssayProxyParamListing:
    """Tests for list_measurement_params() and list_processing_params()."""

    @staticmethod
    def _assay_from_file(isa_file: Path, tmp_path) -> AssayProxy:
        nav, _ = _build_navigator(isa_file, tmp_path)
        return nav.study("Test Study").assay("a_st01_se01")

    # -- empty case (no params defined in fixture) --------------------------

    def test_measurement_params_empty_returns_dataframe(
        self, minimal_single_run_isa_file, tmp_path
    ):
        assay = self._assay_from_file(minimal_single_run_isa_file, tmp_path)
        df = assay.list_measurement_params()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["parameter_name", "value", "unit"]
        assert len(df) == 0

    def test_processing_params_empty_returns_dataframe(
        self, minimal_single_run_isa_file, tmp_path
    ):
        assay = self._assay_from_file(minimal_single_run_isa_file, tmp_path)
        df = assay.list_processing_params()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["parameter_name", "value", "unit"]
        assert len(df) == 0

    # -- populated case (fixture with actual parameterValues) ---------------

    def test_measurement_params_populated(self, minimal_params_isa_file, tmp_path):
        assay = self._assay_from_file(minimal_params_isa_file, tmp_path)
        df = assay.list_measurement_params()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["parameter_name", "value", "unit"]
        assert len(df) == 1
        assert df.at[0, "parameter_name"] == "Sampling rate"
        assert df.at[0, "value"] == 25600

    def test_processing_params_populated(self, minimal_params_isa_file, tmp_path):
        assay = self._assay_from_file(minimal_params_isa_file, tmp_path)
        df = assay.list_processing_params()
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["parameter_name", "value", "unit"]
        assert len(df) == 1
        assert df.at[0, "parameter_name"] == "Window size"
        assert df.at[0, "value"] == 1024


class TestAssayProxyUnitInference:
    @staticmethod
    def _mutate_measurement_value(base_file, tmp_path, value) -> object:
        raw = json.loads(base_file.read_text(encoding="utf-8"))
        assay = raw["studies"][0]["assays"][0]
        for proc in assay["processSequence"]:
            inputs = proc.get("inputs", [])
            if inputs and inputs[0].get("@id", "").startswith("#sample/"):
                proc["parameterValues"][0]["value"] = value
                break
        p = tmp_path / f"i_unit_value_{str(value).replace('/', '_')}.json"
        p.write_text(json.dumps(raw), encoding="utf-8")
        return p

    def test_non_alpha_short_value_is_not_inferred_as_unit(
        self, minimal_params_isa_file, tmp_path
    ):
        isa_file = self._mutate_measurement_value(minimal_params_isa_file, tmp_path, "v2")
        nav, _ = _build_navigator(isa_file, tmp_path)
        assay = nav.study("Test Study").assay("a_st01_se01")
        assert assay._infer_unit() is None

    def test_alpha_short_value_can_be_inferred_as_unit(
        self, minimal_params_isa_file, tmp_path
    ):
        isa_file = self._mutate_measurement_value(minimal_params_isa_file, tmp_path, "nm")
        nav, _ = _build_navigator(isa_file, tmp_path)
        assay = nav.study("Test Study").assay("a_st01_se01")
        assert assay._infer_unit() == "nm"


class TestStudyProxyExportLabeledDataset:
    def test_export_labeled_dataset_reports_skipped_runs(
        self, minimal_multi_run_isa, tmp_csv_dir, tmp_path
    ):
        isa = minimal_multi_run_isa
        for data_file in isa["studies"][0]["assays"][0]["dataFiles"]:
            if (
                data_file.get("type") == "Processed Data File"
                and data_file.get("name", "").endswith("run_03.csv")
            ):
                data_file["name"] = str(tmp_csv_dir / "missing_run_03.csv")

        p = tmp_path / "i_export_missing_run.json"
        p.write_text(json.dumps(isa), encoding="utf-8")

        nav, _ = _build_navigator(p, tmp_csv_dir)
        study = nav.study("Test Study")
        df = study.export_labeled_dataset(file_type="processed")

        summary = df.attrs.get("export_summary")
        assert summary is not None
        assert summary["n_total_runs"] == 3
        assert summary["n_loaded_runs"] == 2
        assert summary["n_skipped_runs"] == 1
        assert len(summary["skipped_runs"]) == 1
