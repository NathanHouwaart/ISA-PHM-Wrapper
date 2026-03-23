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

import pytest

from isa_phm.errors import (
    AmbiguousRunError,
    AssayNotFoundError,
    StudyNotFoundError,
    RunNotFoundError,
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
