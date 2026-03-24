"""
Tests for DataIntegrator.

Covers:
- CSV loading (time, value columns).
- Metadata columns attached after load.
- FIFO cache hit on second call.
- Missing file → DataFileError.
- AmbiguousRunError for multi-run assay with run_id=None.
- Streaming lifecycle features: generator yields correct keys.
"""

from __future__ import annotations

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from isa_phm.errors import AmbiguousRunError, DataFileError
from isa_phm.extractor import MetadataExtractor
from isa_phm.integrator import DataIntegrator
from isa_phm.parser import ISAParser
from isa_phm.preprocessor import ISAPreprocessor
from isa_phm.utils import FEATURE_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_integrator(tmp_path: Path) -> DataIntegrator:
    return DataIntegrator(data_root=tmp_path, cache_maxsize=5)


def _make_investigation(isa_dict: dict, data_root: Path):
    """Parse → preprocess → extract an ISA dict and return InvestigationModel."""
    preprocessor = ISAPreprocessor(data_root=data_root, auto_fix=True)
    repaired, _ = preprocessor.preprocess(isa_dict)
    return MetadataExtractor().extract(repaired)


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

class TestCSVLoading:
    def test_basic_csv_loaded(self, tmp_csv, tmp_path):
        integrator = _build_integrator(tmp_path)
        df = integrator._read_csv(tmp_csv)
        assert list(df.columns) == ["time", "value"]
        assert len(df) > 0

    def test_missing_file_raises(self, tmp_path):
        integrator = _build_integrator(tmp_path)
        with pytest.raises(DataFileError, match="not found"):
            integrator._read_csv(tmp_path / "does_not_exist.csv")

    def test_empty_file_raises(self, tmp_path):
        p = tmp_path / "empty.csv"
        p.write_text("", encoding="utf-8")
        integrator = _build_integrator(tmp_path)
        with pytest.raises(DataFileError):
            integrator._read_csv(p)

    def test_tab_delimited_csv_loaded(self, tmp_path):
        p = tmp_path / "tab.csv"
        with p.open("w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            for i in range(100):
                writer.writerow([i * 0.001, float(i)])
        integrator = _build_integrator(tmp_path)
        df = integrator._read_csv(p)
        assert list(df.columns) == ["time", "value"]
        assert len(df) == 100

    def test_bad_csv_row_raises_instead_of_silent_drop(self, tmp_path):
        p = tmp_path / "bad_rows.csv"
        p.write_text("0,1\n1,2,3\n2,3\n", encoding="utf-8")
        integrator = _build_integrator(tmp_path)
        with pytest.raises(DataFileError, match="Failed to read|Cannot decode"):
            integrator._read_csv(p)

    def test_latin1_config_selection_is_logged(self, tmp_path, caplog):
        p = tmp_path / "latin1.csv"
        # First row is a non-numeric header-like row with Latin-1 byte 0xE9.
        p.write_bytes("t,\xe9\n0,1\n1,2\n".encode("latin-1"))
        integrator = _build_integrator(tmp_path)
        with caplog.at_level("INFO", logger="isa_phm"):
            df = integrator._read_csv(p)
        assert list(df.columns) == ["time", "value"]
        assert len(df) == 2
        assert "encoding=latin-1" in caplog.text


# ---------------------------------------------------------------------------
# DataFrame columns
# ---------------------------------------------------------------------------

class TestDataFrameColumns:
    def test_only_time_and_value_columns(
        self, minimal_single_run_isa_file, tmp_path, tmp_csv
    ):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay = study.assays[0]

        integrator = _build_integrator(tmp_path)
        df = integrator.load(assay, study_id=study.study_id)

        assert list(df.columns) == ["time", "value"], (
            f"Expected only [time, value], got {list(df.columns)}"
        )


# ---------------------------------------------------------------------------
# Cache behaviour
# ---------------------------------------------------------------------------

class TestCache:
    def test_cache_hit_on_second_call(
        self, minimal_single_run_isa_file, tmp_path, tmp_csv
    ):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_path)

        df1 = integrator.load(assay, study_id=study.study_id)
        df2 = integrator.load(assay, study_id=study.study_id)
        # Same object returned from cache.
        assert df1 is df2

    def test_clear_cache(self, minimal_single_run_isa_file, tmp_path):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_path)

        integrator.load(assay, study_id=study.study_id)
        assert len(integrator._cache) == 1
        integrator.clear_cache()
        assert len(integrator._cache) == 0

    def test_fifo_eviction(self, tmp_path):
        """Third entry evicts the first when maxsize=2."""
        integrator = DataIntegrator(data_root=tmp_path, cache_maxsize=2)

        # Inject dummy DataFrames directly into cache.
        import pandas as pd

        integrator._cache.put(("a", "run_01", "processed"), pd.DataFrame())
        integrator._cache.put(("b", "run_01", "processed"), pd.DataFrame())
        # Now cache is full (2 entries).
        integrator._cache.put(("c", "run_01", "processed"), pd.DataFrame())
        # First entry should have been evicted.
        assert integrator._cache.get(("a", "run_01", "processed")) is None
        assert integrator._cache.get(("c", "run_01", "processed")) is not None


# ---------------------------------------------------------------------------
# AmbiguousRunError
# ---------------------------------------------------------------------------

class TestAmbiguousRun:
    def test_multi_run_without_run_id_raises(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        raw = ISAParser(strict=False).load(minimal_multi_run_isa_file)
        inv = _make_investigation(raw, tmp_csv_dir)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_csv_dir)

        with pytest.raises(AmbiguousRunError):
            integrator.load(assay, study_id=study.study_id, run_id=None)


class TestNormalizeTimeGuards:
    def test_empty_dataframe_raises_clear_datafile_error(
        self, minimal_single_run_isa_file, tmp_path
    ):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        inv = _make_investigation(raw, tmp_path)
        run = inv.studies[0].assays[0].runs[0]
        with pytest.raises(DataFileError, match="Cannot normalize time"):
            DataIntegrator._normalize_time(pd.DataFrame({"time": []}), run)


# ---------------------------------------------------------------------------
# Lifecycle streaming
# ---------------------------------------------------------------------------

class TestLifecycleStreaming:
    def test_yields_dicts_with_feature_keys(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        raw = ISAParser(strict=False).load(minimal_multi_run_isa_file)
        inv = _make_investigation(raw, tmp_csv_dir)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_csv_dir)

        rows = list(
            integrator.stream_lifecycle_features(assay, study_id=study.study_id)
        )
        assert len(rows) == 3
        for row in rows:
            for feat in FEATURE_NAMES:
                assert feat in row, f"Feature '{feat}' missing from row"

    def test_lifecycle_df_has_correct_n_rows(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        raw = ISAParser(strict=False).load(minimal_multi_run_isa_file)
        inv = _make_investigation(raw, tmp_csv_dir)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_csv_dir)

        df = integrator.lifecycle_features_df(assay, study_id=study.study_id)
        assert len(df) == 3

    def test_lifecycle_df_has_run_id_column(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        raw = ISAParser(strict=False).load(minimal_multi_run_isa_file)
        inv = _make_investigation(raw, tmp_csv_dir)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_csv_dir)

        df = integrator.lifecycle_features_df(assay, study_id=study.study_id)
        assert "run_id" in df.columns


# ---------------------------------------------------------------------------
# file_type contract + metadata
# ---------------------------------------------------------------------------

class TestFileTypeContract:
    def test_invalid_file_type_raises(self, minimal_single_run_isa_file, tmp_path):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_path)

        with pytest.raises(DataFileError, match="Invalid file_type"):
            integrator.load(assay, study_id=study.study_id, file_type="banana")

    def test_auto_reports_processed_when_available(
        self, minimal_single_run_isa_file, tmp_path
    ):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_path)

        df, meta = integrator.load_with_meta(
            assay,
            study_id=study.study_id,
            file_type="auto",
        )
        assert list(df.columns) == ["time", "value"]
        assert meta.requested_file_type == "auto"
        assert meta.resolved_file_type == "processed"
        assert meta.from_cache is False

    def test_auto_falls_back_to_raw_and_reports_resolved_type(
        self, minimal_single_run_isa_file, tmp_csv, tmp_path
    ):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        assay = raw["studies"][0]["assays"][0]
        for data_file in assay["dataFiles"]:
            if data_file.get("type") == "Raw Data File":
                data_file["name"] = str(tmp_csv)
            elif data_file.get("type") == "Processed Data File":
                data_file["name"] = ""

        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay_model = study.assays[0]
        integrator = _build_integrator(tmp_path)

        _, meta = integrator.load_with_meta(
            assay_model,
            study_id=study.study_id,
            file_type="auto",
        )
        assert meta.requested_file_type == "auto"
        assert meta.resolved_file_type == "raw"
        assert meta.file_path == str(tmp_csv)

    def test_processed_does_not_implicitly_fallback_to_raw(
        self, minimal_single_run_isa_file, tmp_csv, tmp_path
    ):
        raw = ISAParser(strict=False).load(minimal_single_run_isa_file)
        assay = raw["studies"][0]["assays"][0]
        for data_file in assay["dataFiles"]:
            if data_file.get("type") == "Raw Data File":
                data_file["name"] = str(tmp_csv)
            elif data_file.get("type") == "Processed Data File":
                data_file["name"] = ""

        inv = _make_investigation(raw, tmp_path)
        study = inv.studies[0]
        assay_model = study.assays[0]
        integrator = _build_integrator(tmp_path)

        with pytest.raises(DataFileError, match="no 'processed' data file"):
            integrator.load(assay_model, study_id=study.study_id, file_type="processed")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------

class TestConcurrentLoads:
    def test_parallel_repeated_loads_are_stable(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        raw = ISAParser(strict=False).load(minimal_multi_run_isa_file)
        inv = _make_investigation(raw, tmp_csv_dir)
        study = inv.studies[0]
        assay = study.assays[0]
        integrator = _build_integrator(tmp_csv_dir)

        def _job() -> tuple[int, list[str], str]:
            df, meta = integrator.load_with_meta(
                assay,
                study_id=study.study_id,
                run_id="run_01",
                file_type="auto",
            )
            return len(df), list(df.columns), meta.resolved_file_type

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda _: _job(), range(32)))

        assert len(results) == 32
        assert all(r[0] > 0 for r in results)
        assert all(r[1] == ["time", "value"] for r in results)
        assert all(r[2] == "processed" for r in results)
        assert len(integrator._cache) == 1


# ---------------------------------------------------------------------------
# Large-file chunked mode
# ---------------------------------------------------------------------------

class TestLargeFileChunkedMode:
    def test_chunked_lifecycle_features_match_non_chunked(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        raw = ISAParser(strict=False).load(minimal_multi_run_isa_file)
        inv = _make_investigation(raw, tmp_csv_dir)
        study = inv.studies[0]
        assay = study.assays[0]

        baseline = DataIntegrator(
            data_root=tmp_csv_dir,
            cache_maxsize=5,
            enable_chunked_large_file_mode=False,
        ).lifecycle_features_df(assay, study_id=study.study_id, n_workers=1)

        chunked = DataIntegrator(
            data_root=tmp_csv_dir,
            cache_maxsize=5,
            enable_chunked_large_file_mode=True,
            large_file_threshold_mb=0.0,  # force chunk mode for test files
            chunk_rows=64,
        ).lifecycle_features_df(assay, study_id=study.study_id, n_workers=1)

        assert list(chunked["run_id"]) == list(baseline["run_id"])
        for feat in FEATURE_NAMES:
            np.testing.assert_allclose(
                chunked[feat].to_numpy(dtype=float),
                baseline[feat].to_numpy(dtype=float),
                rtol=1e-10,
                atol=1e-12,
            )
