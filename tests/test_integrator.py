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
from pathlib import Path

import numpy as np
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
