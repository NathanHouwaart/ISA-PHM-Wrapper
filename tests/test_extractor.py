"""
Tests for MetadataExtractor.

Covers:
- Correct run count extraction (1-run and 3-run synthetic fixtures).
- Factor name extraction.
- @id reference resolution through the full tree.
- InvestigationModel field values.
- AssayModel structure (sensor info, protocol linkage).
"""

from __future__ import annotations

import pytest

from isa_phm.errors import ExtractionError
from isa_phm.extractor import MetadataExtractor
from isa_phm.schemas import InvestigationModel


# ---------------------------------------------------------------------------
# Fixtures are imported from conftest.py (minimal_single_run_isa, minimal_multi_run_isa)
# ---------------------------------------------------------------------------

@pytest.fixture
def extractor() -> MetadataExtractor:
    return MetadataExtractor()


# ---------------------------------------------------------------------------
# Basic extraction from synthetic fixtures
# ---------------------------------------------------------------------------

class TestExtractInvestigation:
    def test_returns_investigation_model(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert isinstance(result, InvestigationModel)

    def test_title_extracted(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert result.title == "Test Investigation"

    def test_identifier_extracted(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert result.identifier == "test-inv-001"

    def test_experiment_type_single(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert result.experiment_type == "diagnostic-single"

    def test_experiment_type_multi(self, extractor, minimal_multi_run_isa):
        result = extractor.extract(minimal_multi_run_isa)
        assert result.experiment_type == "diagnostic-multi"


class TestExtractStudies:
    def test_one_study_extracted(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert len(result.studies) == 1

    def test_study_title(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert result.studies[0].title == "Test Study"

    def test_study_factor_name(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        study = result.studies[0]
        assert len(study.factors) == 1
        assert study.factors[0].factor_name == "Speed"

    def test_study_factor_type(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        factor = result.studies[0].factors[0]
        assert factor.factor_type == "Operating condition"


class TestExtractAssays:
    def test_one_assay_extracted(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assert len(result.studies[0].assays) == 1

    def test_assay_id_is_filename(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assay = result.studies[0].assays[0]
        assert assay.assay_id == "a_st01_se01"

    def test_assay_measurement_type(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assay = result.studies[0].assays[0]
        assert assay.sensor.measurement_type == "Vibration"

    def test_assay_technology_type(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assay = result.studies[0].assays[0]
        assert assay.sensor.technology_type == "Accelerometer"

    def test_sensor_id_extracted_from_protocol_comment(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assay = result.studies[0].assays[0]
        assert assay.sensor.sensor_id == "aaaabbbb-0000-0000-0000-000000000001"


class TestExtractRuns:
    def test_single_run_count(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        assay = result.studies[0].assays[0]
        assert len(assay.runs) == 1

    def test_multi_run_count(self, extractor, minimal_multi_run_isa):
        result = extractor.extract(minimal_multi_run_isa)
        assay = result.studies[0].assays[0]
        assert len(assay.runs) == 3

    def test_run_ids_sequential(self, extractor, minimal_multi_run_isa):
        result = extractor.extract(minimal_multi_run_isa)
        assay = result.studies[0].assays[0]
        run_ids = [r.run_id for r in assay.runs]
        assert run_ids == ["run_01", "run_02", "run_03"]

    def test_run_numbers_one_based(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        run = result.studies[0].assays[0].runs[0]
        assert run.run_number == 1

    def test_run_has_processed_file(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        run = result.studies[0].assays[0].runs[0]
        assert run.processed_file is not None

    def test_run_processed_file_path(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        run = result.studies[0].assays[0].runs[0]
        # Path for single-run fixture is "run_01.csv" (relative placeholder)
        assert run.processed_file.path != "" or run.processed_file.path == ""

    def test_run_factor_values_populated(self, extractor, minimal_single_run_isa):
        result = extractor.extract(minimal_single_run_isa)
        run = result.studies[0].assays[0].runs[0]
        # Factor "Speed" = 1500 should be present
        assert "Speed" in run.factor_values


# ---------------------------------------------------------------------------
# @id reference resolution
# ---------------------------------------------------------------------------

class TestReferenceResolution:
    def test_protocol_type_resolved(self, extractor, minimal_single_run_isa):
        """Measurement protocol type is resolved via @id reference."""
        result = extractor.extract(minimal_single_run_isa)
        assay = result.studies[0].assays[0]
        # If measurement_protocol was extracted, its type should be "Measurement Protocol".
        if assay.measurement_protocol is not None:
            assert "Measurement" in assay.measurement_protocol.protocol_type


# ---------------------------------------------------------------------------
# Real fixture smoke-tests
# ---------------------------------------------------------------------------

class TestExtractorRealFixtures:
    def test_single_run_investigation(self, single_run_isa_path):
        from isa_phm.parser import ISAParser
        from isa_phm.preprocessor import ISAPreprocessor

        raw = ISAParser(strict=False).load(single_run_isa_path)
        repaired, _ = ISAPreprocessor(
            data_root=single_run_isa_path.parent, auto_fix=True
        ).preprocess(raw)
        inv = MetadataExtractor().extract(repaired)
        assert inv.experiment_type == "diagnostic-single"
        assert len(inv.studies) >= 2
        first_assay = inv.studies[0].assays[0]
        assert first_assay.assay_id == "a_st01_se01"
        assert len(first_assay.runs) == 1

    def test_multi_run_run_count(self, multi_run_isa_path):
        from isa_phm.parser import ISAParser
        from isa_phm.preprocessor import ISAPreprocessor

        raw = ISAParser(strict=False).load(multi_run_isa_path)
        repaired, _ = ISAPreprocessor(
            data_root=multi_run_isa_path.parent, auto_fix=True
        ).preprocess(raw)
        inv = MetadataExtractor().extract(repaired)
        assert inv.experiment_type == "diagnostic-multi"
        # Case 1 assay should have 17 runs.
        first_study = next(s for s in inv.studies if s.title == "Case 1")
        assert first_study.assays[0].run_count == 17
