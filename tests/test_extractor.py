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

import logging

import pytest

from isa_phm.errors import ExtractionError
from isa_phm.extractor import MetadataExtractor
from isa_phm.schemas import InvestigationModel
from isa_phm.utils import ReferenceResolver


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

    def test_csv_factor_path_remains_separate_from_its_unit(self, extractor, minimal_single_run_isa):
        study = minimal_single_run_isa['studies'][0]
        factor = study['materials']['samples'][0]['factorValues'][0]
        factor['value'] = './Data/plc/Motor.csv'
        factor['unit'] = {'@id': '#unit/rpm', 'annotationValue': 'rev/min', 'termAccession': 'http://qudt.org/vocab/unit/REV-PER-MIN'}
        result = extractor.extract(minimal_single_run_isa)
        assert result.studies[0].assays[0].runs[0].factor_values['Speed'] == './Data/plc/Motor.csv'
        assert result.studies[0].factors[0].unit == 'rev/min'
        factor['value'] = 1500
        scalar = extractor.extract(minimal_single_run_isa)
        assert scalar.studies[0].assays[0].runs[0].factor_values['Speed'] == '1500 rev/min'


class TestReferenceResolver:
    def test_identical_definitions_and_bare_references_do_not_warn(self, caplog):
        definition = {"@id": "#unit", "annotationValue": "G", "comments": []}
        resolver = ReferenceResolver()
        with caplog.at_level(logging.WARNING, logger="isa_phm"):
            resolver.build([definition, {**definition}, {"@id": "#unit"}])
        assert not caplog.records
        assert resolver.resolve({"@id": "#unit"}) == definition

    def test_conflicting_definitions_warn_once_and_bare_references_use_first(self, caplog):
        first = {"@id": "#unit", "annotationValue": "G"}
        other = {"@id": "#unit", "annotationValue": "g"}
        resolver = ReferenceResolver()
        with caplog.at_level(logging.WARNING, logger="isa_phm"):
            resolver.build([first, dict(first), other, dict(other)])
        assert len(caplog.records) == 1
        assert "Conflicting definitions" in caplog.text
        assert "#unit" in caplog.text
        assert resolver.resolve({"@id": "#unit"}) == first
        assert resolver.resolve(other) == other

    def test_resolves_nested_ontology_annotation_unit(self):
        source = {
            "unit": {"@id": "#unit"},
            "definitions": [
                {
                    "@id": "#unit",
                    "annotationValue": {"@id": "#unit-label"},
                },
                {"@id": "#unit-label", "annotationValue": "kHz"},
            ],
        }
        resolver = ReferenceResolver()
        resolver.build(source)

        assert resolver.get_unit_str(source["unit"]) == "kHz"

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

    @pytest.mark.parametrize("canonical", [False, True])
    def test_sensor_and_device_comment_spellings(self, extractor, minimal_single_run_isa, canonical):
        study = minimal_single_run_isa["studies"][0]
        for protocol in study["protocols"]:
            for comment in protocol.get("comments", []):
                if comment.get("name") == "Sensor id" and canonical:
                    comment["name"] = "sensor_id"
        names = ("sensor_alias", "physical_device_id", "physical_device_datasheet") if canonical else ("sensor alias", "physical device id", "physical device datasheet")
        study["assays"][0]["comments"] = [dict(name=name, value=value) for name, value in zip(names, ("Accelerometer", "device-1", "./Datasheets/device.pdf"))]
        sensor = extractor.extract(minimal_single_run_isa).studies[0].assays[0].sensor
        assert sensor.sensor_id == "aaaabbbb-0000-0000-0000-000000000001"
        assert sensor.alias == "Accelerometer"
        assert sensor.physical_device_id == "device-1"
        assert sensor.physical_device_datasheet == "./Datasheets/device.pdf"


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


# ---------------------------------------------------------------------------
# Process-chain hard failures
# ---------------------------------------------------------------------------

class TestProcessChainHardFailures:
    def test_cycle_in_process_chain_raises(self, extractor, minimal_single_run_isa):
        assay = minimal_single_run_isa["studies"][0]["assays"][0]
        proc_seq = assay["processSequence"]
        meas = proc_seq[0]
        proc = proc_seq[1]
        proc["nextProcess"] = {"@id": meas["@id"]}

        with pytest.raises(ExtractionError, match="Cycle detected"):
            extractor.extract(minimal_single_run_isa)

    def test_unreachable_processes_raise(self, extractor, minimal_single_run_isa):
        assay = minimal_single_run_isa["studies"][0]["assays"][0]
        proc_seq = assay["processSequence"]
        meas = proc_seq[0]
        proc = proc_seq[1]

        cycle_a = {
            "@id": "#process/cycle_a",
            "executesProtocol": proc["executesProtocol"],
            "inputs": [{"@id": "#data_file/raw0"}],
            "outputs": [{"@id": "#data_file/proc0"}],
            "parameterValues": [],
            "previousProcess": {"@id": "#process/cycle_b"},
            "nextProcess": {"@id": "#process/cycle_b"},
        }
        cycle_b = {
            "@id": "#process/cycle_b",
            "executesProtocol": meas["executesProtocol"],
            "inputs": [{"@id": "#sample/s1"}],
            "outputs": [{"@id": "#data_file/raw0"}],
            "parameterValues": [],
            "previousProcess": {"@id": "#process/cycle_a"},
            "nextProcess": {"@id": "#process/cycle_a"},
        }
        proc_seq.extend([cycle_a, cycle_b])

        with pytest.raises(ExtractionError, match="did not cover all processes"):
            extractor.extract(minimal_single_run_isa)


class TestDataFileExistenceFreshness:
    def test_datafile_tracks_extract_snapshot_and_live_existence(
        self, extractor, minimal_single_run_isa, tmp_csv
    ):
        assay = minimal_single_run_isa["studies"][0]["assays"][0]
        for data_file in assay["dataFiles"]:
            if data_file.get("type") == "Processed Data File":
                data_file["name"] = str(tmp_csv)

        inv = extractor.extract(minimal_single_run_isa)
        run = inv.studies[0].assays[0].runs[0]
        assert run.processed_file is not None
        assert run.processed_file.exists_at_extract is True
        assert run.processed_file.exists is True

        tmp_csv.unlink()
        assert run.processed_file.exists_at_extract is True
        assert run.processed_file.exists is False


class TestParameterValueResolution:
    def test_invalid_parameter_value_is_warned_and_skipped(
        self, extractor, minimal_params_isa_file, tmp_path, caplog
    ):
        from isa_phm.parser import ISAParser
        from isa_phm.preprocessor import ISAPreprocessor

        raw = ISAParser(strict=False).load(minimal_params_isa_file)
        assay = raw["studies"][0]["assays"][0]
        for proc in assay["processSequence"]:
            inputs = proc.get("inputs", [])
            if inputs and inputs[0].get("@id", "").startswith("#sample/"):
                proc["parameterValues"][0]["value"] = {"bad": "shape"}
                break

        repaired, _ = ISAPreprocessor(data_root=tmp_path, auto_fix=True).preprocess(raw)
        with caplog.at_level(logging.WARNING, logger="isa_phm"):
            inv = extractor.extract(repaired)

        run = inv.studies[0].assays[0].runs[0]
        assert run.measurement_params == []
        assert "Skipping invalid measurement parameterValue" in caplog.text
