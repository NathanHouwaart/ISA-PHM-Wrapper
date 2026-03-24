from __future__ import annotations

import pytest

from isa_phm import ISAWrapper
from isa_phm.errors import ValidationError


class TestSemanticManifest:
    def test_manifest_maps_known_and_unknown_fields(self, minimal_semantic_isa_file, tmp_path):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )

        manifest = wrapper.semantic_manifest()
        study = wrapper.investigation.studies[0]
        assay = study.assays[0]

        factors = {f.source_name: f for f in manifest.study_factors[study.study_id]}
        assert factors["Motor speed"].semantic_key == "operating_speed"
        assert factors["Motor speed"].status == "mapped"
        assert factors["Discharge Perssure"].semantic_key == "pressure"
        assert factors["Mystery Knob"].status == "unknown"

        key = f"{study.study_id}:{assay.assay_id}"
        meas = {f.source_name: f for f in manifest.assay_measurement_params[key]}
        proc = {f.source_name: f for f in manifest.assay_processing_params[key]}
        assert meas["Sampling Frequency"].semantic_key == "sampling_frequency"
        assert proc["Completely Custom Param"].status == "unknown"

        assert manifest.diagnostics.total_fields > 0
        assert manifest.diagnostics.mapped_fields >= 3

    def test_override_config_takes_precedence(
        self,
        minimal_semantic_isa_file,
        semantic_override_config,
        tmp_path,
    ):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
            semantic_config_path=semantic_override_config,
        )

        manifest = wrapper.semantic_manifest()
        study = wrapper.investigation.studies[0]
        assay = study.assays[0]

        factors = {f.source_name: f for f in manifest.study_factors[study.study_id]}
        assert factors["Mystery Knob"].semantic_key == "custom_mystery_knob"
        assert factors["Mystery Knob"].provenance == "override_exact"

        key = f"{study.study_id}:{assay.assay_id}"
        proc = {f.source_name: f for f in manifest.assay_processing_params[key]}
        assert proc["Completely Custom Param"].semantic_key == "custom_processing_parameter"
        assert proc["Completely Custom Param"].provenance == "override_exact"


class TestSemanticProxyMethods:
    def test_study_and_assay_semantic_helpers(self, minimal_semantic_isa_file, tmp_path):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        study = wrapper.study("Test Study")
        factors = study.semantic_factors()
        assert len(factors) >= 1
        assert any(f.semantic_key == "operating_speed" for f in factors)

        assay = study.assay("a_st01_se01")
        params = assay.semantic_parameters()
        assert "measurement" in params and "processing" in params
        assert any(p.semantic_key == "sampling_frequency" for p in params["measurement"])


class TestSemanticStrictControls:
    def test_require_override_config_fails_without_config(
        self, minimal_semantic_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        with pytest.raises(ValidationError, match="SEM_OVERRIDE_CONFIG_REQUIRED"):
            wrapper.semantic_manifest(
                strict=True,
                max_unknown_ratio=1.0,
                max_ambiguous_ratio=1.0,
                require_override_config=True,
            )

    def test_strict_passes_with_override_config(
        self,
        minimal_semantic_isa_file,
        semantic_override_config,
        tmp_path,
    ):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
            semantic_config_path=semantic_override_config,
        )
        manifest = wrapper.semantic_manifest(
            strict=True,
            max_unknown_ratio=0.30,
            max_ambiguous_ratio=0.0,
            require_override_config=True,
        )
        assert manifest.diagnostics.strict_violations == []
