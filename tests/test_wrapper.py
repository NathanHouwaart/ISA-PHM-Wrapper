from __future__ import annotations

import json

import pandas as pd
import pytest

from isa_phm import ISAWrapper
from isa_phm.errors import ValidationError
from isa_phm.schemas import DatasetValidationReport


class TestISAWrapperSummaries:
    def test_summary_returns_one_row_dataframe(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )

        df = wrapper.summary()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert "title" in df.columns
        assert "n_studies" in df.columns
        assert int(df.at[0, "n_studies"]) == 1

    def test_extensive_summary_returns_expected_tables(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )

        tables = wrapper.extensive_summary()
        assert isinstance(tables, dict)
        assert {
            "investigation",
            "studies",
            "assays",
            "factors",
            "contacts",
            "publications",
        }.issubset(
            tables.keys()
        )

        assert isinstance(tables["investigation"], pd.DataFrame)
        assert isinstance(tables["studies"], pd.DataFrame)
        assert isinstance(tables["assays"], pd.DataFrame)
        assert isinstance(tables["factors"], pd.DataFrame)
        assert isinstance(tables["contacts"], pd.DataFrame)
        assert isinstance(tables["publications"], pd.DataFrame)

        assert len(tables["studies"]) == 1
        assert len(tables["assays"]) == 1
        assert len(tables["factors"]) == 1

    def test_contacts_and_publications_helpers(
        self, minimal_publication_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_publication_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )

        contacts_df = wrapper.contacts()
        publications_df = wrapper.publications()

        assert len(contacts_df) == 2
        assert "full_name" in contacts_df.columns
        assert "roles" in contacts_df.columns

        assert len(publications_df) == 1
        assert publications_df.at[0, "title"] == "An Example ISA-PHM Publication"
        assert publications_df.at[0, "doi"] == "10.1000/example.doi"
        assert publications_df.at[0, "status"] == "Published"

        # direct investigation model helpers
        assert wrapper.investigation.contacts_df().equals(contacts_df)
        assert wrapper.investigation.publications_df().equals(publications_df)

    def test_investigation_aliases_removed(self, minimal_publication_isa_file, tmp_path):
        wrapper = ISAWrapper(
            minimal_publication_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        with pytest.raises(AttributeError):
            _ = wrapper.investigation_contacts()
        with pytest.raises(AttributeError):
            _ = wrapper.investigation_publications()


class TestSemanticStrictMode:
    def test_semantic_strict_raises_when_unknown_ratio_exceeds_threshold(
        self,
        minimal_semantic_isa_file,
        tmp_path,
    ):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )

        with pytest.raises(ValidationError, match="SEM_UNKNOWN_RATIO_EXCEEDED"):
            wrapper.semantic_manifest(
                strict=True,
                max_unknown_ratio=0.0,
                max_ambiguous_ratio=0.0,
            )

    def test_semantic_diagnostics_include_ratios_and_missing_override_fields(
        self,
        minimal_semantic_isa_file,
        tmp_path,
    ):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        manifest = wrapper.semantic_manifest()
        diag = manifest.diagnostics
        assert 0.0 <= diag.unknown_ratio <= 1.0
        assert 0.0 <= diag.ambiguous_ratio <= 1.0
        assert isinstance(diag.missing_override_fields, list)
        assert "Mystery Knob" in diag.missing_override_fields


class TestPublicationContactLinking:
    def test_publication_authors_resolve_to_contact_names_and_emails(
        self, minimal_publication_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_publication_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        pubs = wrapper.investigation.publications
        assert len(pubs) == 1
        pub = pubs[0]
        assert pub.resolved_author_names == ["Alice Example", "Bob Example"]
        assert pub.resolved_author_emails == ["alice@example.com", "bob@example.com"]
        assert pub.unresolved_author_tokens == []


class TestAIContextExport:
    def test_ai_context_has_required_keys_and_is_json_safe(
        self, minimal_publication_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_publication_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        payload = wrapper.ai_context(include_semantics=True, include_validation=True)

        assert list(payload.keys()) == [
            "schema_version",
            "generated_at_utc",
            "source_path",
            "investigation",
            "contacts",
            "publications",
            "studies",
            "assays",
            "factors",
            "semantic_manifest",
            "validation_report",
        ]
        assert payload["schema_version"] == "isa_phm.ai_context.v1"
        assert isinstance(payload["contacts"], list)
        assert isinstance(payload["publications"], list)
        assert isinstance(payload["studies"], list)
        assert isinstance(payload["assays"], list)
        assert isinstance(payload["factors"], list)
        assert isinstance(payload["semantic_manifest"], dict)
        assert isinstance(payload["validation_report"], dict)

        # Must be JSON-safe and deterministic in structure.
        json.dumps(payload)

    def test_ai_context_can_skip_optional_sections(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        payload = wrapper.ai_context(include_semantics=False, include_validation=False)
        assert "semantic_manifest" not in payload
        assert "validation_report" not in payload


class TestValidateDataset:
    def test_validate_dataset_returns_structured_report(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )

        report = wrapper.validate_dataset(check_files=True)
        assert isinstance(report, DatasetValidationReport)
        assert report.n_errors >= 0
        assert report.n_warnings >= 0
        assert report.n_info >= 0
        assert isinstance(report.issues, list)

    def test_validate_dataset_reports_missing_file(
        self, minimal_single_run_isa_file, tmp_path, tmp_csv
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        tmp_csv.unlink()

        report = wrapper.validate_dataset(check_files=True)
        codes = {issue.code for issue in report.issues}
        assert "FILE_NOT_FOUND" in codes

    def test_validate_dataset_semantic_strict_emits_error_codes(
        self, minimal_semantic_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_semantic_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        report = wrapper.validate_dataset(
            semantic_strict=True,
            max_unknown_ratio=0.0,
            max_ambiguous_ratio=0.0,
        )
        assert report.ok is False
        codes = {issue.code for issue in report.issues}
        assert "SEM_UNKNOWN_RATIO_EXCEEDED" in codes


class TestWrapperPerformanceOptions:
    def test_wrapper_accepts_chunked_mode_configuration(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
            enable_chunked_large_file_mode=False,
            large_file_threshold_mb=12.5,
            chunk_rows=4096,
        )
        assert wrapper._integrator._enable_chunked_large_file_mode is False
        assert wrapper._integrator._large_file_threshold_mb == 12.5
        assert wrapper._integrator._chunk_rows == 4096
