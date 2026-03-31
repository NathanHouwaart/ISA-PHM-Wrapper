from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from isa_phm import ISAWrapper
from isa_phm.errors import ValidationError
from isa_phm.schemas import DatasetValidationReport, OutlierReport


def _build_two_study_isa_file(minimal_single_run_isa: dict, tmp_csv, tmp_path) -> Path:
    """Create a temporary ISA file with two single-run studies."""
    isa = copy.deepcopy(minimal_single_run_isa)

    def _set_processed_paths(study: dict, path: str) -> None:
        for assay in study["assays"]:
            for data_file in assay["dataFiles"]:
                if data_file["type"] == "Processed Data File":
                    data_file["name"] = path

    _set_processed_paths(isa["studies"][0], str(tmp_csv))

    study2 = copy.deepcopy(isa["studies"][0])
    study2["@id"] = "#study/st2"
    study2["identifier"] = "st2-uuid"
    study2["title"] = "Test Study B"
    study2["assays"][0]["@id"] = "#assay/a2"
    study2["assays"][0]["filename"] = "a_st02_se01"
    _set_processed_paths(study2, str(tmp_csv))
    isa["studies"].append(study2)

    p = tmp_path / "i_two_studies_wrapper.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


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

    def test_study_lookup_by_1_based_index(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        study = wrapper.study(1)
        assert study.title == "Test Study"

    def test_compare_studies_is_explicit_only(
        self, minimal_single_run_isa, tmp_csv, tmp_path
    ):
        isa_file = _build_two_study_isa_file(minimal_single_run_isa, tmp_csv, tmp_path)
        wrapper = ISAWrapper(
            isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        out = wrapper.compare_studies(["Test Study B"], assay_id=1, file_type="raw")
        assert list(out.keys()) == ["Test Study B"]


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

    def test_ai_context_matches_golden_snapshot(
        self, minimal_publication_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_publication_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        payload = wrapper.ai_context(include_semantics=True, include_validation=True)
        payload["generated_at_utc"] = "<GENERATED_AT_UTC>"
        payload["source_path"] = "<SOURCE_PATH>"

        snapshot_path = Path(__file__).parent / "golden" / "ai_context_minimal_publication.json"
        expected = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert payload == expected


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

    def test_validate_dataset_deduplicates_repeated_file_not_found(
        self, minimal_multi_run_isa_file, tmp_csv_dir
    ):
        for p in tmp_csv_dir.glob("run_*.csv"):
            p.unlink()

        wrapper = ISAWrapper(
            minimal_multi_run_isa_file,
            data_root=tmp_csv_dir,
            strict_validation=False,
        )
        report = wrapper.validate_dataset(check_files=True)
        file_issues = [issue for issue in report.issues if issue.code == "FILE_NOT_FOUND"]
        assert len(file_issues) == 1
        issue = file_issues[0]
        assert issue.scope == "multiple"
        assert issue.context["n_occurrences"] == 3
        assert issue.context["n_unique_scopes"] == 3


class TestOutlierReportDataFrame:
    def test_outlier_dataframe_has_stable_columns_for_empty_and_nonempty(self):
        empty = OutlierReport(
            n_outliers=0,
            pct_outliers=0.0,
            method="iqr",
            threshold=1.5,
            by_column={},
        )
        non_empty = OutlierReport(
            n_outliers=2,
            pct_outliers=5.0,
            method="iqr",
            threshold=1.5,
            by_column={
                "value": {
                    "n_outliers": 2,
                    "lower_bound": -1.0,
                    "upper_bound": 1.0,
                }
            },
        )

        df_empty = empty.to_dataframe()
        df_non_empty = non_empty.to_dataframe()
        expected = [
            "column",
            "n_outliers",
            "pct_outliers",
            "lower_bound",
            "upper_bound",
            "method",
            "threshold",
        ]
        assert list(df_empty.columns) == expected
        assert list(df_non_empty.columns) == expected
        assert df_empty.at[0, "column"] == "__all__"


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
            csv_bad_lines="warn",
        )
        assert wrapper._integrator._enable_chunked_large_file_mode is False
        assert wrapper._integrator._large_file_threshold_mb == 12.5
        assert wrapper._integrator._chunk_rows == 4096
        assert wrapper._integrator._csv_bad_lines == "warn"
