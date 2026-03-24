from __future__ import annotations

import pandas as pd

from isa_phm import ISAWrapper


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

        # explicit investigation-level aliases
        assert wrapper.investigation_contacts().equals(contacts_df)
        assert wrapper.investigation_publications().equals(publications_df)

        # direct investigation model helpers
        assert wrapper.investigation.contacts_df().equals(contacts_df)
        assert wrapper.investigation.publications_df().equals(publications_df)
