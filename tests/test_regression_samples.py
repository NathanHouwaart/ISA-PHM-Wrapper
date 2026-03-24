from __future__ import annotations

from pathlib import Path

import pytest

from isa_phm import ISAWrapper
from isa_phm.errors import DataFileError


MILLING_JSON = Path(
    r"g:\ISA\ISA-PHM-Wizard\src\tests\fixtures\golden\isa-phm-out-milling.json"
)
SIETZE_JSON = Path(
    r"g:\ISA\ISA-PHM-Wizard\src\tests\fixtures\golden\isa-phm-out-sietze.json"
)
XJTU_JSON = Path(
    r"g:\XJTU-SY_Bearing_Datasets\XJTU-SY Bearing Datasets-ISA-PHM.json"
)


def _require_path(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Regression fixture not found: '{path}'")


class TestGoldenRegression:
    def test_milling_parses_and_builds_semantics(self):
        _require_path(MILLING_JSON)
        wrapper = ISAWrapper(
            MILLING_JSON,
            data_root=MILLING_JSON.parent,
            strict_validation=False,
        )
        assert len(wrapper.investigation.studies) >= 1
        manifest = wrapper.semantic_manifest()
        assert manifest.diagnostics.total_fields >= 0

    def test_sietze_parses_and_builds_semantics(self):
        _require_path(SIETZE_JSON)
        wrapper = ISAWrapper(
            SIETZE_JSON,
            data_root=SIETZE_JSON.parent,
            strict_validation=False,
        )
        assert len(wrapper.investigation.studies) >= 1
        manifest = wrapper.semantic_manifest()
        assert manifest.diagnostics.total_fields >= 0


class TestXJTURegression:
    def test_xjtu_raw_only_auto_vs_processed_contract(self):
        _require_path(XJTU_JSON)
        wrapper = ISAWrapper(
            XJTU_JSON,
            data_root=XJTU_JSON.parent,
            strict_validation=False,
        )

        candidate: tuple[str, str, str] | None = None
        for study in wrapper.investigation.studies:
            for assay in study.assays:
                for run in assay.runs:
                    raw_exists = bool(
                        run.raw_file
                        and run.raw_file.path
                        and Path(run.raw_file.path).exists()
                    )
                    processed_missing = not (
                        run.processed_file and run.processed_file.path
                    )
                    if raw_exists and processed_missing:
                        candidate = (study.study_id, assay.assay_id, run.run_id)
                        break
                if candidate is not None:
                    break
            if candidate is not None:
                break

        if candidate is None:
            pytest.skip(
                "No run found with existing raw file and missing processed file."
            )

        study_id, assay_id, run_id = candidate
        assay_proxy = wrapper.study(study_id).assay(assay_id)

        with pytest.raises(DataFileError, match="no 'processed' data file"):
            assay_proxy.load_dataframe(run_id=run_id, file_type="processed")

        _, meta = assay_proxy.load_dataframe_with_meta(
            run_id=run_id,
            file_type="auto",
        )
        assert meta.requested_file_type == "auto"
        assert meta.resolved_file_type == "raw"
