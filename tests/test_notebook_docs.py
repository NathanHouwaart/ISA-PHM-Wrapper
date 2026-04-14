from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


CANONICAL_NOTEBOOKS = [
    "notebooks/00_getting_started/01_getting_started.ipynb",
    "notebooks/01_api_basics/10_basic_load_and_inspect.ipynb",
    "notebooks/01_api_basics/11_basic_semantic_normalization.ipynb",
    "notebooks/01_api_basics/12_basic_plot_and_export.ipynb",
    "notebooks/02_features/20_lifecycle_features.ipynb",
    "notebooks/02_features/21_compare_studies.ipynb",
    "notebooks/02_features/22_time_domain.ipynb",
    "notebooks/02_features/23_spectral_analysis.ipynb",
    "notebooks/02_features/24_correlation.ipynb",
    "notebooks/02_features/25_assay_groups.ipynb",
    "notebooks/03_workflows/02_diagnostic_workflow.ipynb",
    "notebooks/03_workflows/03_prognostic_workflow.ipynb",
]


DEPRECATED_TOP_LEVEL_NOTEBOOKS = [
    "notebooks/01_getting_started.ipynb",
    "notebooks/02_diagnostic_workflow.ipynb",
    "notebooks/03_prognostic_workflow.ipynb",
    "notebooks/10_basic_load_and_inspect.ipynb",
    "notebooks/11_basic_semantic_normalization.ipynb",
    "notebooks/12_basic_plot_and_export.ipynb",
    "notebooks/20_lifecycle_features.ipynb",
    "notebooks/21_compare_studies.ipynb",
    "notebooks/22_time_domain.ipynb",
    "notebooks/23_spectral_analysis.ipynb",
    "notebooks/24_correlation.ipynb",
    "notebooks/25_assay_groups.ipynb",
]


def _notebook_source(path: Path) -> str:
    data = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(cell.get("source", [])) for cell in data.get("cells", []))


class TestNotebookStructure:
    def test_canonical_notebooks_exist(self):
        missing = [
            rel for rel in CANONICAL_NOTEBOOKS
            if not (REPO_ROOT / rel).exists()
        ]
        assert not missing, f"Missing canonical notebooks: {missing}"

    def test_deprecated_top_level_notebooks_are_removed(self):
        remaining = [
            rel for rel in DEPRECATED_TOP_LEVEL_NOTEBOOKS
            if (REPO_ROOT / rel).exists()
        ]
        assert not remaining, f"Deprecated top-level notebooks should be removed: {remaining}"


class TestNotebookCoverage:
    def test_missing_api_examples_are_present(self):
        source_by_file = {
            rel: _notebook_source(REPO_ROOT / rel)
            for rel in CANONICAL_NOTEBOOKS
        }

        assert "wrapper.extensive_summary(" in source_by_file[
            "notebooks/01_api_basics/10_basic_load_and_inspect.ipynb"
        ]
        assert "wrapper.clear_cache(" in source_by_file[
            "notebooks/01_api_basics/10_basic_load_and_inspect.ipynb"
        ]
        assert "to_ml_dataset(" in source_by_file[
            "notebooks/02_features/20_lifecycle_features.ipynb"
        ]
        assert "assay.run(" in source_by_file[
            "notebooks/02_features/22_time_domain.ipynb"
        ]
        assert "plot_missing_values(" in source_by_file[
            "notebooks/02_features/22_time_domain.ipynb"
        ]
        assert "plot_spectrogram(" in source_by_file[
            "notebooks/02_features/23_spectral_analysis.ipynb"
        ]
        assert "plot_waterfall(" in source_by_file[
            "notebooks/02_features/23_spectral_analysis.ipynb"
        ]

    def test_compare_contract_wording_is_explicit(self):
        compare_source = _notebook_source(
            REPO_ROOT / "notebooks/02_features/21_compare_studies.ipynb"
        )
        group_source = _notebook_source(
            REPO_ROOT / "notebooks/02_features/25_assay_groups.ipynb"
        )
        assert "Only the studies you pass are included." in compare_source
        assert "The base study (`self`) is always included." in compare_source
        assert "Only the studies you pass are included." in group_source
