"""
ISAWrapper — the primary entry point for the ISA-PHM Python wrapper.

Chains the four core components:
    ISAParser  →  ISAPreprocessor  →  MetadataExtractor  →  DataIntegrator

Exposes the fluent proxy API via ``QueryNavigator`` and convenience shortcuts.

Example usage::

    from isa_phm import ISAWrapper

    wrapper = ISAWrapper(
        "path/to/i_investigation.json",
        data_root="path/to/data/",
    )
    print(wrapper.investigation_overview())
    df = wrapper.study("Case 01").assay("a_st01_se01").load_dataframe()
    fig = wrapper.study("Case 01").assay("a_st01_se01").plot_lifecycle()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

from .extractor import MetadataExtractor
from .integrator import DataIntegrator
from .parser import ISAParser
from .plotter import ISAPlotter, PlotConfig
from .preprocessor import ISAPreprocessor
from .proxy import QueryNavigator, StudyProxy
from .semantic import SemanticNormalizer
from .schemas import (
    InvestigationModel,
    InvestigationOverview,
    RepairLog,
    SemanticManifest,
    StudySummary,
)

logger = logging.getLogger("isa_phm")


class ISAWrapper:
    """
    All-in-one facade that parses, preprocesses, and extracts an ISA-PHM dataset.

    Parameters
    ----------
    path : str | Path
        Path to the ISA-JSON file (``i_...json``).
    data_root : str | Path | None
        Directory that contains the actual CSV measurement files.
        If None, defaults to the directory containing the ISA-JSON file.
    auto_fix : bool
        Enable the five ISAPreprocessor auto-fix rules (default True).
    strict_validation : bool
        Use isatools for schema validation (default True).
        Set to False to skip isatools check (e.g., when isatools is not installed).
    cache_maxsize : int
        Maximum DataFrames held in the DataIntegrator's FIFO cache (default 100).
    plot_config : PlotConfig | None
        Optional style overrides passed to ISAPlotter.
    semantic_config_path : str | Path | None
        Optional JSON path with semantic alias overrides.
    """

    def __init__(
        self,
        path: str | Path,
        data_root: str | Path | None = None,
        auto_fix: bool = True,
        strict_validation: bool = True,
        cache_maxsize: int = 100,
        plot_config: PlotConfig | None = None,
        semantic_config_path: str | Path | None = None,
    ) -> None:
        path = Path(path)
        if not path.exists():
            from .errors import ParseError

            raise ParseError(f"ISA-JSON file not found: '{path}'.")

        if data_root is None:
            data_root = path.parent
        data_root = Path(data_root)

        logger.info("ISAWrapper: loading '%s', data_root='%s'.", path, data_root)

        # --- Step 1: Parse ---
        parser = ISAParser(strict=strict_validation)
        raw = parser.load(path)

        # --- Step 2: Preprocess ---
        preprocessor = ISAPreprocessor(data_root=data_root, auto_fix=auto_fix)
        repaired, repair_log = preprocessor.preprocess(raw)

        # --- Step 3: Extract domain model ---
        extractor = MetadataExtractor()
        investigation = extractor.extract(repaired)

        # --- Step 4: Wire integrator + plotter + proxy layer ---
        integrator = DataIntegrator(data_root=data_root, cache_maxsize=cache_maxsize)
        plotter = ISAPlotter(config=plot_config)
        semantic = SemanticNormalizer(override_config_path=semantic_config_path)
        navigator = QueryNavigator(investigation, integrator, plotter, semantic=semantic)

        # Store as instance attributes.
        self._investigation: InvestigationModel = investigation
        self._integrator: DataIntegrator = integrator
        self._plotter: ISAPlotter = plotter
        self._navigator: QueryNavigator = navigator
        self._semantic: SemanticNormalizer = semantic
        self._repair_log: RepairLog = repair_log
        self._source_path: Path = path

        n_repairs = len(repair_log)
        if n_repairs:
            logger.info("ISAWrapper: %d preprocessor repair(s) applied.", n_repairs)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def investigation(self) -> InvestigationModel:
        """The fully-extracted domain model (read-only)."""
        return self._investigation

    @property
    def repair_log(self) -> RepairLog:
        """Preprocessing repair actions.  Empty if no repairs were needed."""
        return self._repair_log

    @property
    def source_path(self) -> Path:
        """Absolute path to the ISA-JSON source file."""
        return self._source_path

    # ------------------------------------------------------------------
    # High-level inspection
    # ------------------------------------------------------------------

    def investigation_overview(self) -> InvestigationOverview:
        """Return a structured summary of the investigation."""
        return self._navigator.investigation_overview()

    def list_studies(self) -> list[StudySummary]:
        """Return a summary row for each study."""
        return self._navigator.list_studies()

    def summary(self) -> "pd.DataFrame":
        """
        Return a compact one-row investigation summary as a DataFrame.

        This is a notebook-friendly alternative to raw model repr output.
        """
        import pandas as pd

        ov = self.investigation_overview()
        desc = " ".join((ov.description or "").split())
        if len(desc) > 180:
            desc = f"{desc[:177]}..."
        return pd.DataFrame(
            [
                {
                    "title": ov.title,
                    "identifier": ov.identifier,
                    "experiment_type": ov.experiment_type,
                    "n_studies": ov.n_studies,
                    "n_contacts": ov.n_contacts,
                    "n_publications": len(self._investigation.publications),
                    "description_short": desc,
                    "source_path": str(self._source_path),
                }
            ]
        )

    def contacts(self) -> "pd.DataFrame":
        """
        Return investigation contacts as a notebook-friendly DataFrame.

        This avoids manual traversal of ``wrapper.investigation.contacts``.
        """
        return self._investigation.contacts_df()

    def investigation_contacts(self) -> "pd.DataFrame":
        """Alias for :meth:`contacts` for explicit investigation-level discovery."""
        return self.contacts()

    def publications(self) -> "pd.DataFrame":
        """
        Return investigation publications as a notebook-friendly DataFrame.

        Notes
        -----
        ``author_tokens`` is kept as a semicolon-joined string because ISA
        commonly stores contact IDs there, not resolved names.
        """
        return self._investigation.publications_df()

    def investigation_publications(self) -> "pd.DataFrame":
        """Alias for :meth:`publications` for explicit investigation-level discovery."""
        return self.publications()

    def extensive_summary(self) -> "dict[str, pd.DataFrame]":
        """
        Return a full investigation summary split into notebook-ready tables.

        Returns a dict with keys:
        - ``investigation``: one-row top-level metadata
        - ``studies``: one row per study
        - ``assays``: one row per assay (sensor channel)
        - ``factors``: one row per factor
        - ``contacts``: one row per contact
        - ``publications``: one row per publication
        """
        import pandas as pd

        ov = self.investigation_overview()
        inv_df = self.summary()

        study_rows: list[dict] = []
        assay_rows: list[dict] = []
        factor_rows: list[dict] = []

        for s in self._investigation.studies:
            study_rows.append(
                {
                    "study_id": s.study_id,
                    "title": s.title,
                    "n_assays": len(s.assays),
                    "n_runs": s.run_count,
                    "n_factors": len(s.factors),
                }
            )

            for a in s.assays:
                n_raw_files = sum(
                    1 for r in a.runs if r.raw_file is not None and bool(r.raw_file.path)
                )
                n_processed_files = sum(
                    1
                    for r in a.runs
                    if r.processed_file is not None and bool(r.processed_file.path)
                )
                assay_rows.append(
                    {
                        "study_id": s.study_id,
                        "study_title": s.title,
                        "assay_id": a.assay_id,
                        "sensor_alias": a.sensor.alias,
                        "sensor_id": a.sensor.sensor_id,
                        "measurement_type": a.sensor.measurement_type,
                        "technology_type": a.sensor.technology_type,
                        "technology_platform": a.sensor.technology_platform,
                        "n_runs": len(a.runs),
                        "n_raw_files": n_raw_files,
                        "n_processed_files": n_processed_files,
                    }
                )

            for f in s.factors:
                factor_rows.append(
                    {
                        "study_id": s.study_id,
                        "study_title": s.title,
                        "factor_name": f.factor_name,
                        "factor_type": f.factor_type,
                        "unit": f.unit or "",
                        "description": f.description or "",
                    }
                )

        studies_df = pd.DataFrame(
            study_rows,
            columns=["study_id", "title", "n_assays", "n_runs", "n_factors"],
        )
        assays_df = pd.DataFrame(
            assay_rows,
            columns=[
                "study_id",
                "study_title",
                "assay_id",
                "sensor_alias",
                "sensor_id",
                "measurement_type",
                "technology_type",
                "technology_platform",
                "n_runs",
                "n_raw_files",
                "n_processed_files",
            ],
        )
        factors_df = pd.DataFrame(
            factor_rows,
            columns=[
                "study_id",
                "study_title",
                "factor_name",
                "factor_type",
                "unit",
                "description",
            ],
        )
        contacts_df = self.contacts().loc[
            :, ["full_name", "email", "affiliation", "roles", "orcid"]
        ]
        publications_df = self.publications()

        # Keep stable ordering for interactive notebooks.
        if not studies_df.empty:
            studies_df = studies_df.sort_values(["title", "study_id"]).reset_index(
                drop=True
            )
        if not assays_df.empty:
            assays_df = assays_df.sort_values(
                ["study_title", "assay_id"]
            ).reset_index(drop=True)
        if not factors_df.empty:
            factors_df = factors_df.sort_values(
                ["study_title", "factor_name"]
            ).reset_index(drop=True)

        # Keep top-level title in investigation table for easy context checks.
        inv_df.loc[:, "n_studies"] = ov.n_studies

        return {
            "investigation": inv_df,
            "studies": studies_df,
            "assays": assays_df,
            "factors": factors_df,
            "contacts": contacts_df,
            "publications": publications_df,
        }

    def semantic_manifest(self) -> SemanticManifest:
        """Return normalized semantic labels for factors and protocol parameters."""
        return self._semantic.build_manifest(self._investigation)

    # ------------------------------------------------------------------
    # Fluent proxy navigation
    # ------------------------------------------------------------------

    def study(self, study_id: str) -> StudyProxy:
        """
        Navigate into a study by UUID or human-readable title.

        Parameters
        ----------
        study_id : str
            Study UUID or title (case-insensitive).

        Returns
        -------
        StudyProxy

        Raises
        ------
        StudyNotFoundError

        Example
        -------
        >>> wrapper.study("BPFO Fault Severity 1 100%").assay("a_st01_se01").load_dataframe()
        """
        return self._navigator.study(study_id)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all DataFrames from the integrator's FIFO cache."""
        self._integrator.clear_cache()

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_studies = len(self._investigation.studies)
        return (
            f"ISAWrapper("
            f"title={self._investigation.title!r}, "
            f"n_studies={n_studies}, "
            f"experiment_type={self._investigation.experiment_type!r})"
        )
