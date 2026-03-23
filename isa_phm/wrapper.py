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

from .extractor import MetadataExtractor
from .integrator import DataIntegrator
from .parser import ISAParser
from .plotter import ISAPlotter, PlotConfig
from .preprocessor import ISAPreprocessor
from .proxy import QueryNavigator, StudyProxy
from .schemas import (
    InvestigationModel,
    InvestigationOverview,
    RepairLog,
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
    """

    def __init__(
        self,
        path: str | Path,
        data_root: str | Path | None = None,
        auto_fix: bool = True,
        strict_validation: bool = True,
        cache_maxsize: int = 100,
        plot_config: PlotConfig | None = None,
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
        navigator = QueryNavigator(investigation, integrator, plotter)

        # Store as instance attributes.
        self._investigation: InvestigationModel = investigation
        self._integrator: DataIntegrator = integrator
        self._plotter: ISAPlotter = plotter
        self._navigator: QueryNavigator = navigator
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
