"""
isa_phm — Python wrapper for ISA-PHM datasets.

Primary entry point: :class:`ISAWrapper`.

Quick start::

    from isa_phm import ISAWrapper

    wrapper = ISAWrapper("i_investigation.json", data_root="data/")
    df = wrapper.study("Case 01").assay("a_st01_se01").load_dataframe()
"""

from .wrapper import ISAWrapper
from .attachments import Attachment
from .plotter import ISAPlotter, PlotConfig
from .proxy import AssayGroup
from .semantic import SemanticNormalizer
from .quantity_unit import QuantityUnitAdvisory, check_quantity_unit
from .tools import ToolRegistry
from .schemas import (
    AnalysisCapabilities,
    AssayAnalysisCapabilities,
    Capability,
    DataLoadMetadata,
    DatasetValidationReport,
    PublicationModel,
    SemanticDiagnostics,
    SemanticField,
    SemanticManifest,
    ValidationIssue,
    MetadataResolution,
)
from .errors import (
    ISAPHMError,
    ParseError,
    ValidationError,
    PreprocessingError,
    ExtractionError,
    DataFileError,
    StudyNotFoundError,
    AssayNotFoundError,
    RunNotFoundError,
    AmbiguousRunError,
    PlotError,
    ExportError,
)

__all__ = [
    "ISAWrapper",
    "Attachment",
    "ISAPlotter",
    "PlotConfig",
    "AssayGroup",
    "SemanticNormalizer",
    "QuantityUnitAdvisory",
    "check_quantity_unit",
    "ToolRegistry",
    "AnalysisCapabilities",
    "AssayAnalysisCapabilities",
    "Capability",
    "DataLoadMetadata",
    "DatasetValidationReport",
    "PublicationModel",
    "SemanticField",
    "SemanticDiagnostics",
    "SemanticManifest",
    "ValidationIssue",
    "MetadataResolution",
    "ISAPHMError",
    "ParseError",
    "ValidationError",
    "PreprocessingError",
    "ExtractionError",
    "DataFileError",
    "StudyNotFoundError",
    "AssayNotFoundError",
    "RunNotFoundError",
    "AmbiguousRunError",
    "PlotError",
    "ExportError",
]
__version__ = "0.2.0"
