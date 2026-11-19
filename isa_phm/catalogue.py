"""Look up explicit identifiers in the pinned shared draft catalogues.

Presence does not imply conversion support or validate a measurement.
"""
from copy import deepcopy
from functools import lru_cache
from importlib import resources
import json
from pathlib import Path

CATALOGUE_VERSION = "qudt-3.5.2"


def _read_catalogue(name: str):
    packaged = resources.files("isa_phm").joinpath("_specification", "catalogues", name)
    if packaged.is_file():
        return json.loads(packaged.read_text(encoding="utf-8"))
    source = Path(__file__).resolve().parents[1] / "vendor/isa-phm-specification/catalogues" / name
    if not source.is_file():
        raise RuntimeError("Shared specification catalogues are unavailable; initialize the ISA-PHM specification submodule.")
    return json.loads(source.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _indexes():
    units = _read_catalogue("qudt-units-3.5.2.json")
    quantities = _read_catalogue("qudt-quantities-3.5.2.json") + _read_catalogue("isa-phm-quantities.json")
    return ({entry["identifier"]: entry for entry in units}, {entry["identifier"]: entry for entry in quantities})


def lookup_unit(identifier: str) -> dict | None:
    """Look up an exact unit identifier; return None if unknown, never infer."""
    item = _indexes()[0].get(identifier)
    return deepcopy(item) if item is not None else None


def lookup_quantity(identifier: str) -> dict | None:
    """Look up an exact quantity identifier, including ISA-PHM additions."""
    item = _indexes()[1].get(identifier)
    return deepcopy(item) if item is not None else None
