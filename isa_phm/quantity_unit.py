"""Advisory checks based on the pinned ISA-PHM Specification rule set."""

from dataclasses import dataclass
from importlib import resources
import json
from pathlib import Path


@dataclass(frozen=True)
class QuantityUnitAdvisory:
    status: str
    warnings: tuple[str, ...] = ()
    reason: str | None = None


def _load_rules():
    # Wheels bundle this file at build time from the pinned submodule. Source
    # and editable installs read the submodule directly; neither reaches out
    # to the network or silently treats a missing rule set as a pass.
    packaged = resources.files("isa_phm").joinpath("_specification", "quantity-unit-warning-rules.json")
    if packaged.is_file():
        return json.loads(packaged.read_text(encoding="utf-8"))
    source = Path(__file__).resolve().parents[1] / "vendor/isa-phm-specification/specification/quantity-unit-warning-rules.json"
    return json.loads(source.read_text(encoding="utf-8"))


def check_quantity_unit(quantity_identifier: str | None, unit_identifier: str | None) -> QuantityUnitAdvisory:
    """Check explicit identifiers against known mismatches, not full compatibility."""
    if not quantity_identifier or not unit_identifier:
        return QuantityUnitAdvisory("not_checked", reason="Both quantity and unit identifiers are required.")
    try:
        data = _load_rules()
        rules = data["rules"]
        if not isinstance(rules, list):
            raise ValueError("Invalid rules list")
        warnings = tuple(rule["message"] for rule in rules
                         if quantity_identifier in rule["quantityIdentifiers"] and unit_identifier in rule["unitIdentifiers"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return QuantityUnitAdvisory("not_checked", reason=f"Specification rules unavailable or invalid: {exc}")
    return QuantityUnitAdvisory("warning" if warnings else "no_known_mismatch", warnings=warnings)
