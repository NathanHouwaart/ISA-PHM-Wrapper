import pytest

from isa_phm.quantity_unit import check_quantity_unit

ACCELERATION = "http://qudt.org/vocab/quantitykind/Acceleration"


def test_known_incompatible_pair_returns_advisory():
    result = check_quantity_unit(ACCELERATION, "http://qudt.org/vocab/unit/GM")
    assert result.status == "warning"
    assert "mass unit" in result.warnings[0]


@pytest.mark.parametrize("unit", ["http://qudt.org/vocab/unit/G", "urn:custom:unit"])
def test_unlisted_pair_is_not_claimed_compatible(unit):
    assert check_quantity_unit(ACCELERATION, unit).status == "no_known_mismatch"


def test_missing_identifiers_or_rules_are_not_checked(monkeypatch):
    assert check_quantity_unit(None, "urn:custom:unit").status == "not_checked"
    def unavailable():
        raise OSError("missing")
    monkeypatch.setattr("isa_phm.quantity_unit._load_rules", unavailable)
    assert check_quantity_unit(ACCELERATION, "http://qudt.org/vocab/unit/GM").status == "not_checked"
