import pytest

from isa_phm.capabilities import resolve_measured_unit
from isa_phm.schemas import ParameterValue
from isa_phm.extractor import _resolve_param_value
from isa_phm.utils import ReferenceResolver


def test_orientation_unit_is_not_a_measured_unit():
    result = resolve_measured_unit([ParameterValue(parameter_name="Sensor Orientation", value=90, unit="°")])
    assert not result.available


@pytest.mark.parametrize("value", ["g", "m/s²", "1", "custom-unit"])
def test_explicit_unit_is_used(value):
    result = resolve_measured_unit([ParameterValue(parameter_name="Measured Unit", value=value)])
    assert result.available and result.value == value


@pytest.mark.parametrize("value", ["Unknown", "Not applicable", ""])
def test_unknown_or_empty_unit_is_unavailable(value):
    assert not resolve_measured_unit([ParameterValue(parameter_name="Measured Unit", value=value)]).available


def test_conflicting_units_are_unavailable():
    result = resolve_measured_unit([ParameterValue(parameter_name="Measured Unit", value="g"), ParameterValue(parameter_name="Measured Unit", value="m/s²")])
    assert not result.available


def test_ontology_value_and_identifier_are_extracted():
    annotation = {"annotationValue": "g", "termAccession": "http://example.org/g"}
    value = _resolve_param_value({"category": {"@id": "#p"}, "value": annotation}, {"#p": "Measured Unit"}, ReferenceResolver())
    assert value.value == "g"
    assert value.value_identifier == "http://example.org/g"
    assert resolve_measured_unit([value]).value == "g"
