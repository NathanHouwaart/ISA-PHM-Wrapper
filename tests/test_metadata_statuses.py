import pytest

from isa_phm.capabilities import resolve_measured_unit, resolve_sampling_frequency
from isa_phm.schemas import ParameterValue


@pytest.mark.parametrize("name", ["fs", "Sample Rate", "Sampling Frequency", "Smpl Rte"])
def test_sampling_rate_aliases_are_not_interpreted(name):
    result = resolve_sampling_frequency([ParameterValue(parameter_name=name, value=25600, unit="Hz")])
    assert not result.available


def test_standard_sampling_rate_is_read_explicitly():
    result = resolve_sampling_frequency([ParameterValue(parameter_name="Sampling Rate", value=25.6, unit="kHz")])
    assert result.available and result.value == 25600


@pytest.mark.parametrize("status", ["Unknown", "Not applicable"])
def test_explicit_status_is_not_a_measured_unit(status):
    result = resolve_measured_unit([ParameterValue(parameter_name="Measured Unit", value=status)])
    assert not result.available
    assert status.lower() in result.reason


@pytest.mark.parametrize("status", ["Unknown", "Not applicable"])
def test_explicit_sampling_status_is_not_an_invalid_number(status):
    result = resolve_sampling_frequency([ParameterValue(parameter_name="Sampling Rate", value=status)])
    assert not result.available
    assert status.lower() in result.reason


def test_status_conflicting_with_known_unit_is_not_silently_accepted():
    result = resolve_measured_unit([
        ParameterValue(parameter_name="Measured Unit", value="Unknown"),
        ParameterValue(parameter_name="Measured Unit", value="g"),
    ])
    assert not result.available
    assert "Conflicting" in result.reason


def test_other_parameter_units_are_not_inferred_as_measured_unit():
    result = resolve_measured_unit([
        ParameterValue(parameter_name="Sampling Frequency", value=25.6, unit="kHz"),
        ParameterValue(parameter_name="Sensor Orientation", value=90, unit="°"),
        ParameterValue(parameter_name="Some Unit", value="g"),
    ])
    assert not result.available
    assert "not provided" in result.reason


@pytest.mark.parametrize("value", ["g", "m/s²", "1", "custom-unit"])
def test_explicit_unit_is_preserved_without_catalogue_guessing(value):
    result = resolve_measured_unit([
        ParameterValue(parameter_name="  measured   UNIT  ", value=value),
        ParameterValue(parameter_name="Sensor Orientation", value=90, unit="°"),
    ])
    assert result.available
    assert result.value == value


def test_conflicting_explicit_units_are_unavailable():
    result = resolve_measured_unit([
        ParameterValue(parameter_name="Measured Unit", value="g"),
        ParameterValue(parameter_name="Measured Unit", value="m/s²"),
    ])
    assert not result.available
    assert "Conflicting" in result.reason
