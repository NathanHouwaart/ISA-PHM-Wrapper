import pytest

from isa_phm.utils import is_unavailable_raw_placeholder


@pytest.mark.parametrize(
    ('record', 'expected'),
    [
        ({'type': 'Raw Data File', 'name': '', 'comments': [{'name': 'availability', 'value': 'not included'}]}, True),
        ({'type': 'Raw Data File', 'name': 'Data/actual.csv', 'comments': [{'name': 'availability', 'value': 'not included'}]}, False),
        ({'type': 'Raw Data File', 'name': '', 'filename': 'Data/actual.csv', 'comments': [{'name': 'availability', 'value': 'not included'}]}, False),
        ({'type': 'Raw Data File', 'name': '', 'comments': []}, False),
        ({'type': 'Derived Data File', 'name': '', 'comments': [{'name': 'availability', 'value': 'not included'}]}, False),
        ({'type': 'Raw Data File', 'name': '', 'comments': [{'name': 'availability', 'value': 'unknown'}]}, False),
    ],
)
def test_recognizes_only_explicit_pathless_raw_placeholders(record, expected):
    assert is_unavailable_raw_placeholder(record) is expected
