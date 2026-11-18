import pytest

from isa_phm import CATALOGUE_VERSION, lookup_quantity, lookup_unit
from isa_phm import catalogue


def test_exact_identifiers_from_shared_catalogues():
    assert CATALOGUE_VERSION == 'qudt-3.5.2'
    assert lookup_unit('http://qudt.org/vocab/unit/GM')['symbol'] == 'g'
    assert lookup_unit('http://qudt.org/vocab/unit/G')['symbol'] == 'G'
    assert lookup_quantity('http://qudt.org/vocab/quantitykind/Acceleration')['name'] == 'Acceleration'
    assert lookup_quantity('urn:isa-phm:quantity:quaternion-component')['name'] == 'Quaternion component'
    assert lookup_unit('g') is None
    assert lookup_quantity('urn:custom:unknown') is None


def test_callers_cannot_mutate_cached_catalogue():
    item = lookup_unit('http://qudt.org/vocab/unit/GM')
    item['symbol'] = 'incorrect'
    item['aliases'].append('incorrect')
    again = lookup_unit('http://qudt.org/vocab/unit/GM')
    assert again['symbol'] == 'g'
    assert 'incorrect' not in again['aliases']


def test_missing_shared_catalogue_fails_clearly(monkeypatch):
    catalogue._indexes.cache_clear()
    def missing(_name):
        raise RuntimeError('Shared specification catalogues are unavailable')
    monkeypatch.setattr(catalogue, '_read_catalogue', missing)
    with pytest.raises(RuntimeError, match='catalogues are unavailable'):
        lookup_unit('http://qudt.org/vocab/unit/G')
    catalogue._indexes.cache_clear()
