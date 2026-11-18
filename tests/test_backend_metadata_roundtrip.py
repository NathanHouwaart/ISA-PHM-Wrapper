"""Cross-repository contract checks; run when the backend checkout is available."""

import io
import copy
import json
import sys
from pathlib import Path

import pytest

from isa_phm.capabilities import resolve_measured_unit, resolve_sampling_frequency
from isa_phm.extractor import MetadataExtractor
from isa_phm import ISAWrapper
from isa_phm.errors import DataFileError

BACKEND = Path(__file__).resolve().parents[2] / "ISA-PHM-Backend"
pytestmark = pytest.mark.skipif(not (BACKEND / "app" / "converter" / "entrypoint.py").exists(), reason="Backend checkout is unavailable")


def test_processed_only_missing_raw_placeholder_roundtrips_and_loads(tmp_path):
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / 'tests/fixtures/minimal_payload.json').read_text(encoding='utf-8-sig'))
        run = payload['studies'][0]['assay_details'][0]['runs'][0]
        run['raw_file_name'] = ''
        run['processed_file_name'] = 'Data/processed.csv'
        output = json.loads(json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder))
        output = json.loads(json.dumps(isajson.load(io.StringIO(json.dumps(output))), cls=isajson.ISAJSONEncoder))
        (tmp_path / 'Data').mkdir()
        (tmp_path / 'Data/processed.csv').write_text('time,value\n0,1\n0.0000390625,2\n', encoding='utf-8')
        path = tmp_path / 'example.json'
        path.write_text(json.dumps(output), encoding='utf-8')
        wrapper = ISAWrapper(path)
        assay = wrapper.study(1).assay(1)
        record = wrapper._investigation.studies[0].assays[0].runs[0]
        assert record.raw_file is None
        assert resolve_sampling_frequency(record.measurement_params).value == 25600
        assert assay.load_dataframe(file_type='processed')['value'].tolist() == [1, 2]
        assert assay.load_dataframe(file_type='auto')['value'].tolist() == [1, 2]
        with pytest.raises(DataFileError, match="no 'raw' data file"):
            assay.load_dataframe(file_type='raw')
        assert not any('no path/name' in item.message for item in wrapper._repair_log.actions)
    finally:
        sys.path.remove(str(BACKEND))


@pytest.mark.parametrize("status", [None, "Unknown", "Not applicable"])
def test_backend_isa_wrapper_roundtrip(status):
    pytest.importorskip("isatools")
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / "tests/fixtures/minimal_payload.json").read_text(encoding="utf-8-sig"))
        study = payload["studies"][0]
        definitions = study["used_setup"]["measurementProtocols"][0]["parameters"]
        definitions.append({"id": "measured-unit", "name": "Measured Unit"})
        mappings = study["assay_details"][0]["measurement_protocols"]
        mappings[0]["value"] = [status, ""] if status else ["25600", "Hz"]
        unit_mapping = {
            "sourceId": "sensor-1", "targetId": "measured-unit", "protocolId": "mp-1",
            "value": [status or "g", ""],
        }
        if status is None:
            unit_mapping["valueIdentifier"] = "https://example.org/unit/standard-gravity"
        mappings.append(unit_mapping)

        serialized = json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder)
        assert not isajson.validate(io.StringIO(serialized))["errors"]
        assert isajson.load(io.StringIO(serialized)) is not None
        result = MetadataExtractor().extract(json.loads(serialized))
        params = result.studies[0].assays[0].runs[0].measurement_params
        unit = next(item for item in params if item.parameter_name == "Measured Unit")
        assert unit.value == (status or "g")
        assert unit.value_identifier == (None if status else "https://example.org/unit/standard-gravity")
        assert resolve_measured_unit(params).available is (status is None)
        assert resolve_sampling_frequency(params).available is (status is None)
    finally:
        sys.path.remove(str(BACKEND))


def test_comma_decimal_sampling_rate_survives_backend_conversion():
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / 'tests/fixtures/minimal_payload.json').read_text(encoding='utf-8-sig'))
        payload['studies'][0]['assay_details'][0]['measurement_protocols'][0]['value'] = ['25,6', 'kHz']
        serialized = json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder)
        result = MetadataExtractor().extract(json.loads(serialized))
        params = result.studies[0].assays[0].runs[0].measurement_params
        assert resolve_sampling_frequency(params).value == 25600
    finally:
        sys.path.remove(str(BACKEND))


def test_protocol_unit_identifier_survives_backend_and_wrapper():
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / 'tests/fixtures/minimal_payload.json').read_text(encoding='utf-8-sig'))
        payload['studies'][0]['assay_details'][0]['measurement_protocols'][0]['unitIdentifier'] = 'http://qudt.org/vocab/unit/HZ'
        output = json.loads(json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder))
        params = MetadataExtractor().extract(output).studies[0].assays[0].runs[0].measurement_params
        rate = next(param for param in params if param.parameter_name == 'Sampling Rate')
        assert rate.unit == 'Hz'
        assert rate.unit_identifier == 'http://qudt.org/vocab/unit/HZ'
    finally:
        sys.path.remove(str(BACKEND))


def test_study_variable_unit_identifier_survives_isa_export():
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / 'tests/fixtures/minimal_payload.json').read_text(encoding='utf-8-sig'))
        payload['study_variables'][0]['unitIdentifier'] = 'http://qudt.org/vocab/unit/N'
        serialized = json.loads(json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder))
        unit = next(item for item in serialized['studies'][0]['unitCategories'] if item.get('annotationValue') == 'N')
        assert unit['termAccession'] == 'http://qudt.org/vocab/unit/N'
        factor = MetadataExtractor().extract(serialized).studies[0].factors[0]
        assert factor.unit == 'N'
        assert factor.unit_identifier == 'http://qudt.org/vocab/unit/N'
    finally:
        sys.path.remove(str(BACKEND))


@pytest.mark.parametrize('identifier', ['http://qudt.org/vocab/quantitykind/Acceleration', 'urn:isa-phm:quantity:quaternion-component', None])
def test_sensor_quantity_identifier_survives_isa_export(identifier):
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / 'tests/fixtures/minimal_payload.json').read_text(encoding='utf-8-sig'))
        sensor = payload['studies'][0]['assay_details'][0]['used_sensor']
        sensor['measurementType'] = 'Quaternion component' if identifier and identifier.startswith('urn:isa-phm:') else 'Acceleration'
        if identifier:
            sensor['quantityKindIdentifier'] = identifier
        serialized = json.loads(json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder))
        assay = serialized['studies'][0]['assays'][0]
        assert assay['measurementType']['termAccession'] == (identifier or '')
        extracted = MetadataExtractor().extract(serialized).studies[0].assays[0].sensor
        assert extracted.measurement_type == sensor['measurementType']
        assert extracted.quantity_kind_identifier == identifier
    finally:
        sys.path.remove(str(BACKEND))


def test_same_symbol_with_different_identifiers_is_not_merged():
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.context import ConversionContext
        from isatools.model import Study

        context = ConversionContext()
        study = Study()
        first = context.add_unit_to_study(study, 'g', 'http://qudt.org/vocab/unit/GM')
        second = context.add_unit_to_study(study, 'g', 'http://qudt.org/vocab/unit/G')
        assert first.id != second.id
        assert len(study.units) == 2
    finally:
        sys.path.remove(str(BACKEND))


def test_physical_device_links_multiple_sensor_outputs():
    pytest.importorskip('isatools')
    sys.path.insert(0, str(BACKEND))
    try:
        from app.converter.entrypoint import create_isa_data
        from isatools import isajson

        payload = json.loads((BACKEND / 'tests/fixtures/minimal_payload.json').read_text(encoding='utf-8-sig'))
        study = payload['studies'][0]
        first = study['assay_details'][0]
        first['used_sensor']['physicalDevice'] = {'id': 'device-1', 'name': 'DAQ', 'model': 'Model A', 'serialNumber': 'SN-1', 'datasheetPath': './Datasheets/spec.pdf'}
        first['used_sensor']['datasheetPath'] = './Datasheets/sensor.pdf'
        second = copy.deepcopy(first)
        second['assay_file_name'] = 'se02'
        second['used_sensor']['id'] = 'sensor-2'
        second['used_sensor']['alias'] = 'S2'
        second['measurement_protocols'][0]['sourceId'] = 'sensor-2'
        second['processing_protocols'][0]['sourceId'] = 'sensor-2'
        study['assay_details'].append(second)
        study['used_setup']['sensors'].append(second['used_sensor'])
        second_study = copy.deepcopy(study)
        second_study['id'] = 'study-2'
        second_study['name'] = 'Study 2'
        payload['studies'].append(second_study)
        data = json.dumps(create_isa_data(payload), cls=isajson.ISAJSONEncoder)
        assert not isajson.validate(io.StringIO(data))['errors']
        extracted = MetadataExtractor().extract(json.loads(data))
        sensors = [assay.sensor for assay in extracted.studies[0].assays]
        assert [sensor.physical_device_id for sensor in sensors] == ['device-1', 'device-1']
        assert all(sensor.physical_device_model == 'Model A' and sensor.physical_device_serial_number == 'SN-1' for sensor in sensors)
        assert all(sensor.physical_device_datasheet == './Datasheets/spec.pdf' for sensor in sensors)
        assert all(sensor.sensor_datasheet == './Datasheets/sensor.pdf' for sensor in sensors)
        assert all(assay.sensor.physical_device_id == 'device-1' for assay in extracted.studies[1].assays)
    finally:
        sys.path.remove(str(BACKEND))
