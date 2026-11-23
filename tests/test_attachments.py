import base64
import json

import pytest

from isa_phm.attachments import discover_attachments
from isa_phm.errors import DataFileError
from isa_phm import ISAWrapper


def test_discovery_preserves_context_and_reports_unsafe_and_missing(tmp_path):
    (tmp_path / "Images").mkdir()
    (tmp_path / "Datasheets").mkdir()
    (tmp_path / "Images" / "setup.png").write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/a2UAAAAASUVORK5CYII="))
    (tmp_path / "Datasheets" / "sensor.pdf").write_bytes(b"%PDF-1.4\n%%EOF")
    document = {"studies": [{"title": "Bearing A", "characteristicCategories": [{"@id": "#bearing", "characteristicType": {"annotationValue": "Bearing"}}], "materials": {"sources": [{"name": "Testbed", "comments": [{"name": "image", "value": "./Images/setup.png"}, {"name": "image", "value": "./Images/missing.jpg"}, {"name": "image", "value": "./Images/../../secret.png"}], "characteristics": [{"category": {"@id": "#bearing"}, "comments": [{"name": "datasheet", "value": "./Datasheets/sensor.pdf"}]}]}]}, "assays": [{"filename": "se01", "comments": [{"name": "sensor alias", "value": "Accel"}, {"name": "datasheet", "value": "./Datasheets/sensor.pdf"}, {"name": "physical device datasheet", "value": "https://example.com/file.pdf"}]}]}]}
    items = discover_attachments(document, tmp_path)
    assert [(x.kind, x.status) for x in items] == [("image", "available"), ("image", "missing"), ("image", "unsafe"), ("datasheet", "available"), ("datasheet", "available"), ("datasheet", "unsafe")]
    assert items[3].owner == "Testbed / Bearing"
    assert items[4].owner == "Accel"
    assert items[5].owner_type == "physical device"
    try:
        import IPython  # noqa: F401 - notebook previews are optional
    except ImportError:
        pass
    else:
        assert type(items[0].preview()).__name__ == "Image"
        assert type(items[3].preview()).__name__ == "IFrame"
    with pytest.raises(DataFileError, match="missing"):
        items[1].preview()
    with pytest.raises(DataFileError, match="unsafe"):
        items[2].preview()


def test_symlink_outside_dataset_is_not_previewed(tmp_path):
    root = tmp_path / "dataset"
    (root / "Images").mkdir(parents=True)
    secret = tmp_path / "outside.png"
    secret.write_bytes(b"not an image")
    try:
        (root / "Images" / "link.png").symlink_to(secret)
    except OSError:
        pytest.skip("Symbolic links unavailable")
    document = {"studies": [{"title": "A", "materials": {"sources": [{"comments": [{"name": "image", "value": "Images/link.png"}]}]}}]}
    assert discover_attachments(document, root)[0].status == "unsafe"


@pytest.mark.parametrize("alias_name, sheet_name", [("sensor alias", "physical device datasheet"), ("sensor_alias", "physical_device_datasheet")])
def test_device_datasheet_comment_names(tmp_path, alias_name, sheet_name):
    (tmp_path / "Datasheets").mkdir()
    (tmp_path / "Datasheets" / "device.pdf").write_bytes(b"pdf")
    assay = {"filename": "a", "comments": [{"name": alias_name, "value": "Accel"}, {"name": sheet_name, "value": "./Datasheets/device.pdf"}]}
    items = discover_attachments({"studies": [{"title": "S", "assays": [assay]}]}, tmp_path)
    assert len(items) == 1
    assert items[0].owner == "Accel" and items[0].status == "available"


def test_wrapper_attachment_filters_use_actual_study(tmp_path, minimal_single_run_isa):
    (tmp_path / "Datasheets").mkdir()
    (tmp_path / "Datasheets" / "sensor.pdf").write_bytes(b"%PDF-1.4\n%%EOF")
    assay = minimal_single_run_isa["studies"][0]["assays"][0]
    assay.setdefault("comments", []).append({"name": "datasheet", "value": "./Datasheets/sensor.pdf"})
    path = tmp_path / "dataset.json"
    path.write_text(json.dumps(minimal_single_run_isa), encoding="utf-8")
    wrapper = ISAWrapper(path, data_root=tmp_path, strict_validation=False)
    items = wrapper.attachments(study=1, kind="datasheet")
    assert len(items) == 1 and items[0].status == "available"
    assert wrapper.attachments(study=1, kind="image") == []
    with pytest.raises(ValueError):
        wrapper.attachments(kind="other")
