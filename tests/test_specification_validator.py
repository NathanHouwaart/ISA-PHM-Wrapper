"""Integration with the separately distributed draft validator."""

import json
import sys
import types

import pytest

from isa_phm import ISAWrapper


def test_specification_checks_original_file_and_requested_root(minimal_single_run_isa_file, tmp_path):
    pytest.importorskip("validator.api", reason="Install the draft ISA-PHM validator")
    wrapper = ISAWrapper(minimal_single_run_isa_file, data_root=tmp_path, strict_validation=False)
    report = wrapper.validate_specification(checks=("layout",))
    assert report["profile"] == "draft"
    assert report["conformant"] is None
    assert report["checks"]["layout"]["status"] == "fail"
    assert report["not_selected"]
    json.dumps(report)


def test_shared_validator_receives_original_path_and_scope(minimal_single_run_isa_file, tmp_path, monkeypatch):
    captured = {}

    def validate(path, **kwargs):
        captured.update(path=path, **kwargs)
        return types.SimpleNamespace(to_dict=lambda: {"status": "incomplete", "checks": {}})

    monkeypatch.setitem(sys.modules, "validator", types.SimpleNamespace(validate_dataset=validate))
    wrapper = ISAWrapper(minimal_single_run_isa_file, data_root=tmp_path, strict_validation=False)
    report = wrapper.validate_specification(checks=("layout",))
    assert report["status"] == "incomplete"
    assert captured == {"path": minimal_single_run_isa_file, "dataset_root": tmp_path, "checks": ("layout",)}


def test_missing_validator_gives_install_instructions(minimal_single_run_isa_file, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "validator", None)
    wrapper = ISAWrapper(minimal_single_run_isa_file, data_root=tmp_path, strict_validation=False)
    with pytest.raises(RuntimeError, match="pip install"):
        wrapper.validate_specification()
