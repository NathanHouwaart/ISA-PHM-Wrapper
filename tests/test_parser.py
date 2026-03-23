"""
Tests for ISAParser.

Covers:
- Valid JSON string round-trips.
- Malformed JSON → ParseError.
- strict=False skips isatools validation.
- File-based load for the real notebook fixtures (skipped if not present).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from isa_phm.errors import ParseError, ValidationError
from isa_phm.parser import ISAParser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_minimal_json(title: str = "Test") -> str:
    """Return a structurally valid ISA-JSON string (minimal)."""
    doc = {
        "comments": [
            {"name": "experiment_type", "value": "diagnostic-single"}
        ],
        "description": "Desc",
        "identifier": "test-001",
        "ontologySourceReferences": [],
        "people": [],
        "studies": [],
        "title": title,
    }
    return json.dumps(doc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestISAParserLoadString:
    def test_valid_json_returns_dict(self):
        parser = ISAParser(strict=False)
        result = parser.load_string(_valid_minimal_json())
        assert isinstance(result, dict)

    def test_title_preserved(self):
        parser = ISAParser(strict=False)
        result = parser.load_string(_valid_minimal_json(title="My Investigation"))
        assert result["title"] == "My Investigation"

    def test_malformed_json_raises_parse_error(self):
        parser = ISAParser(strict=False)
        with pytest.raises(ParseError):
            parser.load_string("{invalid json !!!")

    def test_empty_string_raises_parse_error(self):
        parser = ISAParser(strict=False)
        with pytest.raises(ParseError):
            parser.load_string("")

    def test_array_root_raises_parse_error(self):
        """ISA-JSON must be a top-level object, not an array."""
        parser = ISAParser(strict=False)
        with pytest.raises(ParseError):
            parser.load_string("[1, 2, 3]")


class TestISAParserLoadFile:
    def test_file_not_found_raises_parse_error(self, tmp_path):
        parser = ISAParser(strict=False)
        with pytest.raises(ParseError):
            parser.load(tmp_path / "does_not_exist.json")

    def test_valid_file_returns_dict(self, tmp_path):
        p = tmp_path / "i_test.json"
        p.write_text(_valid_minimal_json(), encoding="utf-8")
        parser = ISAParser(strict=False)
        result = parser.load(p)
        assert isinstance(result, dict)

    def test_malformed_file_raises_parse_error(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json at all", encoding="utf-8")
        parser = ISAParser(strict=False)
        with pytest.raises(ParseError):
            parser.load(p)

    def test_latin1_file_decoded(self, tmp_path):
        """Parser must fall back to Latin-1 encoding if UTF-8 fails."""
        p = tmp_path / "i_latin1.json"
        # Write a JSON with a Latin-1 byte sequence (é = 0xE9).
        text = '{"title": "Caf\xe9", "identifier": "x", "description": "", "comments": [], "ontologySourceReferences": [], "people": [], "studies": []}'
        p.write_bytes(text.encode("latin-1"))
        parser = ISAParser(strict=False)
        result = parser.load(p)
        assert "Caf" in result["title"]


class TestISAParserRealFixtures:
    def test_single_run_fixture_loads(self, single_run_isa_path):
        parser = ISAParser(strict=False)
        result = parser.load(single_run_isa_path)
        assert "studies" in result
        assert len(result["studies"]) >= 1

    def test_single_run_experiment_type(self, single_run_isa_path):
        parser = ISAParser(strict=False)
        raw = parser.load(single_run_isa_path)
        exp_types = [
            c["value"] for c in raw.get("comments", [])
            if c.get("name") == "experiment_type"
        ]
        assert exp_types == ["diagnostic-single"]

    def test_multi_run_fixture_loads(self, multi_run_isa_path):
        parser = ISAParser(strict=False)
        result = parser.load(multi_run_isa_path)
        assert "studies" in result

    def test_multi_run_experiment_type(self, multi_run_isa_path):
        parser = ISAParser(strict=False)
        raw = parser.load(multi_run_isa_path)
        exp_types = [
            c["value"] for c in raw.get("comments", [])
            if c.get("name") == "experiment_type"
        ]
        assert exp_types == ["diagnostic-multi"]
