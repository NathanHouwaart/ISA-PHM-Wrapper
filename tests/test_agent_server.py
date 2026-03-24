from __future__ import annotations

import pytest

from isa_phm.agent_server import (
    AgentServerConfig,
    _openai_tool_specs,
    _parse_tool_arguments,
)


class TestParseToolArguments:
    def test_none_or_empty_returns_empty_dict(self):
        assert _parse_tool_arguments(None) == {}
        assert _parse_tool_arguments("") == {}
        assert _parse_tool_arguments("   ") == {}

    def test_valid_json_object_returns_dict(self):
        parsed = _parse_tool_arguments('{"a": 1, "b": "x"}')
        assert parsed == {"a": 1, "b": "x"}

    def test_non_object_json_raises(self):
        with pytest.raises(ValueError, match="JSON object"):
            _parse_tool_arguments("[1, 2, 3]")


class TestOpenAIToolSpecs:
    def test_contains_expected_tool_names(self):
        specs = _openai_tool_specs()
        names = [spec["function"]["name"] for spec in specs]
        assert names == ["ai_context", "validate_dataset", "load_dataframe_with_meta"]


class TestAgentServerConfig:
    def test_from_env_requires_isa_json(self, monkeypatch):
        monkeypatch.delenv("ISA_PHM_JSON", raising=False)
        with pytest.raises(ValueError, match="ISA_PHM_JSON"):
            AgentServerConfig.from_env()
