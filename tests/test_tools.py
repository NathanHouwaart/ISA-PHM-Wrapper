from __future__ import annotations

import json

from isa_phm import ISAWrapper


class TestToolRegistry:
    def test_list_tools_contains_expected_names(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        names = [tool["name"] for tool in wrapper.list_tools()]
        assert names == [
            "ai_context",
            "load_dataframe_with_meta",
            "validate_dataset",
        ]

    def test_call_ai_context_tool_returns_json_safe_payload(
        self, minimal_publication_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_publication_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        out = wrapper.call_tool(
            "ai_context",
            {"include_semantics": True, "include_validation": True},
        )
        assert out["tool"] == "ai_context"
        assert out["ok"] is True
        assert "result" in out
        json.dumps(out)

    def test_call_load_dataframe_with_meta_tool(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        out = wrapper.call_tool(
            "load_dataframe_with_meta",
            {
                "study_id": "Test Study",
                "assay_id": "a_st01_se01",
                "file_type": "auto",
                "head_rows": 3,
            },
        )
        assert out["ok"] is True
        result = out["result"]
        assert "metadata" in result
        assert result["metadata"]["resolved_file_type"] in {"raw", "processed"}
        assert "dataframe" in result
        assert result["dataframe"]["n_rows"] > 0
        assert len(result["dataframe"]["head"]) <= 3

    def test_unknown_tool_returns_structured_error(
        self, minimal_single_run_isa_file, tmp_path
    ):
        wrapper = ISAWrapper(
            minimal_single_run_isa_file,
            data_root=tmp_path,
            strict_validation=False,
        )
        out = wrapper.call_tool("nope_tool", {})
        assert out["ok"] is False
        assert out["error"]["type"] == "UnknownTool"
