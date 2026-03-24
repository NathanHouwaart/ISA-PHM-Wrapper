# AI Agent Guide

This wrapper includes a JSON-only tool interface so an LLM/agent can call safe, deterministic endpoints.

## Design Goal

Agent integrations should avoid passing Python objects directly.  
Use only JSON input/output via:
- `wrapper.list_tools()`
- `wrapper.call_tool(name, args)`

## Available Tools

- `ai_context`
  - Returns metadata-only normalized dataset context.
- `validate_dataset`
  - Returns structured issues (`error` / `warning` / `info`).
- `load_dataframe_with_meta`
  - Returns load metadata plus JSON-safe dataframe summary/head.

## Recommended Agent Flow

1. Initialize wrapper.
2. Call `validate_dataset` first.
3. Call `ai_context` for metadata grounding.
4. Call `load_dataframe_with_meta` only for explicit run-level inspection.

## Minimal Example

```python
from isa_phm import ISAWrapper

wrapper = ISAWrapper(
    "path/to/i_investigation.json",
    data_root="path/to/data",
    strict_validation=False,
    csv_bad_lines="error",
)

print(wrapper.list_tools())

validation = wrapper.call_tool(
    "validate_dataset",
    {"check_files": True, "semantic_strict": False},
)

context = wrapper.call_tool(
    "ai_context",
    {"include_semantics": True, "include_validation": False},
)

run_payload = wrapper.call_tool(
    "load_dataframe_with_meta",
    {
        "study_id": "Case 01",
        "assay_id": "a_st01_se01",
        "run_id": "run_01",
        "file_type": "auto",
        "head_rows": 5,
    },
)
```

## Safety Notes

- Keep `csv_bad_lines="error"` in production unless you explicitly accept row loss.
- Do not infer unresolved publication authors; rely on `resolved_author_*` fields only.
- Treat `validate_dataset` issues as gating signals before generating insights.

## Versioning

- `ai_context["schema_version"]` is the contract version.
- Add/update golden snapshot tests whenever `ai_context` schema changes.
