# Changelog

## Unreleased

- Documentation and public onboarding:
  - Added root `README.md` with quickstart, API highlights, and safety defaults.
  - Added AI integration guide: `docs/AI_AGENT_GUIDE.md`.
  - Added release gate checklist: `docs/RELEASE_READINESS_CHECKLIST.md`.
  - Expanded `notebooks/01_api_basics/10_basic_load_and_inspect.ipynb` with JSON-only tool examples.
- AI interface:
  - Added JSON-in/JSON-out tool registry (`wrapper.list_tools()`, `wrapper.call_tool(...)`).
  - Added golden snapshot test for `ai_context` payload stability.
  - Added optional FastAPI agent server (`isa-phm-agent-server`) with `/tools`, `/tool`, and OpenAI-backed `/chat`.
- CSV loading controls:
  - Added `csv_bad_lines` configuration (`error`/`warn`/`skip`) on `ISAWrapper`/`DataIntegrator`.
  - Extended `DataLoadMetadata` with CSV diagnostics (`engine`, `sep`, `encoding`, detection source).
- Plot API usability:
  - Added optional plot override arguments (`title`, axis labels, width, height) across study/assay/run plot APIs.
- Tooling hardening:
  - Added `.gitignore` for Python cache and notebook artifacts.
  - Added CI workflow (`.github/workflows/ci.yml`) with a Python 3.10-3.13 test matrix.
  - Added dependency upper bounds and expanded optional dependency groups (`dev`, `docs`, `notebooks`).
  - Added `pytest-xdist` and `pytest-timeout` to dev dependencies.
- Performance and workflow:
  - Added thread-safe cache-key locking in `DataIntegrator` to reduce duplicate concurrent loads.
  - Added chunked large-file lifecycle feature mode with configurable threshold and chunk size.
  - Added benchmark script: `scripts/benchmark_chunked_mode.py`.
  - Standardized notebooks `11` and `12` to small-cell flows and added examples for `ai_context` and `validate_dataset`.

## 0.2.0 (2026-03-24)

- Breaking: removed `ISAWrapper.investigation_contacts()` and `ISAWrapper.investigation_publications()`.
  Use `ISAWrapper.contacts()` and `ISAWrapper.publications()` instead.
- Added deterministic AI export endpoint: `ISAWrapper.ai_context(...)`.
- Added dataset validation endpoint: `ISAWrapper.validate_dataset(...)`.
- Added strict semantic controls on `ISAWrapper.semantic_manifest(...)`.
- Added publication-contact author linking fields and live DataFile existence behavior.
