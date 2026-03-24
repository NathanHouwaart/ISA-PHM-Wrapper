# Changelog

## Unreleased

- Tooling hardening:
  - Added `.gitignore` for Python cache and notebook artifacts.
  - Added CI workflow (`.github/workflows/ci.yml`) with a Python 3.10-3.13 test matrix.
  - Added dependency upper bounds and expanded optional dependency groups (`dev`, `docs`, `notebooks`).
  - Added `pytest-xdist` and `pytest-timeout` to dev dependencies.

## 0.2.0 (2026-03-24)

- Breaking: removed `ISAWrapper.investigation_contacts()` and `ISAWrapper.investigation_publications()`.
  Use `ISAWrapper.contacts()` and `ISAWrapper.publications()` instead.
- Added deterministic AI export endpoint: `ISAWrapper.ai_context(...)`.
- Added dataset validation endpoint: `ISAWrapper.validate_dataset(...)`.
- Added strict semantic controls on `ISAWrapper.semantic_manifest(...)`.
- Added publication-contact author linking fields and live DataFile existence behavior.
