# Changelog

## 0.2.0 (2026-03-24)

- Breaking: removed `ISAWrapper.investigation_contacts()` and `ISAWrapper.investigation_publications()`.
  Use `ISAWrapper.contacts()` and `ISAWrapper.publications()` instead.
- Added deterministic AI export endpoint: `ISAWrapper.ai_context(...)`.
- Added dataset validation endpoint: `ISAWrapper.validate_dataset(...)`.
- Added strict semantic controls on `ISAWrapper.semantic_manifest(...)`.
- Added publication-contact author linking fields and live DataFile existence behavior.
