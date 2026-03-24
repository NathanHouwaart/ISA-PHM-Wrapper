# Release Readiness Checklist

Use this checklist before cutting a public release.

## 1. Contract and API

- [ ] Confirm canonical API names are stable and documented.
- [ ] Confirm deprecations/breaking changes are listed in `CHANGELOG.md`.
- [ ] Confirm `ai_context` schema version is correct and documented.

## 2. Validation and Safety

- [ ] Run `validate_dataset()` smoke checks on representative datasets.
- [ ] Confirm strict CSV default (`csv_bad_lines="error"`) is unchanged.
- [ ] Confirm semantic strict-mode controls are tested.
- [ ] Confirm path resolution behavior is tested for traversal/symlink edge cases.

## 3. Tests

- [ ] Run full test suite:

```bash
python -m pytest -q
```

- [ ] Ensure golden snapshot tests pass (including `ai_context`).
- [ ] Ensure at least one `load_dataframe_with_meta(file_type="auto")` test per representative dataset category.

## 4. CI

- [ ] CI green on supported Python versions.
- [ ] CI workflow matches declared support matrix.
- [ ] No flaky tests in repeated runs.

## 5. Packaging

- [ ] Verify `pyproject.toml` version bump.
- [ ] Verify dependency bounds are intentional.
- [ ] Build package locally (`python -m build`) and inspect wheel/sdist.
- [ ] Publish to TestPyPI before public release.

## 6. Docs

- [ ] `README.md` quickstart works from clean environment.
- [ ] AI guide reflects current tool names/arguments.
- [ ] Notebook examples run without hidden setup assumptions.

## 7. Release Artifacts

- [ ] Tag created and pushed (e.g. `v0.3.0`).
- [ ] Release notes generated from `CHANGELOG.md`.
- [ ] Optional: attach benchmark notes if performance changes are included.
