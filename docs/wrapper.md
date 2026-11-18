# ISA-PHM Python Wrapper — Reference Guide

> **Version**: current (post-session-3 API)
> **Source root**: `python-wrapper/`
> **Entry point**: `from isa_phm import ISAWrapper`

Measured signal units come only from an explicit `Measured Unit` measurement
parameter (case and surrounding/repeated whitespace are ignored). Parameter
units such as the degrees of sensor orientation are not signal units. Missing,
unknown, not-applicable or conflicting answers do not yield a unit. Aggregate
assay and sensor-comparison labels show a unit only when the relevant runs or
sensors agree. Unit values are preserved, including custom values; this does
not validate them against a catalogue or convert them.

---

## Contents

1. [Quick Start](#1-quick-start)
2. [Object Hierarchy](#2-object-hierarchy)
3. [ISAWrapper](#3-isawrapper)
4. [StudyProxy](#4-studyproxy)
5. [AssayProxy](#5-assayproxy)
6. [AssayGroup & _AssayGroupBinding](#6-assaygroup--_assaygroupbinding)
7. [RunProxy](#7-runproxy)
8. [Plots](#8-plots)
9. [Error Types](#9-error-types)
10. [Notebooks](#10-notebooks)

---

## 1. Quick Start

```python
from pathlib import Path
from isa_phm import ISAWrapper
from bokeh.io import output_notebook
from bokeh.plotting import show

output_notebook()   # once per session -- enables in-notebook Bokeh rendering

wrapper = ISAWrapper(
    "path/to/i_investigation.json",
    data_root="path/to/data/",
)

# Inspect the dataset
print(wrapper.summary())
wrapper.list_studies()

# Navigate: study -> assay -> data
study  = wrapper.study(1)               # 1-based index, title, or UUID
assay  = study.assay(1)                 # 1-based index or assay_id string
df     = assay.load_dataframe()         # loads processed file by default

# Plot
show(assay.plot_lifecycle())            # all-run scalar feature curve
show(assay.plot_frequency_domain(fs=20_000))
show(assay.plot_spectrogram(fs=20_000)) # STFT time-frequency heatmap

# Explicit feature table; review provenance and target before modelling
table = assay.feature_table(features=["rms", "peak2peak"], label_column="fv_Bearing Lifetime")
```

---

### Timestamps and sampling rate

Loaded timestamps retain the values in the CSV; the wrapper does not shift or
rescale them based on sampling-rate metadata. When a recognized sampling rate
disagrees with the median timestamp spacing by more than 1%, the wrapper logs a
warning. This comparison assumes timestamps are in seconds: a warning can also
reflect different time units or irregular sampling, not necessarily bad data.
No comparison is made when the sampling rate is missing or ambiguous.

If a zero-based time axis is appropriate for your analysis, create it explicitly:

```python
df = assay.load_dataframe().copy()
df["time"] = df["time"] - df["time"].dropna().iloc[0]
```

This is a change from earlier versions that automatically shifted timestamps and
rescaled them to match sampling-rate metadata. The original CSV is never modified.

### Processed-only data

For processed-only exports, the Backend can represent the unavailable raw input
with a pathless `Raw Data File` bearing the provisional comment
`availability: not included`. The wrapper treats that placeholder as an
unavailable raw file, while retaining the measurement protocol metadata and
loading the processed file. Explicit raw loading raises `DataFileError`; use
`file_type="processed"` or `file_type="auto"` for the available data.

This convention is a draft, not a claim of full ISA-PHM standard conformance.
Existing exports that incorrectly use the processed file as both raw and
processed are not automatically repaired; regenerate or review them.

## 2. Object Hierarchy

```
ISAWrapper                          <- your only import
  +-- .study(id) -> StudyProxy
        +-- .assay(id) -> AssayProxy
        |     +-- .run(id) -> RunProxy
        +-- .assay_group(group) -> _AssayGroupBinding

AssayGroup([ids], name)             <- study-agnostic descriptor, define once
  +-- .compare_with([StudyProxy, ...])
```

All `study()` and `assay()` calls accept:

- **1-based integer** -- `wrapper.study(1)`, `study.assay(3)`
- **String ID** -- exact `study_id` / `assay_id` UUID
- **Title / case-insensitive match** -- `wrapper.study("Bearing 2_1")`

---

## 3. ISAWrapper

### Constructor

```python
ISAWrapper(
    path,                                   # str | Path -- ISA-JSON file
    data_root=None,                         # str | Path | None -- CSV root dir
    auto_fix=True,                          # apply 5 safe preprocessor repairs
    strict_validation=True,                 # isatools schema check
    cache_maxsize=100,                      # DataFrames held in FIFO cache
    enable_chunked_large_file_mode=True,    # stream large CSV files
    large_file_threshold_mb=64.0,
    chunk_rows=250_000,
    csv_bad_lines="error",                  # "error" | "warn" | "skip"
    plot_config=None,                       # PlotConfig override
    semantic_config_path=None,              # JSON semantic alias overrides
)
```

### Methods

| Method | Returns | Notes |
|---|---|---|
| `summary()` | `DataFrame` | One-row investigation summary |
| `extensive_summary()` | `dict[str, DataFrame]` | Keys: `investigation`, `studies`, `assays`, `factors`, `contacts`, `publications` |
| `investigation_overview()` | `InvestigationOverview` | Pydantic model |
| `list_studies()` | `list[StudySummary]` | No file I/O |
| `study(id)` | `StudyProxy` | id = int (1-based), UUID, or title |
| `contacts()` | `DataFrame` | Investigation contacts |
| `publications()` | `DataFrame` | Investigation publications |
| `compare_studies(study_ids, assay_id=, assay_group=, file_type=, n_workers=)` | `dict[str, DataFrame]` | Lifecycle features across multiple studies |
| `semantic_manifest(strict=, ...)` | `SemanticManifest` | Normalized factor/parameter labels |
| `ai_context(include_semantics=, include_validation=)` | `dict` | JSON-safe payload for LLM pipelines |
| `list_tools()` | `list[dict]` | AI tool schemas |
| `call_tool(name, args)` | `dict` | Invoke one AI tool by name |
| `repair_log` *(property)* | `RepairLog` | Preprocessing repairs that were applied |
| `source_path` *(property)* | `Path` | Absolute path to the ISA-JSON file |
| `investigation` *(property)* | `InvestigationModel` | Full extracted domain model |

### `compare_studies` example

```python
# Single sensor, multiple studies:
lc_dict = wrapper.compare_studies([1, 2, 3], assay_id=1, file_type="raw")
show(wrapper.study(1)._plotter.plot_multi_lifecycle(lc_dict, feature="rms"))

# Sensor group, multiple studies:
from isa_phm import AssayGroup
PE = AssayGroup([9, 10, 11], name="PE Accel XYZ")
lc_dict = wrapper.compare_studies(["Bearing 2_1", "Bearing 2_2"], assay_group=PE)
```

---

## 4. StudyProxy

Acquired via `wrapper.study(id)`.

### Properties

| Property | Type | Description |
|---|---|---|
| `title` | `str` | Study title |
| `has_runs` | `bool` | True when `run_count > 1` (prognostic study) |
| `run_count` | `int` | Total number of runs across all assays |

### Inspection methods

| Method | Returns | Notes |
|---|---|---|
| `overview()` | `StudyOverview` | Full summary including assay + factor lists |
| `list_assays()` | `list[AssaySummary]` | Sensor channels -- no file I/O |
| `list_factors()` | `list[FactorSummary]` | Experimental factors |
| `variable_overview()` | `list[DataFrame]` | One DataFrame per unique condition (factor x value pairs) |
| `test_matrix()` | `DataFrame` | Factor x run pivot table |
| `operating_conditions()` | `DataFrame` | Subset of `test_matrix()`: operating-condition factors only |
| `fault_conditions()` | `DataFrame` | Subset of `test_matrix()`: fault/damage/RUL factors only |
| `get_fault_labels(assay_id=)` | `DataFrame` | Fault factor values per (assay, run) pair |
| `sensor_catalog()` | `DataFrame` | One row per assay with sensor metadata + inferred fs/unit |
| `semantic_factors()` | `list[SemanticField]` | Normalized semantic labels for this study's factors |

### Navigation

```python
assay   = study.assay(1)             # AssayProxy
binding = study.assay_group(group)   # _AssayGroupBinding
```

### Cross-study analysis

```python
# Shortcut: all assays in group, this study only
lc = study.lifecycle_features(group=PE, file_type="raw")

# Compare this study against others (self is always first)
lc = study.compare_with([s2, s3], assay_id=1, file_type="raw")
```

### Multi-sensor helpers

```python
# Wide DataFrame: time + one column per sensor alias
df_wide = study.load_multi_sensor_dataframe(sensors=["vib_h", "vib_v"], run_id="run_05")

# Full time-series export annotated with ISA metadata + factor labels
df_labeled = study.export_labeled_dataset(file_type="raw", sensors=["vib_h"])

# Interactive boxplot comparing amplitude across all sensors
show(study.plot_sensor_boxplot(outlier_upper=1e7))
```

### Sensor correlation plots

```python
# Heatmap: how RMS co-varies across sensors over the lifecycle
show(study.plot_sensor_lifecycle_correlation(feature="rms"))

# Normalised cross-correlation between two sensors (single run)
show(study.plot_cross_correlation(1, 2, run_id="run_05"))
```

---

## 5. AssayProxy

Acquired via `study.assay(id)`.

### Properties

| Property | Type | Description |
|---|---|---|
| `assay_id` | `str` | Assay UUID string |
| `run_count` | `int` | Number of runs |

### Inspection

| Method | Returns | Notes |
|---|---|---|
| `overview()` | `AssayOverview` | Sensor metadata + run list |
| `list_runs()` | `list[RunSummary]` | All runs; `[]` for single-run assays |
| `columns()` | `list[str]` | Standard DataFrame column names |
| `sensor_info()` | `dict` | All known metadata for this channel |
| `list_measurement_params(run_id=)` | `DataFrame` | Protocol measurement parameters |
| `list_processing_params(run_id=)` | `DataFrame` | Protocol processing parameters |
| `semantic_parameters(run_id=)` | `dict[str, list[SemanticField]]` | Normalized parameter labels |

### Data loading

```python
df  = assay.load_dataframe(run_id=None, file_type="processed")
# file_type: "raw" | "processed" | "auto"
# run_id: omit for single-run (diagnostic) assays

df, meta = assay.load_dataframe_with_meta(run_id=..., file_type=...)
```

**Standard DataFrame columns** (always present):
`time, value, study_id, assay_id, run_id, sensor_alias, measurement_type, unit, file_type`

### Lifecycle features

```python
# One row per run, scalar features for all runs
lc = assay.lifecycle_features(file_type="processed", n_workers=None)
# Columns: run_id, run_number, study_id, assay_id,
#          rms, max, mean, peak2peak, kurtosis, std, crest_factor, skewness,
#          fv_<factor_name>...

# Feature table (not automatically model-ready)
df_ml = assay.feature_table(
    label_column="fv_Bearing Lifetime",   # renamed to "target"
    features=["rms", "peak2peak"],        # researcher-selected features
    file_type="processed",
)
X = df_ml[["rms", "peak2peak"]]  # exclude identifiers and provenance
y = df_ml["target"]
```

### Quality checks

```python
# Missing value summary
report = assay.missing_values_report(run_id=..., file_type="processed")
# report.n_missing, .pct_missing, .by_column

# Outlier detection (sample-level, within one run)
report = assay.detect_outliers(run_id=..., method="iqr", threshold=3.0, upper=1e7)
# report.n_outliers, .pct_outliers, .by_column

# Outlier correction -> returns a cleaned copy of df
df_clean = assay.fix_outliers(df, method="iqr", threshold=3.0, strategy="clip")
# strategy: "clip" | "nan" | "drop"

# Missing value imputation -> returns a filled copy of df
df_filled = assay.fill_missing_values(df, strategy="interpolate")
# strategy: "interpolate" | "ffill" | "bfill" | "mean" | "zero"
```

> **Outlier vs Lifecycle anomaly detection**
>
> `detect_outliers` / `fix_outliers` are **sample-level** -- they find individual
> bad readings within one run's time series (e.g. sensor overflow spikes).
> They do **not** identify which *run numbers* mark a health-state change.
> Time-series anomaly detection at the lifecycle/run level is planned for a
> future release.

### Navigation into runs

```python
run = assay.run("run_05")   # -> RunProxy
```

### Plots

See [Section 8 -- Plots](#8-plots) for full parameter lists.

```python
show(assay.plot_distribution(run_id="run_01"))
show(assay.plot_lifecycle(feature="kurtosis"))
show(assay.plot_frequency_domain(fs=20_000))
show(assay.plot_psd(fs=20_000, nperseg=1024))
show(assay.plot_spectrogram(fs=20_000, nperseg=256, overlap=0.75))
show(assay.plot_waterfall(n_runs=10, fs=20_000))
show(assay.plot_correlation())
show(assay.plot_variability())
show(assay.plot_missing_values())
show(assay.plot_timeseries(show_outliers=True, outlier_upper=1e7))
show(assay.plot_outlier_comparison(df, df_clean))
```

---

## 6. AssayGroup & _AssayGroupBinding

`AssayGroup` is a **frozen, immutable descriptor** -- define it once at the top of a notebook and pass it wherever needed. It holds no data and is not bound to any study.

```python
from isa_phm import AssayGroup

PE_ACCEL = AssayGroup([9, 10, 11], name="PE Accelerometer (X/Y/Z)")
MEMS     = AssayGroup([21, 22, 23], name="MEMS Accelerometer")
```

### `AssayGroup` API

| Attribute | Type | Description |
|---|---|---|
| `name` | `str` | Human-readable label used in plot legends / dict keys |
| `assay_ids` | `tuple[str | int, ...]` | Immutable tuple of assay IDs (1-based ints or strings) |

| Method | Returns | Notes |
|---|---|---|
| `compare_with(studies, file_type=, n_workers=)` | `dict[str, DataFrame]` | Lifecycle features for all provided studies; fully explicit -- no hidden binding |

```python
# Multi-study comparison -- explicit, no implicit self-inclusion
lc = PE_ACCEL.compare_with([s1, s2, s3], file_type="raw")

# Single-study -- shortcut on StudyProxy
lc = s1.lifecycle_features(group=PE_ACCEL)

# Or via wrapper
lc = wrapper.compare_studies(["Bearing 2_1", "Bearing 2_2"], assay_group=PE_ACCEL)
```

### `_AssayGroupBinding` (internal)

Bound to a specific study via `study.assay_group(group)`. Provides `lifecycle_features()` for a single study. Not intended for direct instantiation.

```python
binding = study.assay_group(PE_ACCEL)
lc = binding.lifecycle_features(file_type="raw", n_workers=4)
# -> dict[str, DataFrame] keyed by "StudyTitle -- sensor_alias"
```

Result dict keys are always `"<StudyTitle> -- <sensor_alias>"`.

---

## 7. RunProxy

Acquired via `assay.run("run_05")` -- represents a single acquisition run.

| Method | Returns | Notes |
|---|---|---|
| `overview()` | `RunOverview` | Run metadata including factor values |
| `load_dataframe(file_type=)` | `DataFrame` | File I/O -- same standard columns |
| `factor_values()` | `dict[str, Any]` | Factor values for this specific run |
| `list_measurement_params()` | `DataFrame` | Protocol measurement parameters |
| `list_processing_params()` | `DataFrame` | Protocol processing parameters |

---

## 8. Plots

All plots are **Bokeh figures** -- call `show(fig)` to display in a notebook. They never call `show()` internally. All accept optional `title`, `xlabel`, `ylabel`, `width`, `height` overrides.

### Plot summary

| Plot | Available on | Description |
|---|---|---|
| `plot_distribution(...)` | `AssayProxy` | Amplitude histogram + KDE for one run |
| `plot_lifecycle(feature=)` | `AssayProxy` | Scalar feature across all runs (lifecycle curve) |
| `plot_frequency_domain(fs=)` | `AssayProxy` | Single-sided FFT amplitude spectrum |
| `plot_psd(fs=, nperseg=)` | `AssayProxy` | Welch Power Spectral Density |
| `plot_spectrogram(fs=, nperseg=, overlap=)` | `AssayProxy` | STFT time x frequency heatmap |
| `plot_waterfall(n_runs=10, fs=)` | `AssayProxy` | Stacked FFT spectra across lifecycle runs |
| `plot_correlation(columns=)` | `AssayProxy` | Feature correlation heatmap across runs |
| `plot_variability(run_ids=)` | `AssayProxy` | Boxplot of amplitude by run |
| `plot_missing_values(...)` | `AssayProxy` | NaN presence heatmap |
| `plot_timeseries(show_outliers=, ...)` | `AssayProxy` | Time-domain waveform with optional outlier overlay |
| `plot_outlier_comparison(df, df_clean)` | `AssayProxy` | Linked before/after waveform panels |
| `plot_sensor_boxplot(outlier_upper=, ...)` | `StudyProxy` | Amplitude boxplot across all sensor channels |
| `plot_sensor_lifecycle_correlation(feature=)` | `StudyProxy` | RMS/feature correlation heatmap across sensors |
| `plot_cross_correlation(id1, id2, run_id=)` | `StudyProxy` | Normalised cross-correlation between two sensors |
| `plot_multi_lifecycle(dfs_dict, feature=)` | `ISAPlotter` (direct) | Overlay lifecycle curves for multiple assays |

### Key plot parameters

**`plot_frequency_domain`**
```python
assay.plot_frequency_domain(
    df=None,          # pre-loaded/cleaned DataFrame (optional)
    run_id=None,
    fs=20_000,        # Hz -- auto-inferred from ISA-JSON if None
    column="value",
    log_scale=True,   # True = dB scale; False = linear amplitude
)
```

**`plot_psd`**
```python
assay.plot_psd(
    run_id=None,
    fs=20_000,
    nperseg=1024,     # Welch segment length
    window="hann",
)
```

**`plot_spectrogram`**
```python
assay.plot_spectrogram(
    run_id=None,
    fs=20_000,
    column="value",
    nperseg=256,      # STFT window length
    overlap=0.75,     # fraction of window overlap (0-1)
)
```

**`plot_waterfall`**
```python
assay.plot_waterfall(
    run_ids=None,     # explicit list, or omit to sample evenly
    n_runs=10,        # number of evenly-spaced runs when run_ids=None
    fs=20_000,
    column="value",
    nperseg=1024,
    offset_scale=0.3, # vertical spacing between traces
)
```

**`plot_timeseries`**
```python
assay.plot_timeseries(
    run_id=None,
    show_outliers=True,
    outlier_method="iqr",      # "iqr" | "zscore" | "fixed"
    outlier_threshold=3.0,
    outlier_upper=1e7,         # hard threshold -- overrides method
    max_points=10_000,
)
```

---

## 9. Error Types

All exceptions inherit from `ISAPHMError`.

| Exception | When raised |
|---|---|
| `ParseError` | Malformed JSON |
| `ValidationError` | ISA schema violation |
| `PreprocessingError` | Unfixable ISA-JSON issue |
| `ExtractionError` | Valid ISA but not PHM-compatible |
| `DataFileError` | File not found or unreadable |
| `StudyNotFoundError` | `study_id` not in dataset |
| `AssayNotFoundError` | `assay_id` not in study |
| `RunNotFoundError` | `run_id` not in assay |
| `AmbiguousRunError` | Multi-run assay, `run_id` omitted |
| `PlotError` | Empty/invalid data for the requested plot |

Every message includes the offending ID and lists valid alternatives.

---

## 10. Notebooks

Notebooks live under `python-wrapper/notebooks/`:

```
00_getting_started/
  01_getting_started.ipynb          <- minimal end-to-end example

01_api_basics/
  10_basic_load_and_inspect.ipynb   <- ISAWrapper, summary, list_studies
  11_basic_semantic_normalization.ipynb
  12_basic_plot_and_export.ipynb

02_features/
  20_lifecycle_features.ipynb       <- lifecycle_features(), plot_lifecycle()
  21_compare_studies.ipynb          <- compare_studies(), plot_multi_lifecycle()
  22_time_domain.ipynb              <- plot_timeseries(), outlier detection/correction
  23_spectral_analysis.ipynb        <- FFT, PSD, spectrogram, waterfall
  24_correlation.ipynb              <- plot_correlation(), plot_cross_correlation()
  25_assay_groups.ipynb             <- AssayGroup, compare_with(), lifecycle_features(group=)

03_workflows/
  02_diagnostic_workflow.ipynb      <- complete diagnostic dataset walkthrough
  03_prognostic_workflow.ipynb      <- complete prognostic dataset walkthrough
```

### WRAPPER_ROOT in notebooks

All notebooks resolve the package root with:

```python
from pathlib import Path
WRAPPER_ROOT = Path("../..").resolve()   # notebooks are two levels deep
```

---

## Appendix -- Standard DataFrame Columns

Every `load_dataframe()` call returns a DataFrame with these columns (some may be `None` when not available in the source file):

| Column | Type | Description |
|---|---|---|
| `time` | float | Sample timestamp in seconds |
| `value` | float | Measurement value |
| `study_id` | str | Study UUID |
| `assay_id` | str | Assay UUID |
| `run_id` | str | Run identifier |
| `sensor_alias` | str | Human-readable sensor name |
| `measurement_type` | str | e.g. `"Vibration"`, `"Temperature"` |
| `unit` | str | Physical unit (e.g. `"g"`, `"C"`) |
| `file_type` | str | `"raw"` or `"processed"` |

## Appendix -- Lifecycle Feature Columns

`assay.lifecycle_features()` returns a DataFrame with one row per run:

| Column | Description |
|---|---|
| `run_id` | Run identifier |
| `run_number` | Integer run index |
| `study_id` / `assay_id` | Source identifiers |
| `rms` | Root mean square |
| `max` | Maximum amplitude |
| `mean` | Mean amplitude |
| `peak2peak` | Peak-to-peak range |
| `kurtosis` | Kurtosis (sensitive to impulses) |
| `std` | Standard deviation |
| `crest_factor` | Peak / RMS ratio |
| `skewness` | Signal asymmetry |
| `fv_<factor_name>` | One column per ISA factor (e.g. `fv_Bearing Lifetime`) |
