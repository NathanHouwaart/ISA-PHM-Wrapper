"""
DataIntegrator — load ISA-PHM measurement data files into pandas DataFrames.

Two-track access (SR-5: must handle >2500 runs)
------------------------------------------------
Track 1 — FIFO cache (interactive, single-run access)
    Each DataFrame is stored by key (assay_id, run_id, file_type).
    Max 100 entries; oldest evicted when full.
    Used by: AssayProxy.load_dataframe(), RunProxy.load_dataframe().

Track 2 — Streaming generator (lifecycle over many runs)
    Loads one run's CSV at a time, computes 8 scalar features, yields a dict.
    Never keeps more than one run's DataFrame in memory simultaneously.
    Used by: AssayProxy.lifecycle_features().

CSV loading contract (SR-2)
---------------------------
- Format: CSV or TSV only.
- Encoding: UTF-8 (Latin-1 fallback).
- Delimiter: auto-detected via pandas sep=None, engine="python".
- Shape: exactly 2 columns (time, value). First column = time, second = measurement.
- Result columns: [time, value, study_id, assay_id, run_id, sensor_alias,
                   measurement_type, file_type]
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Literal

import numpy as np
import pandas as pd

from .errors import AmbiguousRunError, DataFileError, RunNotFoundError
from .schemas import AssayModel, DataFile, RunRecord
from .utils import compute_features

logger = logging.getLogger("isa_phm")

# Standard DataFrame columns always present after integration.
STANDARD_COLUMNS = (
    "time",
    "value",
)


class _FIFOCache:
    """Simple FIFO eviction cache backed by an OrderedDict."""

    def __init__(self, maxsize: int = 100) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict = OrderedDict()

    def get(self, key: tuple) -> pd.DataFrame | None:
        return self._store.get(key)

    def put(self, key: tuple, df: pd.DataFrame) -> None:
        if key in self._store:
            return  # Already cached; don't re-insert.
        if len(self._store) >= self._maxsize:
            self._store.popitem(last=False)  # Remove oldest.
        self._store[key] = df

    def __contains__(self, key: tuple) -> bool:
        return key in self._store

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class DataIntegrator:
    """
    Load and cache ISA-PHM data files as enriched pandas DataFrames.

    Parameters
    ----------
    data_root : Path
        Base directory for resolving relative paths (passed through from
        the preprocessor; absolute paths are used as-is).
    cache_maxsize : int
        Maximum number of DataFrames held in the FIFO cache (default 100).
    """

    def __init__(self, data_root: Path, cache_maxsize: int = 100) -> None:
        self._data_root = Path(data_root)
        self._cache = _FIFOCache(maxsize=cache_maxsize)

    # ------------------------------------------------------------------
    # Public: single-run loading (Track 1, cached)
    # ------------------------------------------------------------------

    def load(
        self,
        assay: AssayModel,
        study_id: str,
        run_id: str | None = None,
        file_type: Literal["raw", "processed"] = "processed",
    ) -> pd.DataFrame:
        """
        Load a run's data file into a DataFrame.

        For diagnostic assays (1 run) run_id may be None — it is inferred
        automatically.  For multi-run assays run_id is required.

        Parameters
        ----------
        assay : AssayModel
        study_id : str
             Needed for metadata columns.
        run_id : str | None
        file_type : "raw" | "processed"

        Returns
        -------
        pd.DataFrame with columns STANDARD_COLUMNS.

        Raises
        ------
        AmbiguousRunError
            Multi-run assay and run_id is None.
        RunNotFoundError
            run_id not found in assay.runs.
        DataFileError
            File missing, empty path, or CSV load failure.
        """
        run = self._resolve_run(assay, run_id)
        cache_key = (assay.assay_id, run.run_id, file_type)

        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", cache_key)
            return cached

        df = self._load_run(assay, study_id, run, file_type)
        self._cache.put(cache_key, df)
        return df

    # ------------------------------------------------------------------
    # Public: streaming lifecycle features (Track 2, no cache)
    # ------------------------------------------------------------------

    def stream_lifecycle_features(
        self,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed"] = "processed",
    ) -> Generator[dict, None, None]:
        """
        Yield one feature dict per run without loading all runs simultaneously.

        Each yielded dict has keys:
            run_id, run_number, study_id, assay_id,
            rms, max, mean, peak2peak, kurtosis, std, crest_factor, skewness.

        Runs whose data file is missing or empty are skipped (logged as WARNING).

        Parameters
        ----------
        assay : AssayModel
        study_id : str
        file_type : "processed" (default) | "raw"

        Yields
        ------
        dict
        """
        for run in assay.runs:
            data_file = run.processed_file if file_type == "processed" else run.raw_file

            if data_file is None or not data_file.path:
                logger.warning(
                    "Run '%s' of assay '%s' has no %s data file path — skipping.",
                    run.run_id, assay.assay_id, file_type,
                )
                continue

            # Check cache first (may have been loaded interactively).
            cache_key = (assay.assay_id, run.run_id, file_type)
            cached = self._cache.get(cache_key)

            try:
                if cached is not None:
                    raw_values = self._get_value_column(cached)
                else:
                    df = self._read_csv(Path(data_file.path))
                    raw_values = self._get_value_column(df)
                    # Do NOT cache — streaming mode keeps only one DF alive.
            except DataFileError as exc:
                logger.warning(
                    "Skipping run '%s' of assay '%s': %s",
                    run.run_id, assay.assay_id, exc,
                )
                continue

            values = raw_values.dropna().to_numpy(dtype=np.float64)
            features = compute_features(values)

            yield {
                "run_id": run.run_id,
                "run_number": run.run_number,
                "study_id": study_id,
                "assay_id": assay.assay_id,
                **features,
                **{f"fv_{k}": v for k, v in run.factor_values.items()},
            }

    def _compute_run_features(
        self,
        run: "RunRecord",
        assay: AssayModel,
        study_id: str,
        file_type: str,
    ) -> dict | None:
        """
        Load one run's CSV and compute scalar features.

        Returns None if the run should be skipped (missing path or load error).
        Thread-safe: _read_csv uses no shared mutable state.
        """
        data_file = run.processed_file if file_type == "processed" else run.raw_file

        if data_file is None or not data_file.path:
            logger.warning(
                "Run '%s' of assay '%s' has no %s data file path — skipping.",
                run.run_id, assay.assay_id, file_type,
            )
            return None

        cache_key = (assay.assay_id, run.run_id, file_type)
        cached = self._cache.get(cache_key)

        try:
            if cached is not None:
                raw_values = self._get_value_column(cached)
            else:
                df = self._read_csv(Path(data_file.path))
                raw_values = self._get_value_column(df)
        except DataFileError as exc:
            logger.warning(
                "Skipping run '%s' of assay '%s': %s",
                run.run_id, assay.assay_id, exc,
            )
            return None

        values = raw_values.dropna().to_numpy(dtype=np.float64)
        features = compute_features(values)
        return {
            "run_id": run.run_id,
            "run_number": run.run_number,
            "study_id": study_id,
            "assay_id": assay.assay_id,
            **features,
            **{f"fv_{k}": v for k, v in run.factor_values.items()},
        }

    def lifecycle_features_df(
        self,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed"] = "processed",
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        """
        Load all runs and return a lifecycle feature summary DataFrame.

        Columns: run_id, run_number, study_id, assay_id,
                 rms, max, mean, peak2peak, kurtosis, std, crest_factor, skewness,
                 fv_<factor_name> for each factor.

        Parameters
        ----------
        n_workers : int | None
            Number of threads for parallel CSV loading.  None (default) lets
            the executor choose (``min(32, cpu_count + 4)``).  Pass 1 to
            force sequential loading.

        Returns an empty DataFrame if no runs could be loaded.
        """
        effective_workers = n_workers if n_workers is not None else min(32, (os.cpu_count() or 1) + 4)

        if effective_workers == 1 or len(assay.runs) <= 1:
            # Sequential path — preserves streaming memory behaviour.
            rows = list(self.stream_lifecycle_features(assay, study_id, file_type))
        else:
            rows_unordered: list[dict] = []
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(self._compute_run_features, run, assay, study_id, file_type): run
                    for run in assay.runs
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        rows_unordered.append(result)
            # Restore run order (futures complete in arbitrary order).
            rows = sorted(rows_unordered, key=lambda r: r["run_number"])

        if not rows:
            logger.warning(
                "No lifecycle features extracted for assay '%s'.", assay.assay_id
            )
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all cached DataFrames to free memory."""
        n = len(self._cache)
        self._cache.clear()
        logger.info("DataIntegrator cache cleared (%d entries removed).", n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_run(self, assay: AssayModel, run_id: str | None) -> RunRecord:
        """Return the RunRecord for run_id, or auto-select for single-run assays."""
        if run_id is None:
            if len(assay.runs) == 1:
                return assay.runs[0]
            if len(assay.runs) == 0:
                raise DataFileError(
                    f"Assay '{assay.assay_id}' has no runs extracted from the ISA-JSON."
                )
            raise AmbiguousRunError(
                f"Assay '{assay.assay_id}' has {len(assay.runs)} runs. "
                f"Specify run_id= explicitly. "
                f"Available: {[r.run_id for r in assay.runs]}"
            )

        run = assay.get_run(run_id)
        if run is None:
            available = [r.run_id for r in assay.runs]
            raise RunNotFoundError(
                f"run_id '{run_id}' not found in assay '{assay.assay_id}'. "
                f"Available run IDs: {available}"
            )
        return run

    def _load_run(
        self,
        assay: AssayModel,
        study_id: str,
        run: RunRecord,
        file_type: str,
    ) -> pd.DataFrame:
        """Load one run from disk and return a (time, value) DataFrame."""
        data_file: DataFile | None = (
            run.processed_file if file_type == "processed" else run.raw_file
        )

        if data_file is None or not data_file.path:
            raise DataFileError(
                f"Assay '{assay.assay_id}', run '{run.run_id}': "
                f"no '{file_type}' data file is recorded in the ISA-JSON. "
                f"Use file_type='raw' or file_type='processed' to match what the ISA-JSON contains."
            )

        df = self._read_csv(Path(data_file.path))
        self._normalize_time(df, run)
        return df

    @staticmethod
    def _normalize_time(df: pd.DataFrame, run: "RunRecord") -> None:
        """
        Re-scale the ``time`` column to seconds, in-place.

        If the ISA-JSON protocol parameters declare a sampling frequency (Hz or
        kHz), use it to convert the raw numeric timestamp column::

            dt_seconds = 1 / fs
            scale = dt_seconds / median_raw_dt
            time_seconds = (time_raw - time_raw[0]) * scale

        When no fs is declared the column is only zeroed (start = 0) and left
        in whatever units the CSV uses.
        """
        t = df["time"]
        # Infer fs from run measurement params (same logic as AssayProxy._infer_fs)
        fs: float | None = None
        for pv in run.measurement_params:
            if pv.unit and ("hz" in pv.unit.lower()):
                try:
                    val = float(str(pv.value).replace(",", "."))
                    fs = val * 1000.0 if "khz" in pv.unit.lower() else val
                    break
                except (ValueError, TypeError):
                    continue

        t0 = float(t.iloc[0])
        if fs and fs > 0:
            dt_raw = float(t.diff().dropna().abs().median())
            if dt_raw > 0:
                scale = (1.0 / fs) / dt_raw
                df["time"] = (t - t0) * scale
                return
        # No fs available — just zero-base the column
        df["time"] = t - t0

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """
        Read a 2-column measurement CSV/TSV into a DataFrame.

        Returns a DataFrame with columns named "time" and "value",
        where "time" is always float64 (seconds) and "value" is float64.

        Fast path: tries the C engine with explicit separators (no sniffing).
        Sniffing via engine="python" is only used as a last resort — it is
        ~10–20× slower per file and holds the GIL, which kills thread parallelism.

        Raises
        ------
        DataFileError
            File not found, encoding failure, shape mismatch, or empty file.
        """
        if not path.exists():
            raise DataFileError(
                f"Data file not found: '{path}'. "
                f"Set data_root to the directory containing your measurement files."
            )

        logger.debug("Reading CSV: '%s'.", path)

        df: pd.DataFrame | None = None

        # --- Fast path: C engine, no delimiter sniffing, releases the GIL ---
        for sep in (",", "\t", ";"):
            for encoding in ("utf-8", "latin-1"):
                try:
                    candidate = pd.read_csv(
                        path,
                        sep=sep,
                        engine="c",
                        encoding=encoding,
                        header=None,
                        dtype=str,
                        on_bad_lines="warn",
                    )
                    if candidate.shape[1] >= 2:
                        df = candidate
                        break
                except UnicodeDecodeError:
                    continue
                except Exception:
                    break
            if df is not None:
                break

        # --- Slow fallback: Python engine with auto-detection (handles exotic formats) ---
        if df is None:
            for encoding in ("utf-8", "latin-1"):
                try:
                    df = pd.read_csv(
                        path,
                        sep=None,
                        engine="python",
                        encoding=encoding,
                        header=None,
                        dtype=str,
                        on_bad_lines="warn",
                    )
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as exc:
                    raise DataFileError(
                        f"Failed to read '{path}' as CSV: {exc}"
                    ) from exc
            else:
                raise DataFileError(
                    f"Cannot decode '{path}'. Tried UTF-8 and Latin-1."
                )

        if df is None or df.empty:
            raise DataFileError(f"Data file is empty: '{path}'.")

        # Drop header row if first row is non-numeric (e.g. 'timestamp,value').
        if not _is_numeric_row(df.iloc[0]):
            df = df.iloc[1:].reset_index(drop=True)

        if df.empty:
            raise DataFileError(f"Data file is empty after removing header: '{path}'.")

        if df.shape[1] < 2:
            raise DataFileError(
                f"Data file '{path}' has only {df.shape[1]} column(s). "
                f"ISA-PHM one-column rule requires exactly 2 columns: time + value."
            )

        if df.shape[1] > 2:
            logger.warning(
                "'%s' has %d columns; using first two as (time, value).",
                path, df.shape[1],
            )

        df = df.iloc[:, :2].copy()
        df.columns = ["time", "value"]

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["time"] = _to_float_seconds(df["time"])

        return df

    @staticmethod
    def _get_value_column(df: pd.DataFrame) -> pd.Series:
        """Return the 'value' column, or the second column if 'value' is absent."""
        if "value" in df.columns:
            return df["value"]
        return df.iloc[:, 1]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _is_numeric_row(row: pd.Series) -> bool:
    """Return True if all values in the row look numeric."""
    try:
        pd.to_numeric(row, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _to_float_seconds(series: pd.Series) -> pd.Series:
    """
    Convert a time column to float64.

    Handles:
    - Already-numeric values (returned as-is; unit normalised later by _normalize_time)
    - Timedelta strings: HH:MM:SS.ffffff  →  total_seconds()
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    # Fall back to timedelta parsing (covers HH:MM:SS.ffffff).
    try:
        return pd.to_timedelta(series).dt.total_seconds()
    except Exception:
        return numeric
