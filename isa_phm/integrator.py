"""
DataIntegrator: load ISA-PHM measurement files into pandas DataFrames.

Two access modes:
1) Cached single-run loading for interactive usage.
2) Streaming lifecycle feature extraction for many runs.
"""

from __future__ import annotations

import csv
import logging
import os
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Generator, Literal, cast

import numpy as np
import pandas as pd

from .errors import AmbiguousRunError, DataFileError, RunNotFoundError
from .schemas import AssayModel, DataFile, DataLoadMetadata, RunRecord
from .utils import compute_features

logger = logging.getLogger("isa_phm")

STANDARD_COLUMNS = (
    "time",
    "value",
)

REQUESTED_FILE_TYPES: tuple[str, ...] = ("raw", "processed", "auto")
RESOLVED_FILE_TYPES: tuple[str, ...] = ("raw", "processed")


class _FIFOCache:
    """Simple FIFO eviction cache backed by OrderedDict."""

    def __init__(self, maxsize: int = 100) -> None:
        self._maxsize = maxsize
        self._store: OrderedDict = OrderedDict()

    def get(self, key: tuple) -> pd.DataFrame | None:
        return self._store.get(key)

    def put(self, key: tuple, df: pd.DataFrame) -> None:
        if key in self._store:
            return
        if len(self._store) >= self._maxsize:
            self._store.popitem(last=False)
        self._store[key] = df

    def clear(self) -> None:
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


class DataIntegrator:
    """Load and cache ISA-PHM data files as normalized DataFrames."""

    def __init__(self, data_root: Path, cache_maxsize: int = 100) -> None:
        self._data_root = Path(data_root)
        self._cache = _FIFOCache(maxsize=cache_maxsize)

    # ------------------------------------------------------------------
    # Public: single-run loading
    # ------------------------------------------------------------------

    def load(
        self,
        assay: AssayModel,
        study_id: str,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> pd.DataFrame:
        """
        Load a run into a DataFrame with columns [time, value].

        ``file_type='auto'`` prefers processed and falls back to raw.
        """
        df, _ = self.load_with_meta(
            assay=assay,
            study_id=study_id,
            run_id=run_id,
            file_type=file_type,
        )
        return df

    def load_with_meta(
        self,
        assay: AssayModel,
        study_id: str,
        run_id: str | None = None,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> tuple[pd.DataFrame, DataLoadMetadata]:
        """Load a run and return DataFrame plus resolved file-load metadata."""
        del study_id  # Reserved for future metadata enrichment.

        requested_file_type = self._validate_file_type(file_type)
        run = self._resolve_run(assay, run_id)
        data_file, resolved_file_type = self._resolve_data_file(
            assay_id=assay.assay_id,
            run=run,
            requested_file_type=requested_file_type,
        )

        cache_key = (assay.assay_id, run.run_id, resolved_file_type)
        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.debug("Cache hit: %s", cache_key)
            return cached, DataLoadMetadata(
                assay_id=assay.assay_id,
                run_id=run.run_id,
                requested_file_type=requested_file_type,
                resolved_file_type=resolved_file_type,
                file_path=data_file.path,
                from_cache=True,
            )

        df = self._read_csv(Path(data_file.path))
        self._normalize_time(df, run)
        self._cache.put(cache_key, df)
        return df, DataLoadMetadata(
            assay_id=assay.assay_id,
            run_id=run.run_id,
            requested_file_type=requested_file_type,
            resolved_file_type=resolved_file_type,
            file_path=data_file.path,
            from_cache=False,
        )

    # ------------------------------------------------------------------
    # Public: streaming lifecycle features
    # ------------------------------------------------------------------

    def stream_lifecycle_features(
        self,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed", "auto"] = "processed",
    ) -> Generator[dict, None, None]:
        """Yield one lifecycle feature row per run without caching all runs."""
        requested_file_type = self._validate_file_type(file_type)

        for run in assay.runs:
            try:
                data_file, resolved_file_type = self._resolve_data_file(
                    assay_id=assay.assay_id,
                    run=run,
                    requested_file_type=requested_file_type,
                )
                cache_key = (assay.assay_id, run.run_id, resolved_file_type)
                cached = self._cache.get(cache_key)
                if cached is not None:
                    raw_values = self._get_value_column(cached)
                else:
                    df = self._read_csv(Path(data_file.path))
                    raw_values = self._get_value_column(df)
            except DataFileError as exc:
                logger.warning(
                    "Skipping run '%s' of assay '%s': %s",
                    run.run_id,
                    assay.assay_id,
                    exc,
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
        run: RunRecord,
        assay: AssayModel,
        study_id: str,
        file_type: Literal["raw", "processed", "auto"],
    ) -> dict | None:
        """Helper for threaded lifecycle feature extraction."""
        try:
            data_file, resolved_file_type = self._resolve_data_file(
                assay_id=assay.assay_id,
                run=run,
                requested_file_type=self._validate_file_type(file_type),
            )
            cache_key = (assay.assay_id, run.run_id, resolved_file_type)
            cached = self._cache.get(cache_key)
            if cached is not None:
                raw_values = self._get_value_column(cached)
            else:
                df = self._read_csv(Path(data_file.path))
                raw_values = self._get_value_column(df)
        except DataFileError as exc:
            logger.warning(
                "Skipping run '%s' of assay '%s': %s",
                run.run_id,
                assay.assay_id,
                exc,
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
        file_type: Literal["raw", "processed", "auto"] = "processed",
        n_workers: int | None = None,
    ) -> pd.DataFrame:
        """Return lifecycle features for all runs as a DataFrame."""
        requested_file_type = self._validate_file_type(file_type)
        effective_workers = (
            n_workers if n_workers is not None else min(32, (os.cpu_count() or 1) + 4)
        )

        if effective_workers == 1 or len(assay.runs) <= 1:
            rows = list(
                self.stream_lifecycle_features(
                    assay=assay,
                    study_id=study_id,
                    file_type=requested_file_type,
                )
            )
        else:
            rows_unordered: list[dict] = []
            with ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = {
                    pool.submit(
                        self._compute_run_features,
                        run,
                        assay,
                        study_id,
                        requested_file_type,
                    ): run
                    for run in assay.runs
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result is not None:
                        rows_unordered.append(result)
            rows = sorted(rows_unordered, key=lambda r: r["run_number"])

        if not rows:
            logger.warning(
                "No lifecycle features extracted for assay '%s'.",
                assay.assay_id,
            )
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def clear_cache(self) -> None:
        """Evict all cached DataFrames."""
        n = len(self._cache)
        self._cache.clear()
        logger.info("DataIntegrator cache cleared (%d entries removed).", n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_run(self, assay: AssayModel, run_id: str | None) -> RunRecord:
        if run_id is None:
            if len(assay.runs) == 1:
                return assay.runs[0]
            if len(assay.runs) == 0:
                raise DataFileError(
                    f"Assay '{assay.assay_id}' has no runs extracted from the ISA-JSON."
                )
            raise AmbiguousRunError(
                f"Assay '{assay.assay_id}' has {len(assay.runs)} runs. "
                "Specify run_id explicitly. "
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

    @staticmethod
    def _validate_file_type(
        file_type: str,
    ) -> Literal["raw", "processed", "auto"]:
        normalized = str(file_type).strip().lower()
        if normalized not in REQUESTED_FILE_TYPES:
            raise DataFileError(
                f"Invalid file_type '{file_type}'. "
                f"Use one of: {REQUESTED_FILE_TYPES}."
            )
        return cast(Literal["raw", "processed", "auto"], normalized)

    @staticmethod
    def _resolve_data_file(
        assay_id: str,
        run: RunRecord,
        requested_file_type: Literal["raw", "processed", "auto"],
    ) -> tuple[DataFile, Literal["raw", "processed"]]:
        if requested_file_type == "auto":
            if run.processed_file is not None and run.processed_file.path:
                return run.processed_file, "processed"
            if run.raw_file is not None and run.raw_file.path:
                return run.raw_file, "raw"
            raise DataFileError(
                f"Assay '{assay_id}', run '{run.run_id}': no data file is recorded "
                "in the ISA-JSON (neither 'processed' nor 'raw')."
            )

        resolved_file_type = cast(
            Literal["raw", "processed"],
            requested_file_type,
        )
        data_file = (
            run.processed_file if resolved_file_type == "processed" else run.raw_file
        )
        if data_file is None or not data_file.path:
            raise DataFileError(
                f"Assay '{assay_id}', run '{run.run_id}': "
                f"no '{resolved_file_type}' data file is recorded in the ISA-JSON. "
                "Use file_type='auto' to prefer processed and fall back to raw."
            )
        return data_file, resolved_file_type

    @staticmethod
    def _normalize_time(df: pd.DataFrame, run: RunRecord) -> None:
        """Scale/shift time column to seconds in-place when sampling freq is known."""
        t = df["time"]

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

        df["time"] = t - t0

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read a 2-column measurement CSV/TSV into [time, value]."""
        if not path.exists():
            raise DataFileError(
                f"Data file not found: '{path}'. "
                "Set data_root to the directory containing your measurement files."
            )

        logger.debug("Reading CSV: '%s'.", path)
        df: pd.DataFrame | None = None

        # Fast path: C engine with explicit common separators.
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

        # Fallback 1: sniff delimiter once, then retry C engine.
        if df is None:
            sniff_sep: str | None = None
            for encoding in ("utf-8", "latin-1"):
                try:
                    sample = path.read_text(encoding=encoding)
                    sample = sample[:32768]
                    sniff_sep = csv.Sniffer().sniff(
                        sample,
                        delimiters=",\t;| ",
                    ).delimiter
                    break
                except (UnicodeDecodeError, csv.Error, OSError):
                    continue

            if sniff_sep:
                for encoding in ("utf-8", "latin-1"):
                    try:
                        candidate = pd.read_csv(
                            path,
                            sep=sniff_sep,
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

        # Fallback 2: Python engine with auto-detection.
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
                    raise DataFileError(f"Failed to read '{path}' as CSV: {exc}") from exc
            else:
                raise DataFileError(
                    f"Cannot decode '{path}'. Tried UTF-8 and Latin-1."
                )

        if df is None or df.empty:
            raise DataFileError(f"Data file is empty: '{path}'.")

        # Drop header row if first row is non-numeric (e.g. timestamp,value).
        if not _is_numeric_row(df.iloc[0]):
            df = df.iloc[1:].reset_index(drop=True)

        if df.empty:
            raise DataFileError(f"Data file is empty after removing header: '{path}'.")

        if df.shape[1] < 2:
            raise DataFileError(
                f"Data file '{path}' has only {df.shape[1]} column(s). "
                "ISA-PHM one-column rule requires exactly 2 columns: time + value."
            )

        if df.shape[1] > 2:
            logger.warning("'%s' has %d columns; using first two as (time, value).", path, df.shape[1])

        df = df.iloc[:, :2].copy()
        df.columns = ["time", "value"]
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["time"] = _to_float_seconds(df["time"])
        return df

    @staticmethod
    def _get_value_column(df: pd.DataFrame) -> pd.Series:
        if "value" in df.columns:
            return df["value"]
        return df.iloc[:, 1]


def _is_numeric_row(row: pd.Series) -> bool:
    """Return True if all values in row parse as numeric."""
    try:
        pd.to_numeric(row, errors="raise")
        return True
    except (ValueError, TypeError):
        return False


def _to_float_seconds(series: pd.Series) -> pd.Series:
    """Convert time column to float seconds where possible."""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric
    try:
        return pd.to_timedelta(series).dt.total_seconds()
    except Exception:
        return numeric
