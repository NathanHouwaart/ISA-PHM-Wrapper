from __future__ import annotations

import argparse
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import pandas as pd

from isa_phm import ISAWrapper
from isa_phm.errors import DataFileError


def _iter_target_assays(
    wrapper: ISAWrapper,
    study_limit: int | None = None,
    assay_limit: int | None = None,
) -> list[tuple[str, str]]:
    targets: list[tuple[str, str]] = []
    studies = wrapper.list_studies()
    if study_limit is not None:
        studies = studies[:study_limit]

    for summary in studies:
        study = wrapper.study(summary.study_id)
        assays = study.list_assays()
        if assay_limit is not None:
            assays = assays[:assay_limit]
        for assay in assays:
            targets.append((summary.study_id, assay.assay_id))
    return targets


def _run_once(
    *,
    isa_json: Path,
    data_root: Path,
    file_type: str,
    n_workers: int | None,
    strict_validation: bool,
    enable_chunked_large_file_mode: bool,
    large_file_threshold_mb: float,
    chunk_rows: int,
    study_limit: int | None,
    assay_limit: int | None,
) -> dict[str, Any]:
    wrapper = ISAWrapper(
        isa_json,
        data_root=data_root,
        strict_validation=strict_validation,
        enable_chunked_large_file_mode=enable_chunked_large_file_mode,
        large_file_threshold_mb=large_file_threshold_mb,
        chunk_rows=chunk_rows,
    )
    targets = _iter_target_assays(
        wrapper,
        study_limit=study_limit,
        assay_limit=assay_limit,
    )

    total_rows = 0
    total_rms = 0.0
    processed_assays = 0
    failures: list[dict[str, str]] = []

    tracemalloc.start()
    t0 = time.perf_counter()
    for study_id, assay_id in targets:
        try:
            assay = wrapper.study(study_id).assay(assay_id)
            features = assay.lifecycle_features(
                file_type=file_type, n_workers=n_workers
            )
            if not isinstance(features, pd.DataFrame):
                continue
            if not features.empty:
                total_rows += int(features.shape[0])
                if "rms" in features.columns:
                    total_rms += float(pd.to_numeric(features["rms"], errors="coerce").fillna(0).sum())
                processed_assays += 1
        except (DataFileError, ValueError) as exc:
            failures.append(
                {
                    "study_id": study_id,
                    "assay_id": assay_id,
                    "error": str(exc),
                }
            )
    elapsed_sec = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    return {
        "enable_chunked_large_file_mode": enable_chunked_large_file_mode,
        "large_file_threshold_mb": large_file_threshold_mb,
        "chunk_rows": chunk_rows,
        "file_type": file_type,
        "n_workers": n_workers,
        "n_target_assays": len(targets),
        "n_processed_assays": processed_assays,
        "total_rows": total_rows,
        "total_rms_sum": round(total_rms, 8),
        "n_failures": len(failures),
        "failures_sample": failures[:5],
        "elapsed_sec": round(elapsed_sec, 4),
        "tracemalloc_peak_mb": round(peak / (1024 * 1024), 2),
    }


def run_benchmark(
    *,
    isa_json: Path,
    data_root: Path,
    file_type: str,
    n_workers: int | None,
    strict_validation: bool,
    repeats: int,
    large_file_threshold_mb: float,
    chunk_rows: int,
    study_limit: int | None,
    assay_limit: int | None,
) -> dict[str, Any]:
    baseline_runs = []
    chunked_runs = []
    for _ in range(repeats):
        baseline_runs.append(
            _run_once(
                isa_json=isa_json,
                data_root=data_root,
                file_type=file_type,
                n_workers=n_workers,
                strict_validation=strict_validation,
                enable_chunked_large_file_mode=False,
                large_file_threshold_mb=large_file_threshold_mb,
                chunk_rows=chunk_rows,
                study_limit=study_limit,
                assay_limit=assay_limit,
            )
        )
        chunked_runs.append(
            _run_once(
                isa_json=isa_json,
                data_root=data_root,
                file_type=file_type,
                n_workers=n_workers,
                strict_validation=strict_validation,
                enable_chunked_large_file_mode=True,
                large_file_threshold_mb=large_file_threshold_mb,
                chunk_rows=chunk_rows,
                study_limit=study_limit,
                assay_limit=assay_limit,
            )
        )

    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        vals = [float(r[key]) for r in rows]
        return sum(vals) / max(len(vals), 1)

    baseline_elapsed = _mean(baseline_runs, "elapsed_sec")
    chunked_elapsed = _mean(chunked_runs, "elapsed_sec")
    baseline_peak = _mean(baseline_runs, "tracemalloc_peak_mb")
    chunked_peak = _mean(chunked_runs, "tracemalloc_peak_mb")

    baseline_rows = int(baseline_runs[-1]["total_rows"]) if baseline_runs else 0
    chunked_rows = int(chunked_runs[-1]["total_rows"]) if chunked_runs else 0
    baseline_rms = float(baseline_runs[-1]["total_rms_sum"]) if baseline_runs else 0.0
    chunked_rms = float(chunked_runs[-1]["total_rms_sum"]) if chunked_runs else 0.0

    speedup = (baseline_elapsed / chunked_elapsed) if chunked_elapsed > 0 else None
    memory_ratio = (chunked_peak / baseline_peak) if baseline_peak > 0 else None

    return {
        "inputs": {
            "isa_json": str(isa_json),
            "data_root": str(data_root),
            "file_type": file_type,
            "n_workers": n_workers,
            "strict_validation": strict_validation,
            "repeats": repeats,
            "large_file_threshold_mb": large_file_threshold_mb,
            "chunk_rows": chunk_rows,
            "study_limit": study_limit,
            "assay_limit": assay_limit,
        },
        "baseline_runs": baseline_runs,
        "chunked_runs": chunked_runs,
        "comparison": {
            "avg_elapsed_sec_baseline": round(baseline_elapsed, 4),
            "avg_elapsed_sec_chunked": round(chunked_elapsed, 4),
            "speedup_baseline_over_chunked": round(speedup, 4) if speedup else None,
            "avg_peak_mb_baseline": round(baseline_peak, 2),
            "avg_peak_mb_chunked": round(chunked_peak, 2),
            "peak_memory_ratio_chunked_over_baseline": round(memory_ratio, 4)
            if memory_ratio
            else None,
            "total_rows_match": baseline_rows == chunked_rows,
            "total_rms_sum_match": abs(baseline_rms - chunked_rms) < 1e-8,
            "baseline_total_rows": baseline_rows,
            "chunked_total_rows": chunked_rows,
            "baseline_total_rms_sum": baseline_rms,
            "chunked_total_rms_sum": chunked_rms,
        },
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark ISA-PHM lifecycle feature extraction with chunked mode on/off."
    )
    p.add_argument("--isa-json", required=True, type=Path, help="Path to ISA JSON file.")
    p.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Root directory for data files. Defaults to ISA JSON parent.",
    )
    p.add_argument(
        "--file-type",
        choices=["raw", "processed", "auto"],
        default="auto",
        help="Data file preference used during lifecycle loading.",
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Thread worker count for lifecycle_features (default: auto).",
    )
    p.add_argument(
        "--strict-validation",
        action="store_true",
        help="Enable strict ISA schema validation in parser.",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="How many times to run each mode.",
    )
    p.add_argument(
        "--large-file-threshold-mb",
        type=float,
        default=64.0,
        help="Minimum file size in MB for chunked mode.",
    )
    p.add_argument(
        "--chunk-rows",
        type=int,
        default=250_000,
        help="Rows per chunk in chunked mode.",
    )
    p.add_argument(
        "--study-limit",
        type=int,
        default=None,
        help="Optional limit on number of studies to benchmark.",
    )
    p.add_argument(
        "--assay-limit",
        type=int,
        default=None,
        help="Optional limit on assays per study to benchmark.",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write benchmark result JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    isa_json: Path = args.isa_json
    data_root: Path = args.data_root if args.data_root is not None else isa_json.parent

    result = run_benchmark(
        isa_json=isa_json,
        data_root=data_root,
        file_type=args.file_type,
        n_workers=args.n_workers,
        strict_validation=bool(args.strict_validation),
        repeats=max(1, int(args.repeats)),
        large_file_threshold_mb=float(args.large_file_threshold_mb),
        chunk_rows=max(1, int(args.chunk_rows)),
        study_limit=args.study_limit,
        assay_limit=args.assay_limit,
    )

    print(json.dumps(result, indent=2))
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nWrote benchmark report to: {args.output_json}")


if __name__ == "__main__":
    main()
