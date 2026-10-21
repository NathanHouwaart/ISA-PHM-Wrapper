"""Check that a produced ISA-JSON file is structurally usable by isa-phm.

This is deliberately a file-based boundary check: it does not import the
Wizard or Backend.  Run it on an exported ``ISA-PHM-Out.json`` after either
producer changes its conversion logic.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Support direct execution from a source checkout without requiring install.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from isa_phm import ISAWrapper


def check(isa_json: Path, data_root: Path | None = None) -> dict[str, object]:
    """Parse and inspect an exported ISA-JSON file without loading signal data."""
    # Compatibility concerns Wrapper extraction and public traversal.  Strict
    # ISA schema validation is a separate producer concern and requires the
    # optional isatools validator to be installed in this environment.
    wrapper = ISAWrapper(
        isa_json,
        data_root=data_root or isa_json.parent,
        strict_validation=False,
    )
    study_count = 0
    assay_count = 0
    run_count = 0
    frequency_ready_runs = 0

    for study_summary in wrapper.list_studies():
        study_count += 1
        study = wrapper.study(study_summary.study_id)
        for assay_summary in study.list_assays():
            assay_count += 1
            assay = study.assay(assay_summary.assay_id)
            capabilities = assay.analysis_capabilities()
            run_count += len(capabilities.runs)
            frequency_ready_runs += sum(
                run.frequency_domain.available for run in capabilities.runs
            )

    return {
        "compatible": True,
        "source": str(isa_json.resolve()),
        "study_count": study_count,
        "assay_count": assay_count,
        "run_count": run_count,
        "frequency_ready_runs": frequency_ready_runs,
        "repairs_applied": len(wrapper.repair_log),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check an ISA-JSON export against the isa-phm public API."
    )
    parser.add_argument("isa_json", type=Path, help="Path to an exported ISA-JSON file.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Optional data root; defaults to the ISA-JSON file's directory.",
    )
    args = parser.parse_args()
    logging.getLogger("isa_phm").setLevel(logging.ERROR)
    try:
        print(json.dumps(check(args.isa_json, args.data_root), indent=2))
    except Exception as exc:  # command-line boundary: show an actionable failure
        print(f"ISA-PHM compatibility check failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
