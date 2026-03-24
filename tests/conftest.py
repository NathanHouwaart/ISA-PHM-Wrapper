"""
conftest.py — shared fixtures for the ISA-PHM wrapper test suite.

Real ISA-JSON fixtures are taken from the existing notebooks/ directory
(two-levels up from the python-wrapper/tests/ directory).

CSV data fixtures are created in a temporary directory under each test.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Paths to real ISA-JSON fixtures
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
_NOTEBOOKS = _HERE.parent.parent / "notebooks"

SINGLE_RUN_PATH = _NOTEBOOKS / "single-run-isa.json"
MULTI_RUN_PATH = _NOTEBOOKS / "multi-run-isa.json"


@pytest.fixture(scope="session")
def single_run_isa_path() -> Path:
    if not SINGLE_RUN_PATH.exists():
        pytest.skip(f"single-run-isa.json not found at '{SINGLE_RUN_PATH}'.")
    return SINGLE_RUN_PATH


@pytest.fixture(scope="session")
def multi_run_isa_path() -> Path:
    if not MULTI_RUN_PATH.exists():
        pytest.skip(f"multi-run-isa.json not found at '{MULTI_RUN_PATH}'.")
    return MULTI_RUN_PATH


# ---------------------------------------------------------------------------
# Minimal synthetic ISA-JSON fixture (no external files needed)
# ---------------------------------------------------------------------------

def _make_minimal_isa(
    *,
    n_runs: int = 1,
    experiment_type: str = "diagnostic-single",
    data_file_paths: list[str] | None = None,
) -> dict:
    """
    Build a minimal but structurally valid ISA-PHM JSON dict.

    Generates ``n_runs`` pairs of (measurement, processing) processes, each
    outputting one processed data file.
    """
    if data_file_paths is None:
        data_file_paths = [f"run_{i+1:02d}.csv" for i in range(n_runs)]

    # Factor definitions
    factor_id = "#factor/f1"
    factor_ann_id = "#ontology_annotation/ft1"

    # Sample
    sample_id = "#sample/s1"

    # Protocol IDs
    meas_proto_id = "#protocol/meas1"
    proc_proto_id = "#protocol/proc1"
    sensor_uuid = "aaaabbbb-0000-0000-0000-000000000001"

    # Build inter-linked processSequence
    processes = []
    data_files = []
    samples_list = []  # Assay-level sample refs
    raw_df_id_base = "#data_file/raw"
    proc_df_id_base = "#data_file/proc"

    prev_proc_id = None
    for i in range(n_runs):
        meas_pid = f"#process/meas{i}"
        proc_pid = f"#process/proc{i}"
        raw_dfid = f"{raw_df_id_base}{i}"
        proc_dfid = f"{proc_df_id_base}{i}"

        raw_df = {
            "@id": raw_dfid,
            "comments": [],
            "name": "",
            "type": "Raw Data File",
        }
        proc_df = {
            "@id": proc_dfid,
            "comments": [],
            "name": data_file_paths[i],
            "type": "Processed Data File",
        }
        data_files.extend([raw_df, proc_df])
        samples_list.append({"@id": sample_id})

        meas_proc: dict = {
            "@id": meas_pid,
            "comments": [],
            "executesProtocol": {"@id": meas_proto_id},
            "inputs": [{"@id": sample_id}],
            "outputs": [{"@id": raw_dfid}],
            "parameterValues": [],
        }
        proc_proc: dict = {
            "@id": proc_pid,
            "comments": [],
            "executesProtocol": {"@id": proc_proto_id},
            "inputs": [{"@id": raw_dfid}],
            "outputs": [{"@id": proc_dfid}],
            "parameterValues": [],
            "previousProcess": {"@id": meas_pid},
        }
        meas_proc["nextProcess"] = {"@id": proc_pid}
        if prev_proc_id is not None:
            meas_proc["previousProcess"] = {"@id": prev_proc_id}
        if i < n_runs - 1:
            proc_proc["nextProcess"] = {"@id": f"#process/meas{i+1}"}

        processes.extend([meas_proc, proc_proc])
        prev_proc_id = proc_pid

    study = {
        "@id": "#study/st1",
        "identifier": "st1-uuid",
        "title": "Test Study",
        "description": "Minimal test study",
        "comments": [],
        "factors": [
            {
                "@id": factor_id,
                "factorName": "Speed",
                "factorType": {
                    "@id": factor_ann_id,
                    "annotationValue": "Operating condition",
                    "comments": [],
                    "termAccession": "",
                    "termSource": "",
                },
                "comments": [],
            }
        ],
        "protocols": [
            {
                "@id": meas_proto_id,
                "comments": [{"name": "Sensor id", "value": sensor_uuid}],
                "name": f"Vibration measurement ({sensor_uuid})",
                "parameters": [],
                "protocolType": {
                    "@id": "#ontology_annotation/pt_meas",
                    "annotationValue": "Measurement Protocol",
                    "comments": [],
                    "termAccession": "",
                    "termSource": "",
                },
            },
            {
                "@id": proc_proto_id,
                "comments": [],
                "name": "Signal processing",
                "parameters": [],
                "protocolType": {
                    "@id": "#ontology_annotation/pt_proc",
                    "annotationValue": "Processing Protocol",
                    "comments": [],
                    "termAccession": "",
                    "termSource": "",
                },
            },
        ],
        "materials": {
            "samples": [
                {
                    "@id": sample_id,
                    "comments": [],
                    "factorValues": [
                        {
                            "category": {"@id": factor_id},
                            "comments": [],
                            "value": 1500,
                        }
                    ],
                    "name": "Sample 01",
                }
            ],
            "sources": [],
        },
        "assays": [
            {
                "@id": "#assay/a1",
                "comments": [],
                "filename": "a_st01_se01",
                "measurementType": {
                    "@id": "#ontology_annotation/mt1",
                    "annotationValue": "Vibration",
                    "comments": [],
                    "termAccession": "",
                    "termSource": "",
                },
                "technologyType": {
                    "@id": "#ontology_annotation/tt1",
                    "annotationValue": "Accelerometer",
                    "comments": [],
                    "termAccession": "",
                    "termSource": "",
                },
                "technologyPlatform": "Generic",
                "unitCategories": [],
                "dataFiles": data_files,
                "materials": {
                    "samples": samples_list,
                    "otherMaterials": [],
                },
                "processSequence": processes,
            }
        ],
    }

    return {
        "comments": [
            {"name": "ud_identifier", "value": "i_test"},
            {"name": "experiment_type", "value": experiment_type},
            {"name": "license", "value": "MIT"},
        ],
        "description": "Test investigation",
        "identifier": "test-inv-001",
        "ontologySourceReferences": [],
        "people": [],
        "studies": [study],
        "title": "Test Investigation",
    }


@pytest.fixture
def minimal_single_run_isa() -> dict:
    """Minimal 1-run ISA-JSON dict (in-memory, no files needed)."""
    return _make_minimal_isa(n_runs=1, experiment_type="diagnostic-single")


@pytest.fixture
def minimal_multi_run_isa(tmp_path) -> dict:
    """Minimal 3-run ISA-JSON dict with paths pointing to tmp CSV files."""
    paths = [str(tmp_path / f"run_{i+1:02d}.csv") for i in range(3)]
    return _make_minimal_isa(
        n_runs=3, experiment_type="diagnostic-multi", data_file_paths=paths
    )


# ---------------------------------------------------------------------------
# CSV data helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_csv(tmp_path) -> Path:
    """Create a small synthetic 2-column CSV (time, value) in tmp_path."""
    rng = np.random.default_rng(42)
    t = np.linspace(0, 1, 1000)
    v = rng.standard_normal(1000)
    p = tmp_path / "signal.csv"
    with p.open("w", newline="") as f:
        writer = csv.writer(f)
        for ti, vi in zip(t, v):
            writer.writerow([ti, vi])
    return p


@pytest.fixture
def tmp_csv_dir(tmp_path) -> Path:
    """Create 3 CSV files (run_01.csv … run_03.csv) in tmp_path."""
    rng = np.random.default_rng(0)
    for i in range(3):
        p = tmp_path / f"run_{i+1:02d}.csv"
        t = np.linspace(0, 1, 500)
        v = rng.standard_normal(500)
        with p.open("w", newline="") as f:
            writer = csv.writer(f)
            for ti, vi in zip(t, v):
                writer.writerow([ti, vi])
    return tmp_path


@pytest.fixture
def minimal_single_run_isa_file(tmp_path, minimal_single_run_isa, tmp_csv) -> Path:
    """
    Write the minimal single-run ISA dict to a JSON file and point the
    processed data file path at tmp_csv.
    """
    isa = minimal_single_run_isa
    # Update the processed file name/path to point at the real CSV.
    for assay in isa["studies"][0]["assays"]:
        for df in assay["dataFiles"]:
            if df["type"] == "Processed Data File":
                df["name"] = str(tmp_csv)
    p = tmp_path / "i_test.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


@pytest.fixture
def minimal_multi_run_isa_file(tmp_path, tmp_csv_dir) -> Path:
    """
    Write a 3-run ISA dict to a JSON file; processed file paths point at the
    CSV files created in tmp_csv_dir.
    """
    paths = [str(tmp_csv_dir / f"run_{i+1:02d}.csv") for i in range(3)]
    isa = _make_minimal_isa(
        n_runs=3, experiment_type="diagnostic-multi", data_file_paths=paths
    )
    p = tmp_path / "i_multi_test.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


def _make_fault_isa(data_file_paths: list[str]) -> dict:
    """Build a 3-run ISA dict that includes a *Quantitative fault specification* factor."""
    isa = _make_minimal_isa(
        n_runs=3, experiment_type="diagnostic-fault", data_file_paths=data_file_paths
    )
    fault_factor_id = "#factor/fault1"
    fault_ann_id = "#ontology_annotation/ft_fault"
    study = isa["studies"][0]
    study["factors"].append({
        "@id": fault_factor_id,
        "factorName": "Fault size",
        "factorType": {
            "@id": fault_ann_id,
            "annotationValue": "Quantitative fault specification",
            "comments": [],
            "termAccession": "",
            "termSource": "",
        },
        "comments": [],
    })
    for sample in study["materials"]["samples"]:
        sample["factorValues"].append({
            "category": {"@id": fault_factor_id},
            "comments": [],
            "value": 0.5,
        })
    return isa


@pytest.fixture
def minimal_fault_isa_file(tmp_path, tmp_csv_dir) -> Path:
    """3-run ISA JSON with a 'Fault size' quantitative fault specification factor."""
    paths = [str(tmp_csv_dir / f"run_{i+1:02d}.csv") for i in range(3)]
    isa = _make_fault_isa(data_file_paths=paths)
    p = tmp_path / "i_fault_test.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


def _make_params_isa(data_file_paths: list[str]) -> dict:
    """
    Build a single-run ISA dict that includes one ``parameterValue`` in both
    the measurement process (Sampling rate = 25600) and the processing process
    (Window size = 1024).
    """
    isa = _make_minimal_isa(
        n_runs=1,
        experiment_type="diagnostic-params",
        data_file_paths=data_file_paths,
    )
    study = isa["studies"][0]

    # Protocol parameter @ids
    meas_param_id = "#proto_param/fs"
    proc_param_id = "#proto_param/window"

    # Add parameter definitions to protocols
    meas_proto = next(p for p in study["protocols"] if "measurement" in p["name"].lower())
    proc_proto = next(p for p in study["protocols"] if "processing" in p["name"].lower())

    meas_proto["parameters"].append({
        "@id": meas_param_id,
        "parameterName": {
            "annotationValue": "Sampling rate",
            "termAccession": "",
            "termSource": "",
            "comments": [],
        },
    })
    proc_proto["parameters"].append({
        "@id": proc_param_id,
        "parameterName": {
            "annotationValue": "Window size",
            "termAccession": "",
            "termSource": "",
            "comments": [],
        },
    })

    # Add parameterValues to the matching processes in the assay
    assay = study["assays"][0]
    for proc in assay["processSequence"]:
        proto_ref_id = proc.get("executesProtocol", {}).get("@id", "")
        if proto_ref_id == meas_proto["@id"]:
            proc["parameterValues"].append({
                "category": {"@id": meas_param_id},
                "value": 25600,
            })
        elif proto_ref_id == proc_proto["@id"]:
            proc["parameterValues"].append({
                "category": {"@id": proc_param_id},
                "value": 1024,
            })

    return isa


@pytest.fixture
def minimal_params_isa_file(tmp_path, tmp_csv) -> Path:
    """Single-run ISA JSON with populated measurement and processing parameterValues."""
    isa = _make_params_isa(data_file_paths=[str(tmp_csv)])
    # Point processed file at the real CSV
    for assay in isa["studies"][0]["assays"]:
        for df in assay["dataFiles"]:
            if df["type"] == "Processed Data File":
                df["name"] = str(tmp_csv)
    p = tmp_path / "i_params_test.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


def _make_publication_isa(data_file_paths: list[str]) -> dict:
    """Single-run ISA dict with investigation-level people + publications."""
    isa = _make_minimal_isa(
        n_runs=1,
        experiment_type="diagnostic-single",
        data_file_paths=data_file_paths,
    )
    isa["people"] = [
        {
            "firstName": "Alice",
            "lastName": "Example",
            "email": "alice@example.com",
            "affiliation": "Test Lab",
            "roles": [{"annotationValue": "Investigation"}],
            "comments": [{"name": "orcid", "value": "0000-0001-0000-0001"}],
        },
        {
            "firstName": "Bob",
            "lastName": "Example",
            "email": "bob@example.com",
            "affiliation": "Test Lab",
            "roles": [{"annotationValue": "Data curation"}],
            "comments": [],
        },
    ]
    isa["publications"] = [
        {
            "title": "An Example ISA-PHM Publication",
            "doi": "10.1000/example.doi",
            "pubMedID": "",
            "status": {
                "annotationValue": "Published",
                "comments": [],
                "termAccession": "",
                "termSource": "",
            },
            "authorList": "#alice-contact; #bob-contact",
            "comments": [
                {"name": "Corresponding author ID", "value": "alice-contact"}
            ],
        }
    ]
    return isa


@pytest.fixture
def minimal_publication_isa_file(tmp_path, tmp_csv) -> Path:
    """Single-run ISA JSON with top-level contacts and one publication."""
    isa = _make_publication_isa(data_file_paths=[str(tmp_csv)])
    for assay in isa["studies"][0]["assays"]:
        for df in assay["dataFiles"]:
            if df["type"] == "Processed Data File":
                df["name"] = str(tmp_csv)
    p = tmp_path / "i_publication_test.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


def _make_semantic_isa(data_file_paths: list[str]) -> dict:
    """Single-run ISA dict with diverse factor/parameter names for semantic tests."""
    isa = _make_params_isa(data_file_paths=data_file_paths)
    study = isa["studies"][0]

    # Replace factors with naming variants and an unknown.
    study["factors"] = [
        {
            "@id": "#factor/f_speed",
            "factorName": "Motor speed",
            "factorType": {
                "@id": "#oa/ft_speed",
                "annotationValue": "Operating condition",
                "comments": [],
                "termAccession": "",
                "termSource": "",
            },
            "comments": [{"name": "unit", "value": "RPM"}],
        },
        {
            "@id": "#factor/f_fault",
            "factorName": "Fault Severity",
            "factorType": {
                "@id": "#oa/ft_fault",
                "annotationValue": "Quantitative fault specification",
                "comments": [],
                "termAccession": "",
                "termSource": "",
            },
            "comments": [],
        },
        {
            "@id": "#factor/f_press",
            "factorName": "Discharge Perssure",
            "factorType": {
                "@id": "#oa/ft_press",
                "annotationValue": "Operating condition",
                "comments": [],
                "termAccession": "",
                "termSource": "",
            },
            "comments": [],
        },
        {
            "@id": "#factor/f_unknown",
            "factorName": "Mystery Knob",
            "factorType": {
                "@id": "#oa/ft_unknown",
                "annotationValue": "Other",
                "comments": [],
                "termAccession": "",
                "termSource": "",
            },
            "comments": [],
        },
    ]

    sample = study["materials"]["samples"][0]
    sample["factorValues"] = [
        {"category": {"@id": "#factor/f_speed"}, "comments": [], "value": 1500},
        {"category": {"@id": "#factor/f_fault"}, "comments": [], "value": 2},
        {"category": {"@id": "#factor/f_press"}, "comments": [], "value": 120},
        {"category": {"@id": "#factor/f_unknown"}, "comments": [], "value": "X"},
    ]

    # Rename protocol parameters for semantic mapping checks.
    meas_proto = next(p for p in study["protocols"] if "measurement" in p["name"].lower())
    proc_proto = next(p for p in study["protocols"] if "processing" in p["name"].lower())
    meas_proto["parameters"][0]["parameterName"]["annotationValue"] = "Sampling Frequency"
    proc_proto["parameters"][0]["parameterName"]["annotationValue"] = "Completely Custom Param"

    return isa


@pytest.fixture
def minimal_semantic_isa_file(tmp_path, tmp_csv) -> Path:
    isa = _make_semantic_isa(data_file_paths=[str(tmp_csv)])
    for assay in isa["studies"][0]["assays"]:
        for df in assay["dataFiles"]:
            if df["type"] == "Processed Data File":
                df["name"] = str(tmp_csv)
    p = tmp_path / "i_semantic_test.json"
    p.write_text(json.dumps(isa), encoding="utf-8")
    return p


@pytest.fixture
def semantic_override_config(tmp_path) -> Path:
    cfg = {
        "factor": {
            "mystery knob": "custom_mystery_knob"
        },
        "parameter": {
            "completely custom param": "custom_processing_parameter"
        },
    }
    p = tmp_path / "semantic_overrides.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p
