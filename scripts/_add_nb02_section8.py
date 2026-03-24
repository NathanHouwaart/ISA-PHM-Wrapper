"""One-off script: add section 8 (fault labels) to notebook 02."""
import json
import uuid
from pathlib import Path

NB_PATH = Path(__file__).parent.parent / "notebooks" / "02_diagnostic_workflow.ipynb"

nb = json.loads(NB_PATH.read_text(encoding="utf-8"))

md_cell = {
    "id": uuid.uuid4().hex[:8],
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8. Fault labels \u2014 bridge to machine learning\n",
        "\n",
        "`study.get_fault_labels()` extracts every **fault-related factor** (factor type\n",
        "matching `fault`, `damage`, or `rul`) into a tidy DataFrame keyed by `assay_id`\n",
        "and `run_id`.\n",
        "\n",
        "Merge it directly with `assay.lifecycle_features()` to get a labelled feature matrix:\n",
        "\n",
        "```python\n",
        "features = assay.lifecycle_features()   # signal statistics per run\n",
        'labels   = study.get_fault_labels()     # fault severity per run\n',
        'df       = features.merge(labels, on=["assay_id", "run_id"])\n',
        "```\n",
        "\n",
        "- **Diagnostic dataset**: fixed fault level \u2014 every run gets the same label.\n",
        "- **Prognostic dataset**: label column holds the evolving damage level / RUL.\n",
        "\n",
        "> **No fault factors defined?** `get_fault_labels()` returns an empty DataFrame\n",
        "> with the correct columns \u2014 safe to call on any dataset.\n",
    ],
}

code_cell = {
    "id": uuid.uuid4().hex[:8],
    "cell_type": "code",
    "metadata": {},
    "execution_count": None,
    "outputs": [],
    "source": [
        "# Fault-related factor values for every (assay, run) pair in this study\n",
        "labels = study.get_fault_labels()\n",
        "display(labels)\n",
        "\n",
        "# Optionally filter to a single sensor channel:\n",
        "# labels_se01 = study.get_fault_labels(assay_id='a_st01_se01')\n",
        "\n",
        "# Merge with lifecycle features for a ready-to-train DataFrame:\n",
        "# features = assay.lifecycle_features()\n",
        "# training_df = features.merge(labels, on=['assay_id', 'run_id'])\n",
        "# display(training_df.head())",
    ],
}

insert_after_id = "0caa6f68"
idx = next(
    (i for i, c in enumerate(nb["cells"]) if c.get("id") == insert_after_id), None
)
if idx is None:
    raise RuntimeError(f"Cell {insert_after_id!r} not found")

print(f"Inserting after cell index {idx} (id={insert_after_id})")
nb["cells"].insert(idx + 1, code_cell)
nb["cells"].insert(idx + 1, md_cell)
print(f"Total cells after edit: {len(nb['cells'])}")

NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("Saved.")
