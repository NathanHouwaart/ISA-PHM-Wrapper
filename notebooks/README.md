# Wrapper notebooks

Install the wrapper and notebook dependencies from the repository root:

```bash
python -m pip install -e ".[notebooks]"
jupyter lab
```

Use the Jupyter kernel for that same Python environment. The notebooks import
`isa_phm` normally and do not search for a local checkout. If you see
`ModuleNotFoundError: No module named 'isa_phm'`, select the environment
where you installed it or install it into the selected kernel's environment.

Open a notebook and edit its first dataset configuration cell. Replace
`path/to/your-dataset.isa-phm.json` with the actual ISA-PHM JSON file on your
computer. The examples use that file's parent as the dataset root; change
`DATA_ROOT` if your dataset uses another root. Dataset-specific notebooks may
also live alongside their local datasets. The datasets are not included in this
repository.

The notebooks in `02_features` and the prognostic workflow were written for a
specific XJTU dataset: some refer to named studies, multiple sensors, or the
full collection of bearings. Pointing them at another dataset, including the
small subset, can require changes to those selections. The diagnostic workflow
similarly expects compatible diagnostic data. The examples do not guarantee
that a feature, label, sampling rate, or unit is suitable for every dataset.

The notebooks are distributed without saved outputs. Run cells in order after
configuring your own dataset.

For the XJTU raw-and-mean-removed example, start with
[`04_raw_and_processed.ipynb`](03_workflows/04_raw_and_processed.ipynb).
It compares one measurement's explicitly selected raw and processed files,
the recorded protocol, and built-in signal features with source provenance.
