"""Keep the example notebooks' ordinary Python cells syntactically valid."""

import ast
import json
from pathlib import Path
import re

import pytest


NOTEBOOKS = sorted((Path(__file__).resolve().parents[1] / "notebooks").rglob("*.ipynb"))


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_code_cell_syntax(notebook):
    document = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        # IPython magics are not Python syntax; they require a notebook kernel.
        if any(line.lstrip().startswith(("%", "!")) for line in source.splitlines()):
            continue
        try:
            ast.parse(source)
        except SyntaxError as exc:
            pytest.fail(f"{notebook.name}, cell {index}: {exc}")


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_source_has_no_machine_specific_windows_paths(notebook):
    document = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        source = "".join(cell["source"])
        assert not re.search(r"\b[A-Za-z]:[\\/]", source), (
            f"{notebook.name}, cell {index} contains a Windows absolute path"
        )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_has_no_saved_outputs(notebook):
    document = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        if cell["cell_type"] == "code":
            assert not cell.get("outputs"), f"{notebook.name}, cell {index} has saved output"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.name)
def test_notebook_does_not_modify_python_import_path(notebook):
    document = json.loads(notebook.read_text(encoding="utf-8"))
    for index, cell in enumerate(document["cells"]):
        source = "".join(cell["source"])
        assert "sys.path" not in source, f"{notebook.name}, cell {index} modifies sys.path"
