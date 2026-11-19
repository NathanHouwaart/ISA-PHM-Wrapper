"""The validator dependency and shared catalogues must move together."""
import re
import subprocess
from pathlib import Path

import pytest


def test_validator_dependency_matches_specification_submodule():
    root = Path(__file__).resolve().parents[1]
    submodule = root / 'vendor/isa-phm-specification'
    if not (submodule / '.git').exists():
        pytest.skip('Git submodule metadata is not available in this source distribution')
    text = (root / 'pyproject.toml').read_text(encoding='utf-8')
    match = re.search(r'isa-phm-validator @ git\+https://github\.com/NathanHouwaart/ISA-PHM-Specification\.git@([0-9a-f]{40})', text)
    assert match, 'Pin the validator dependency to a full Specification commit'
    commit = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(submodule), 'rev-parse', 'HEAD'],
                            check=True, capture_output=True, text=True).stdout.strip()
    assert commit == match.group(1), 'Update the validator dependency and Specification submodule together'
