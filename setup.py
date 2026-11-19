"""Bundle the pinned draft specification inputs into distribution artifacts."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithSpecification(build_py):
    def run(self):
        root = Path(__file__).parent / "vendor/isa-phm-specification"
        files = {
            "specification/quantity-unit-warning-rules.json": "quantity-unit-warning-rules.json",
            "catalogues/qudt-units-3.5.2.json": "catalogues/qudt-units-3.5.2.json",
            "catalogues/qudt-quantities-3.5.2.json": "catalogues/qudt-quantities-3.5.2.json",
            "catalogues/isa-phm-quantities.json": "catalogues/isa-phm-quantities.json",
        }
        if any(not (root / name).is_file() for name in files):
            raise RuntimeError("ISA-PHM specification submodule missing; initialize submodules before building.")
        super().run()
        for source_name, target_name in files.items():
            target = Path(self.build_lib) / "isa_phm/_specification" / target_name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / source_name, target)


setup(cmdclass={"build_py": BuildWithSpecification})
