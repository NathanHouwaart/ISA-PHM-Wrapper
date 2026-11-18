"""Bundle the pinned specification rule file into distribution artifacts."""
from pathlib import Path
import shutil

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildWithSpecification(build_py):
    def run(self):
        source = Path(__file__).parent / "vendor/isa-phm-specification/specification/quantity-unit-warning-rules.json"
        if not source.is_file():
            raise RuntimeError("ISA-PHM specification submodule missing; initialize submodules before building.")
        super().run()
        target = Path(self.build_lib) / "isa_phm/_specification/quantity-unit-warning-rules.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


setup(cmdclass={"build_py": BuildWithSpecification})
