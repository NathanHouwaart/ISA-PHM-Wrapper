"""
isa_phm — Python wrapper for ISA-PHM datasets.

Primary entry point: :class:`ISAWrapper`.

Quick start::

    from isa_phm import ISAWrapper

    wrapper = ISAWrapper("i_investigation.json", data_root="data/")
    df = wrapper.study("Case 01").assay("a_st01_se01").load_dataframe()
"""

from .wrapper import ISAWrapper

__all__ = ["ISAWrapper"]
__version__ = "0.1.0"
