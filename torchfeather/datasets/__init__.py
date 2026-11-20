from collections.abc import Callable
from dataclasses import dataclass

"""
if someone try to import from this_module import *, 
only export DatasetConfig." It's just export control — no runtime effect on the class itself
"""
__all__ = ["DatasetConfig"]

@dataclass
class DatasetConfig:
    path: str
    loader: Callable
    sample_processor: Callable