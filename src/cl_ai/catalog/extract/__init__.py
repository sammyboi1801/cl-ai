"""Source-tier extractors.

Each tier implements the `Source` protocol from `base` and yields `RawTool`
findings tagged with their provenance. Stage 3 (`catalog.normalize`) merges
them under highest-tier-wins.
"""

from .base import (
    DESTRUCTIVE_HINTS,
    OS_TO_SHELLS,
    ParamKind,
    RawExample,
    RawParam,
    RawTool,
    Source,
    infer_capabilities,
    shells_for_os,
)
from .tldr import TldrSource

__all__ = [
    "DESTRUCTIVE_HINTS",
    "OS_TO_SHELLS",
    "ParamKind",
    "RawExample",
    "RawParam",
    "RawTool",
    "Source",
    "TldrSource",
    "infer_capabilities",
    "shells_for_os",
]
