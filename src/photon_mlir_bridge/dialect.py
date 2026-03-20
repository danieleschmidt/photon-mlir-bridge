"""
PhotonicDialect: Registry for photonic operations.

Manages the set of valid ops that can appear in a PhotonicGraph.
Comes pre-loaded with core hardware primitives (MZI, detector, modulator).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OpDef:
    """Definition of a photonic operation."""
    name: str
    num_inputs: int
    num_outputs: int
    attrs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"OpDef({self.name!r}, inputs={self.num_inputs}, "
            f"outputs={self.num_outputs}, attrs={self.attrs})"
        )


class PhotonicDialect:
    """
    Registry of photonic operations understood by the compiler.

    Built-in ops:
      - photonic.mzi      : Mach-Zehnder interferometer (2-in, 2-out)
      - photonic.detect   : Photodetector (1-in, 1-out)
      - photonic.modulate : Optical modulator (1-in, 1-out)
      - photonic.mesh     : MZI mesh for matrix-vector multiply (n-in, m-out)
    """

    BUILTIN_OPS: List[Dict[str, Any]] = [
        {
            "name": "photonic.mzi",
            "num_inputs": 2,
            "num_outputs": 2,
            "attrs": {"theta": 0.0, "phi": 0.0},
        },
        {
            "name": "photonic.detect",
            "num_inputs": 1,
            "num_outputs": 1,
            "attrs": {"responsivity": 1.0},
        },
        {
            "name": "photonic.modulate",
            "num_inputs": 1,
            "num_outputs": 1,
            "attrs": {"bandwidth_ghz": 50.0},
        },
        {
            "name": "photonic.mesh",
            "num_inputs": -1,   # variable
            "num_outputs": -1,  # variable
            "attrs": {"rows": 0, "cols": 0},
        },
    ]

    def __init__(self) -> None:
        self._ops: Dict[str, OpDef] = {}
        for op in self.BUILTIN_OPS:
            self.register_op(
                op["name"],
                op["num_inputs"],
                op["num_outputs"],
                op["attrs"],
            )

    def register_op(
        self,
        name: str,
        num_inputs: int,
        num_outputs: int,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> OpDef:
        """Register a new (or overwrite an existing) operation."""
        op_def = OpDef(
            name=name,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            attrs=attrs or {},
        )
        self._ops[name] = op_def
        return op_def

    def get_op(self, name: str) -> OpDef:
        """Retrieve an op definition by name. Raises KeyError if unknown."""
        if name not in self._ops:
            raise KeyError(f"Unknown op: {name!r}. Registered ops: {list(self._ops)}")
        return self._ops[name]

    def list_ops(self) -> List[str]:
        """Return sorted list of all registered op names."""
        return sorted(self._ops.keys())

    def __repr__(self) -> str:
        return f"PhotonicDialect(ops={self.list_ops()})"
