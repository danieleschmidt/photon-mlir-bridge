"""
Intermediate Representation (IR) for the photonic compiler.

PhotonicOp  - a single computation node (like an MLIR operation)
PhotonicGraph - a DAG of PhotonicOps (like an MLIR region/block)
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import uuid


class PhotonicOp:
    """
    A single photonic operation node in the computation graph.

    Attributes
    ----------
    name    : op type, e.g. "photonic.mzi"
    inputs  : list of symbolic input wire names
    outputs : list of symbolic output wire names
    attrs   : op-specific attributes (theta, phi, etc.)
    id      : unique node identifier
    """

    def __init__(
        self,
        name: str,
        inputs: Optional[List[str]] = None,
        outputs: Optional[List[str]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.name = name
        self.inputs: List[str] = inputs or []
        self.outputs: List[str] = outputs or []
        self.attrs: Dict[str, Any] = attrs or {}
        self.id: str = str(uuid.uuid4())[:8]

    def __repr__(self) -> str:
        parts = [f"%{self.id} = {self.name}"]
        if self.inputs:
            parts.append(f"({', '.join(self.inputs)})")
        if self.attrs:
            attr_str = ", ".join(f"{k}={v!r}" for k, v in self.attrs.items())
            parts.append(f" [{attr_str}]")
        if self.outputs:
            parts.append(f" -> ({', '.join(self.outputs)})")
        return "".join(parts)


class PhotonicGraph:
    """
    A named computation graph composed of PhotonicOps.

    Behaves like a flat MLIR block: ordered list of ops with
    symbolic SSA-style wire names.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._ops: List[PhotonicOp] = []

    def add_op(self, op: PhotonicOp) -> "PhotonicGraph":
        """Append an op to the graph. Returns self for chaining."""
        self._ops.append(op)
        return self

    def get_ops(self) -> List[PhotonicOp]:
        """Return all ops in insertion order."""
        return list(self._ops)

    def __len__(self) -> int:
        return len(self._ops)

    def __repr__(self) -> str:
        lines = [f"graph @{self.name} {{"]
        for op in self._ops:
            lines.append(f"  {op!r}")
        lines.append("}")
        return "\n".join(lines)
