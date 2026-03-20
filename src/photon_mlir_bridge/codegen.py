"""
CodeGen: lowers a PhotonicGraph to executable pseudocode for a
silicon photonic accelerator.

The output is a human-readable instruction stream that maps directly
to hardware-level control sequences (phase settings, drive voltages,
readout triggers).
"""

from __future__ import annotations
from typing import List

from .dialect import PhotonicDialect
from .ir import PhotonicGraph, PhotonicOp
from .rewriter import GraphRewriter


# Mapping from op name to instruction template
_INSTRUCTION_TEMPLATES = {
    "photonic.mzi": (
        "MZI_SET   theta={theta:.4f} phi={phi:.4f}  "
        "in=({inputs})  out=({outputs})"
    ),
    "photonic.detect": (
        "DETECT    responsivity={responsivity:.2f}  "
        "in=({inputs})  out=({outputs})"
    ),
    "photonic.modulate": (
        "MODULATE  bw={bandwidth_ghz:.1f}GHz  "
        "in=({inputs})  out=({outputs})"
    ),
    "photonic.mesh": (
        "MESH_MVM  rows={rows}  cols={cols}  layer={layer_name}  "
        "in=({inputs})  out=({outputs})"
    ),
    "photonic.detect_modulate": (
        "DET_MOD   fused=True  "
        "in=({inputs})  out=({outputs})"
    ),
}

_DEFAULT_TEMPLATE = "OP {name}  in=({inputs})  out=({outputs})  attrs={attrs}"


class CodeGen:
    """
    Emits pseudocode for a PhotonicGraph.

    Parameters
    ----------
    dialect : PhotonicDialect
        Used for op validation and attribute defaults.
    """

    def __init__(self, dialect: PhotonicDialect) -> None:
        self.dialect = dialect

    # ------------------------------------------------------------------
    # Single-op emission
    # ------------------------------------------------------------------

    def emit_op(self, op: PhotonicOp) -> str:
        """
        Emit a single pseudocode instruction for *op*.

        Falls back to a generic template for unrecognised op names.
        """
        tmpl = _INSTRUCTION_TEMPLATES.get(op.name, _DEFAULT_TEMPLATE)

        # Build a namespace for template substitution
        ns = {
            "name": op.name,
            "inputs": ", ".join(op.inputs) if op.inputs else "∅",
            "outputs": ", ".join(op.outputs) if op.outputs else "∅",
            "attrs": op.attrs,
        }
        # Merge op attrs (with defaults where missing)
        ns.update(op.attrs)
        # Fill missing numeric placeholders with 0 / ""
        for placeholder in ("theta", "phi", "responsivity", "bandwidth_ghz",
                            "rows", "cols", "layer_name"):
            if placeholder not in ns:
                ns[placeholder] = 0

        try:
            return tmpl.format(**ns)
        except (KeyError, ValueError):
            # Fallback if template fails
            return _DEFAULT_TEMPLATE.format(**ns)

    # ------------------------------------------------------------------
    # Full-graph emission
    # ------------------------------------------------------------------

    def emit_graph(self, graph: PhotonicGraph) -> str:
        """
        Emit a complete pseudocode program for *graph*.

        Returns a multi-line string with a header, body, and footer.
        """
        lines: List[str] = []
        lines.append(f"; === photon-mlir-bridge codegen ===")
        lines.append(f"; graph: @{graph.name}")
        lines.append(f"; ops:   {len(graph)}")
        lines.append("; " + "-" * 60)

        for i, op in enumerate(graph.get_ops()):
            lines.append(f"  [{i:04d}]  {self.emit_op(op)}")

        lines.append("; " + "-" * 60)
        lines.append(f"; end @{graph.name}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Full compilation pipeline
    # ------------------------------------------------------------------

    def compile(self, graph: PhotonicGraph) -> str:
        """
        Full compilation pipeline:
          1. Validate all ops against the dialect registry
          2. Run the optimizer (fuse detect+modulate pairs)
          3. Emit pseudocode

        Returns the compiled code string.
        """
        # Step 1: validate
        for op in graph.get_ops():
            try:
                self.dialect.get_op(op.name)
            except KeyError:
                raise ValueError(
                    f"Op {op.name!r} in graph @{graph.name!r} is not "
                    f"registered in the dialect."
                ) from None

        # Step 2: optimize
        rewriter = GraphRewriter(self.dialect)
        optimized = rewriter.optimize(graph)

        # Step 3: emit
        return self.emit_graph(optimized)
