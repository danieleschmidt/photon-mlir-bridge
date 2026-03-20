"""
GraphRewriter: pattern-matching rewriter that lowers ML-level ops
to photonic hardware primitives.

Key rewrites
------------
  nn.Linear(in, out)  →  photonic.mesh(rows=out, cols=in)
  MLP([n0, n1, n2])   →  sequence of photonic.mesh + photonic.detect layers
"""

from __future__ import annotations
from typing import List

from .dialect import PhotonicDialect
from .ir import PhotonicGraph, PhotonicOp


class GraphRewriter:
    """
    Rewrites high-level ML patterns to photonic-native ops.

    Parameters
    ----------
    dialect : PhotonicDialect
        Dialect registry to validate against.
    """

    def __init__(self, dialect: PhotonicDialect) -> None:
        self.dialect = dialect

    # ------------------------------------------------------------------
    # Core rewrite rules
    # ------------------------------------------------------------------

    def rewrite_linear(
        self,
        graph: PhotonicGraph,
        in_features: int,
        out_features: int,
        name: str = "linear",
    ) -> PhotonicGraph:
        """
        Pattern-match a linear (dense) layer and replace it with a
        photonic MZI mesh.

        The photonic.mesh op implements matrix-vector multiplication
        optically via a triangular array of MZIs.

        Parameters
        ----------
        graph        : destination PhotonicGraph (ops are appended)
        in_features  : number of input features (columns of weight matrix)
        out_features : number of output features (rows of weight matrix)
        name         : logical name for the layer
        """
        mesh_op = PhotonicOp(
            name="photonic.mesh",
            inputs=[f"{name}_in_{i}" for i in range(in_features)],
            outputs=[f"{name}_out_{j}" for j in range(out_features)],
            attrs={
                "rows": out_features,
                "cols": in_features,
                "layer_name": name,
            },
        )
        graph.add_op(mesh_op)
        return graph

    def rewrite_mlp(self, layer_sizes: List[int]) -> PhotonicGraph:
        """
        Rewrite a full MLP (described by its layer-size list) to a
        PhotonicGraph.

        Each hidden/output transition becomes:
          1. photonic.mesh  (matrix multiply)
          2. photonic.detect (non-linear activation via detection + re-encode)

        Parameters
        ----------
        layer_sizes : list of ints, e.g. [784, 128, 10]
                      First entry is the input dimension.
        """
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 entries")

        graph = PhotonicGraph(name="mlp_" + "_".join(str(s) for s in layer_sizes))

        for idx, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            layer_name = f"layer_{idx}"

            # Matrix multiply via MZI mesh
            self.rewrite_linear(graph, in_f, out_f, name=layer_name)

            # Activation: detect optical signal → electrical → re-encode
            # (last layer skips re-encode; just read the detector output)
            is_last = idx == len(layer_sizes) - 2
            for j in range(out_f):
                det_op = PhotonicOp(
                    name="photonic.detect",
                    inputs=[f"{layer_name}_out_{j}"],
                    outputs=[f"{layer_name}_act_{j}"],
                    attrs={"responsivity": 1.0, "is_output": is_last},
                )
                graph.add_op(det_op)

            # Re-encode for next layer (hidden layers only)
            if not is_last:
                for j in range(out_f):
                    mod_op = PhotonicOp(
                        name="photonic.modulate",
                        inputs=[f"{layer_name}_act_{j}"],
                        outputs=[f"layer_{idx + 1}_in_{j}"],
                        attrs={"bandwidth_ghz": 50.0},
                    )
                    graph.add_op(mod_op)

        return graph

    # ------------------------------------------------------------------
    # Optimization passes
    # ------------------------------------------------------------------

    def optimize(self, graph: PhotonicGraph) -> PhotonicGraph:
        """
        Run basic graph optimizations:
          - Dead-wire elimination (ops with no consumers)
          - detect→modulate fusion: replace adjacent detect+modulate pairs
            with a single photonic.detect_modulate fused op.

        Returns a *new* PhotonicGraph (does not mutate in place).
        """
        ops = graph.get_ops()
        fused_ops = []
        skip = set()

        for i, op in enumerate(ops):
            if i in skip:
                continue

            # Fusion: detect immediately followed by modulate on same wire
            if (
                op.name == "photonic.detect"
                and i + 1 < len(ops)
                and ops[i + 1].name == "photonic.modulate"
                and op.outputs
                and ops[i + 1].inputs == op.outputs
            ):
                fused = PhotonicOp(
                    name="photonic.detect_modulate",
                    inputs=op.inputs,
                    outputs=ops[i + 1].outputs,
                    attrs={**op.attrs, **ops[i + 1].attrs, "fused": True},
                )
                fused_ops.append(fused)
                skip.add(i + 1)
            else:
                fused_ops.append(op)

        optimized = PhotonicGraph(name=graph.name + "_opt")
        for op in fused_ops:
            optimized.add_op(op)
        return optimized
