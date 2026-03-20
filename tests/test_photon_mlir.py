"""
Tests for the photon-mlir-bridge compiler.

Run with:
    ~/anaconda3/bin/python3 -m pytest tests/ -v
"""

import sys
import os
import pytest

# Allow running without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photon_mlir_bridge import PhotonicDialect, OpDef, PhotonicOp, PhotonicGraph
from photon_mlir_bridge import GraphRewriter, CodeGen


# ──────────────────────────────────────────────────────────────────────
# PhotonicDialect tests
# ──────────────────────────────────────────────────────────────────────

class TestPhotonicDialect:

    def test_dialect_init(self):
        """Dialect initialises without error and has ops registered."""
        d = PhotonicDialect()
        assert len(d.list_ops()) > 0

    def test_dialect_register_op(self):
        """Custom ops can be registered."""
        d = PhotonicDialect()
        before = len(d.list_ops())
        d.register_op("custom.op", num_inputs=3, num_outputs=2, attrs={"gain": 1.5})
        assert len(d.list_ops()) == before + 1
        assert "custom.op" in d.list_ops()

    def test_dialect_get_op(self):
        """get_op returns correct OpDef."""
        d = PhotonicDialect()
        op_def = d.get_op("photonic.mzi")
        assert isinstance(op_def, OpDef)
        assert op_def.name == "photonic.mzi"
        assert op_def.num_inputs == 2
        assert op_def.num_outputs == 2

    def test_dialect_get_op_unknown_raises(self):
        """get_op raises KeyError for unknown ops."""
        d = PhotonicDialect()
        with pytest.raises(KeyError):
            d.get_op("nonexistent.op")

    def test_dialect_builtin_ops(self):
        """All expected built-in ops are present."""
        d = PhotonicDialect()
        ops = d.list_ops()
        assert "photonic.mzi" in ops
        assert "photonic.detect" in ops
        assert "photonic.modulate" in ops
        assert "photonic.mesh" in ops

    def test_dialect_list_ops(self):
        """list_ops returns a sorted list of strings."""
        d = PhotonicDialect()
        ops = d.list_ops()
        assert isinstance(ops, list)
        assert all(isinstance(o, str) for o in ops)
        assert ops == sorted(ops)


# ──────────────────────────────────────────────────────────────────────
# IR tests
# ──────────────────────────────────────────────────────────────────────

class TestPhotonicOp:

    def test_photonic_op_creation(self):
        """PhotonicOp stores name, inputs, outputs, attrs."""
        op = PhotonicOp(
            name="photonic.mzi",
            inputs=["in_0", "in_1"],
            outputs=["out_0", "out_1"],
            attrs={"theta": 0.5, "phi": 1.2},
        )
        assert op.name == "photonic.mzi"
        assert op.inputs == ["in_0", "in_1"]
        assert op.outputs == ["out_0", "out_1"]
        assert op.attrs["theta"] == 0.5
        assert isinstance(op.id, str) and len(op.id) > 0

    def test_photonic_op_defaults(self):
        """PhotonicOp has sane defaults for optional fields."""
        op = PhotonicOp(name="photonic.detect")
        assert op.inputs == []
        assert op.outputs == []
        assert op.attrs == {}


class TestPhotonicGraph:

    def test_photonic_graph_add_op(self):
        """add_op inserts ops into the graph."""
        g = PhotonicGraph("test_graph")
        op = PhotonicOp("photonic.mzi", ["a", "b"], ["c", "d"])
        g.add_op(op)
        assert len(g) == 1
        assert g.get_ops()[0] is op

    def test_photonic_graph_get_ops(self):
        """get_ops returns all ops in insertion order."""
        g = PhotonicGraph("test_graph")
        ops = [PhotonicOp(f"photonic.mzi") for _ in range(3)]
        for op in ops:
            g.add_op(op)
        retrieved = g.get_ops()
        assert len(retrieved) == 3
        for orig, ret in zip(ops, retrieved):
            assert orig is ret

    def test_photonic_graph_repr(self):
        """__repr__ includes graph name and op listing."""
        g = PhotonicGraph("my_graph")
        g.add_op(PhotonicOp("photonic.detect", ["x"], ["y"]))
        r = repr(g)
        assert "my_graph" in r
        assert "photonic.detect" in r
        assert r.startswith("graph @my_graph")

    def test_photonic_graph_chaining(self):
        """add_op returns the graph for chaining."""
        g = PhotonicGraph("chain")
        result = g.add_op(PhotonicOp("photonic.mzi"))
        assert result is g


# ──────────────────────────────────────────────────────────────────────
# GraphRewriter tests
# ──────────────────────────────────────────────────────────────────────

class TestGraphRewriter:

    def test_rewriter_rewrite_linear(self):
        """rewrite_linear adds a photonic.mesh op to the graph."""
        d = PhotonicDialect()
        rw = GraphRewriter(d)
        g = PhotonicGraph("test")
        rw.rewrite_linear(g, in_features=4, out_features=3, name="fc")
        ops = g.get_ops()
        assert len(ops) == 1
        assert ops[0].name == "photonic.mesh"
        assert ops[0].attrs["rows"] == 3
        assert ops[0].attrs["cols"] == 4
        assert ops[0].attrs["layer_name"] == "fc"

    def test_rewriter_rewrite_mlp(self):
        """rewrite_mlp produces a graph with mesh + detect + modulate ops."""
        d = PhotonicDialect()
        rw = GraphRewriter(d)
        graph = rw.rewrite_mlp([4, 3, 2])
        ops = graph.get_ops()
        # Should have: mesh + 3 detects + 3 modulates + mesh + 2 detects
        assert len(ops) > 0
        op_names = [op.name for op in ops]
        assert "photonic.mesh" in op_names
        assert "photonic.detect" in op_names
        assert "photonic.modulate" in op_names

    def test_rewriter_mlp_layer_count(self):
        """rewrite_mlp [784, 128, 10] produces correct number of mesh ops."""
        d = PhotonicDialect()
        rw = GraphRewriter(d)
        graph = rw.rewrite_mlp([784, 128, 10])
        mesh_ops = [op for op in graph.get_ops() if op.name == "photonic.mesh"]
        assert len(mesh_ops) == 2  # one per layer transition

    def test_rewriter_mlp_invalid_raises(self):
        """rewrite_mlp raises ValueError for single-element layer list."""
        d = PhotonicDialect()
        rw = GraphRewriter(d)
        with pytest.raises(ValueError):
            rw.rewrite_mlp([64])

    def test_rewriter_optimize(self):
        """optimize fuses detect→modulate pairs."""
        d = PhotonicDialect()
        rw = GraphRewriter(d)
        g = PhotonicGraph("test_opt")
        det = PhotonicOp("photonic.detect", ["x"], ["y"], {"responsivity": 1.0})
        mod = PhotonicOp("photonic.modulate", ["y"], ["z"], {"bandwidth_ghz": 50.0})
        g.add_op(det)
        g.add_op(mod)
        opt = rw.optimize(g)
        ops = opt.get_ops()
        assert len(ops) == 1
        assert ops[0].name == "photonic.detect_modulate"
        assert ops[0].inputs == ["x"]
        assert ops[0].outputs == ["z"]


# ──────────────────────────────────────────────────────────────────────
# CodeGen tests
# ──────────────────────────────────────────────────────────────────────

class TestCodeGen:

    def test_codegen_emit_op(self):
        """emit_op returns a non-empty string for a known op."""
        d = PhotonicDialect()
        cg = CodeGen(d)
        op = PhotonicOp("photonic.mzi", ["a", "b"], ["c", "d"],
                        {"theta": 0.3, "phi": 0.7})
        result = cg.emit_op(op)
        assert isinstance(result, str)
        assert len(result) > 0
        assert "MZI_SET" in result

    def test_codegen_emit_graph(self):
        """emit_graph returns a multi-line program string."""
        d = PhotonicDialect()
        cg = CodeGen(d)
        g = PhotonicGraph("small")
        g.add_op(PhotonicOp("photonic.detect", ["x"], ["y"], {"responsivity": 1.0}))
        code = cg.emit_graph(g)
        assert isinstance(code, str)
        assert "small" in code
        assert "DETECT" in code

    def test_end_to_end_compile(self):
        """Full pipeline: MLP [784, 128, 10] compiles without error."""
        d = PhotonicDialect()
        rw = GraphRewriter(d)
        cg = CodeGen(d)

        graph = rw.rewrite_mlp([784, 128, 10])
        code = cg.compile(graph)

        assert isinstance(code, str)
        assert len(code) > 0
        # Header present
        assert "photon-mlir-bridge" in code
        # Key ops present
        assert "MESH_MVM" in code
        assert "DETECT" in code
