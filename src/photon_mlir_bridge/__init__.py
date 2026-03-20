"""
photon_mlir_bridge
==================

A Python compiler that bridges ML frameworks to silicon photonic accelerators.

Pipeline
--------
  ML model description
       │
       ▼
  GraphRewriter  ─── rewrites nn.Linear / MLP → photonic ops
       │
       ▼
  PhotonicGraph  ─── IR: ordered list of PhotonicOps
       │
       ▼
  CodeGen        ─── emits hardware pseudocode (phase settings, drive voltages)

Example
-------
>>> from photon_mlir_bridge import PhotonicDialect, GraphRewriter, CodeGen
>>> dialect = PhotonicDialect()
>>> rw = GraphRewriter(dialect)
>>> graph = rw.rewrite_mlp([784, 128, 10])
>>> cg = CodeGen(dialect)
>>> print(cg.compile(graph))
"""

from .dialect import PhotonicDialect, OpDef
from .ir import PhotonicOp, PhotonicGraph
from .rewriter import GraphRewriter
from .codegen import CodeGen

__all__ = [
    "PhotonicDialect",
    "OpDef",
    "PhotonicOp",
    "PhotonicGraph",
    "GraphRewriter",
    "CodeGen",
]

__version__ = "0.1.0"
