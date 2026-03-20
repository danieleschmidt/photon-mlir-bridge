"""
demo.py — End-to-end demonstration of the photon-mlir-bridge compiler.

Compiles a 2-layer MLP (784→128→10) to silicon-photonic pseudocode.

Usage
-----
    python examples/demo.py
"""

import sys
import os

# Allow running from the repo root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from photon_mlir_bridge import PhotonicDialect, GraphRewriter, CodeGen


def main() -> None:
    print("=" * 65)
    print("  photon-mlir-bridge  |  2-layer MLP demo")
    print("=" * 65)

    # ── 1. Create the dialect (registers built-in ops) ──────────────────
    dialect = PhotonicDialect()
    print(f"\n[dialect] {dialect}")

    # ── 2. Rewrite a 2-layer MLP to a PhotonicGraph ─────────────────────
    layer_sizes = [784, 128, 10]
    print(f"\n[rewriter] Lowering MLP {layer_sizes} → photonic graph …")

    rw = GraphRewriter(dialect)
    graph = rw.rewrite_mlp(layer_sizes)

    print(f"[rewriter] Graph '{graph.name}' created with {len(graph)} ops")

    # ── 3. Show the IR dump ──────────────────────────────────────────────
    print("\n── IR dump (first 12 ops) ──────────────────────────────────")
    for op in graph.get_ops()[:12]:
        print(f"  {op!r}")
    if len(graph) > 12:
        print(f"  … ({len(graph) - 12} more ops)")

    # ── 4. Compile → pseudocode ──────────────────────────────────────────
    print("\n── Compiled pseudocode (excerpt) ───────────────────────────")
    cg = CodeGen(dialect)
    code = cg.compile(graph)
    lines = code.splitlines()
    # Print header + first 15 body lines + footer
    cutoff = 18
    for line in lines[:cutoff]:
        print(line)
    if len(lines) > cutoff + 2:
        print(f"  … ({len(lines) - cutoff - 2} more instructions)")
    for line in lines[-2:]:
        print(line)

    print("\n[done] Total compiled lines:", len(lines))
    print("=" * 65)


if __name__ == "__main__":
    main()
