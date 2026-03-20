# photon-mlir-bridge

A Python compiler that bridges ML frameworks to silicon photonic accelerators.

## Overview

Silicon photonic chips perform matrix-vector multiplication at the speed of light using Mach-Zehnder interferometer (MZI) meshes. This compiler lowers ML models (starting with linear/dense layers and MLPs) to hardware pseudocode for photonic ASICs.

```
  ML model (nn.Linear, MLP)
         │
         ▼
  GraphRewriter  ──  rewrites dense layers → photonic.mesh ops
         │
         ▼
  PhotonicGraph  ──  IR: ordered list of PhotonicOps
         │
         ▼
  CodeGen        ──  emits hardware pseudocode
                     (phase settings, drive voltages, readout triggers)
```

## Quick Start

```python
from photon_mlir_bridge import PhotonicDialect, GraphRewriter, CodeGen

# 1. Create dialect (registers built-in ops)
dialect = PhotonicDialect()

# 2. Rewrite a 2-layer MLP to a photonic graph
rw = GraphRewriter(dialect)
graph = rw.rewrite_mlp([784, 128, 10])

# 3. Compile to pseudocode
cg = CodeGen(dialect)
code = cg.compile(graph)
print(code)
```

Output (excerpt):
```
; === photon-mlir-bridge codegen ===
; graph: @mlp_784_128_10_opt
; ops:   144
; ------------------------------------------------------------
  [0000]  MESH_MVM  rows=128  cols=784  layer=layer_0  in=(layer_0_in_0, ...) out=(...)
  [0001]  DET_MOD   fused=True  in=(layer_0_out_0)  out=(layer_1_in_0)
  ...
```

## Architecture

### `PhotonicDialect`
Registry of valid photonic ops. Built-in ops:

| Op | Description | Inputs | Outputs |
|----|-------------|--------|---------|
| `photonic.mzi` | Mach-Zehnder interferometer | 2 | 2 |
| `photonic.detect` | Photodetector | 1 | 1 |
| `photonic.modulate` | Optical modulator | 1 | 1 |
| `photonic.mesh` | MZI mesh (matrix multiply) | N | M |

### `PhotonicGraph` (IR)
Ordered list of `PhotonicOp` nodes. Each op has a name, input/output wire list, and attribute dict. Prints as a readable IR dump:

```
graph @my_graph {
  %a1b2c3d4 = photonic.mesh [rows=10, cols=4, layer_name='fc'] -> (out_0, out_1, ...)
}
```

### `GraphRewriter`
Pattern-matching rewriter:
- `rewrite_linear(graph, in, out)` → inserts `photonic.mesh`
- `rewrite_mlp(sizes)` → full MLP lowering (mesh + detect + modulate per layer)
- `optimize(graph)` → fuses `detect→modulate` pairs into `detect_modulate`

### `CodeGen`
Lowers `PhotonicGraph` → pseudocode string:
- `emit_op(op)` → single instruction
- `emit_graph(graph)` → full program
- `compile(graph)` → validate + optimize + emit

## Running the Demo

```bash
python examples/demo.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Requirements

Pure Python standard library — no external dependencies.

## License

MIT
