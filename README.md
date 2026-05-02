# Seaf Energy Kernel

Seaf Energy Kernel is the sparse-constraint and equilibrium-learning part of
the three-kernel AI stack. It models prediction as the equilibrium of a graph
of local constraints and trains restriction maps with Equilibrium Propagation.

The PDF calls this idea "Sheaf Energy Kernel"; this folder keeps the requested
project name while the code namespace remains `sheaf_energy`.

## What Belongs Here

- Sparse graph with diagonal sheaf restriction maps.
- Coordinate solver for free and nudged equilibria.
- Auditable per-vertex and per-edge energy decomposition.
- Explicit Equilibrium Propagation gradients for restriction maps.
- Pure-Python reference implementation and C++17 performance core.

## Layout

```text
include/sheaf_energy/   C++17 header-only performance core
python/sheaf_energy/    Pure-Python mathematical reference
tests/cpp/              C++ tests and microbenchmarks
tests/python/           Python unit tests
```

## Build And Test

```bash
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

```bash
PYTHONPATH=python python -m unittest discover -s tests/python -v
```

PowerShell:

```powershell
$env:PYTHONPATH='python'; python -m unittest discover -s tests/python -v
```

## Mathematical Contract

For diagonal restriction maps, the edge energy is:

```text
||R_eu h_u - R_ev h_v||^2
```

The implemented Equilibrium Propagation estimator is:

```text
grad J(theta) ~= (grad_theta E(h_beta) - grad_theta E(h_0)) / beta
theta <- theta - eta * grad J(theta)
```

Both C++ and Python tests include a finite-difference check for this gradient.
