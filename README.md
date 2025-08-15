# ArtificialLandscapes

[![Build Status](https://github.com/NicolasL-S/ArtificialLandscapes.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/NicolasL-S/ArtificialLandscapes.jl/actions/workflows/CI.yml?query=branch%3Amain)

ArtificialLandscapes is a pure-Julia collection of well-known optimization test problems. It is designed to be fast by avoiding non-Julia libraries and using manually-coded derivatives. Combined with usual tricks like memory preallocation, type stability, and minimal abstraction, gradients can be evaluated 1000x faster than in similar packages.

This project is in its infancy. At the moment, only starting points, objective functions and gradients are provided; no Hessian yet. Also, only unconstrained problems have been added so far.

To install:
```Julia
] add ArtificialLandscapes
```

### API
Functions stored in the dictionary ``landscapes`` generate problems as ``@NamedTuple``.  To see the full list of problems:
```Julia
using ArtificialLandscapes
keys(landscapes)
```
All problems provide a starting point, objective and an in-place gradient function. To generate a problem:
```Julia
x0, obj, grad! = landscapes["LUKSAN11LS"]()
```
Solving using Optim:
```Julia
using Optim
optimize(obj, grad!, x0, Optim.Options(iterations = 10000))
```
The problems can be generated with varying precisions, and sometimes varying specifications:
```Julia
x0, obj, grad! = landscapes["Extended Powell"](;n = 100, T = Float16)
```
To learn more about a problem and its available keyword arguments, it must be extracted from the dictionary.
```Julia
p = landscapes["LUKSAN11LS"]
? p
```
### Accuracy

As much as possible, problems should be faithful to their original versions. A notable exception are those from the CUTEst library. Since they have become a widely accepted standard, every effort is made to match the outputs of CUTEst.jl, even in the rare occasions when they deviate from the original.

### See also

OptimTestProblems.jl, OptimizationTestFunctions.jl, BenchmarkFunctions.jl, CUTEst.jl, OptimizationProblems.jl, S2MPJ.jl.