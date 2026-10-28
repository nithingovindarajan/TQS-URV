# A "proof-of-concept" preliminary implementation of TQSMatrices

"""
	TQSMatrices

Proof-of-concept implementation of Tree Quasi-Separable (TQS) matrices.

A TQS matrix `T` is an `M × N` matrix whose rows and columns are partitioned
according to the nodes of a tree graph. Node `i` owns the output block `b[i]`
(`m[i]` rows) and the input block `x[i]` (`n[i]` columns). Instead of storing `T`
densely, each node stores a small set of generator matrices (a [`Spinner`](@ref)).
Information moves between neighbouring nodes `i → j` through a *state* vector
`h[j][i]` of dimension `ρ[(i, j)]`.

Main entry points:

- [`Spinner`](@ref): the generators of a single node.
- [`TQSMatrix`](@ref): a collection of spinners on a tree. It can be built from
  spinners directly, or computed from a dense [`GraphPartitionedMatrix`](@ref)
  with `TQSMatrix(A, root, tol, ρ_max)` (the construction algorithm).
- `*(T, x, tree)`: a fast matrix-vector product using an upsweep/downsweep over
  the tree.
- `Matrix(T, tree)` and `GraphPartitionedMatrix(T, tree)`: convert back to dense.

# Typical workflow

```julia
A = GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list)
T, tree, ρ = TQSMatrix(A, root, 1e-12, Inf)   # construct TQS representation
b = *(T, x, tree)                             # fast mat-vec
relative_error(A.mat, Matrix(T, tree))        # check accuracy
```

# Notation used throughout

- `m[i]`, `n[i]`: row/column block size of node `i`.
- `mrange[i]`, `nrange[i]`: row/column index ranges of node `i` in the full matrix.
- `p_in[j]` (on node `i`): dimension of the state that arrives at `i` from `j`.
- `p_out[j]` (on node `i`): dimension of the state that leaves `i` towards `j`.
- `ρ[(i, j)]`: state dimension on the directed edge `i → j`, so
  `ρ[(i, j)] == spinner_i.p_out[j] == spinner_j.p_in[i]`.
"""
module TQSMatrices

###########
# exports #
###########
export ZeroMatrix, Spinner, GIRS_is_consistent, TQSMatrix, IndexedVector, IndexedMatrix
export is_a_tree, GIRS_has_no_bounce_back_operators, construct_tree
export GraphPartitionedMatrix, get_block, get_hankelblock, StateGraph, TransGraph
export construct_range_vec, HankelFactorization, Xblock, Yblock
export lowrankapprox, relative_error

############
# packages #
############
using LinearAlgebra
using IterTools
using Base.Iterators

include("containers.jl")
include("utils.jl")
include("tree.jl")
include("graph_partitioned.jl")
include("spinner.jl")
include("tqsmatrix.jl")
include("matvec.jl")
include("construction.jl")

end
