# A "proof-of-concept" preliminary implementation of TQSMatrices

"""
	TQSMatrices

Proof-of-concept implementation of Tree Quasi-Separable (TQS) matrices.

A TQS matrix `T` is an `M × N` matrix whose rows and columns are partitioned
according to the nodes of a tree. Node `i` owns the output block `b[i]`
(`m[i]` rows) and the input block `x[i]` (`n[i]` columns). Instead of storing `T`
densely, each node stores a small set of generator matrices (a [`Spinner`](@ref)).
Information moves between neighbouring nodes `i → j` through a *state* vector
`h[j][i]` of dimension `ρ[(i, j)]`.

# Types

- [`UndirectedGraph`](@ref): an undirected graph on integer node labels (cycles allowed).
- [`AbstractRootedTree`](@ref): a connected acyclic `UndirectedGraph` together with
  a root. Two orientations, which only differ in how the nodes are grouped into
  parallel rounds:
  - [`RootOrientedTree`](@ref): levels counted down from the root (any root).
  - [`LeafOrientedTree`](@ref): leaf levels (k-leaves) counted up from the leaves,
    i.e. onion peeling (the root is a [`center`](@ref) node).
- [`BlockPartition`](@ref): how the rows and columns are split into one block per node.
- [`GraphPartitionedMatrix`](@ref): a dense matrix with a graph and a block partition.
- [`Spinner`](@ref): the generators of a single node.
- [`GIRSRepresentation`](@ref): spinners on any graph (cycles and bounce-back
  operators allowed).
- [`TQSMatrix`](@ref): spinners on a rooted tree, without bounce-back operators.
  Same fields as a `GIRSRepresentation`, with the graph replaced by a rooted tree.
  The tree type is a type parameter: [`RootOrientedTQSMatrix`](@ref) and
  [`LeafOrientedTQSMatrix`](@ref).

# Typical workflow

```julia
A = GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list)
T, ρ = TQSMatrix(A, root, 1e-12, Inf)          # construct TQS representation
b = T * x                                      # fast mat-vec
x = ulvsolve(T, b)                             # fast solve (square blocks only)
x = sparsesolve(T, b)                          # sparse embedding: METIS order, UMFPACK LU (also for GIRS)
relative_error(A.mat, Matrix(T))               # check accuracy

G = GIRSRepresentation([spinner1, spinner2, ...])   # spinners on a graph
T = TQSMatrix(G, root)                              # requires a tree, no bounce-backs
T = TQSMatrix(G, LeafOrientedTree(G.graph))         # leaf-oriented, rooted at the center
```

# Notation used throughout

- `m[i]`, `n[i]`: row/column block size of node `i`.
- `mrange[i]`, `nrange[i]`: row/column index ranges of node `i` in the full matrix
  (stored in a [`BlockPartition`](@ref)).
- `p_in[j]` (on node `i`): dimension of the state that arrives at `i` from `j`.
- `p_out[j]` (on node `i`): dimension of the state that leaves `i` towards `j`.
- `ρ[(i, j)]`: state dimension on the directed edge `i → j`, so
  `ρ[(i, j)] == spinner_i.p_out[j] == spinner_j.p_in[i]`.
"""
module TQSMatrices

###########
# exports #
###########
export ZeroMatrix, IndexedVector, IndexedMatrix
export BlockPartition, construct_range_vec
export UndirectedGraph, is_connected_acyclic, peel_layers, center
export AbstractRootedTree, RootOrientedTree, LeafOrientedTree, upsweep_rounds, depth
export GraphPartitionedMatrix, get_block, get_hankelblock
export Spinner, GIRS_is_consistent, has_no_bounce_back_operators
export GIRSRepresentation, TQSMatrix, RootOrientedTQSMatrix, LeafOrientedTQSMatrix
export StateGraph, TransGraph
export ulvsolve
export SparseEmbedding, lift, extract
export sparsesolve, SparseEmbeddingLU
export HankelFactorization, Xblock, Yblock, lowrankapprox, relative_error

############
# packages #
############
using LinearAlgebra
using SparseArrays
using IterTools
using Base.Iterators

include("containers.jl")
include("utils.jl")
include("partition.jl")
include("graphs.jl")
include("graph_partitioned.jl")
include("spinner.jl")
include("girs.jl")
include("tqsmatrix.jl")
include("matvec.jl")
include("construction.jl")
include("ulvsolve.jl")
include("sparse_embedding.jl")
include("sparsesolve.jl")

end
