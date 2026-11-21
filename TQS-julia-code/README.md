# Tree Quasi-Separable Matrices

An implementation of TQS matrices, and specifically, the construction algorithm to convert any dense matrix into a minimal TQS representation. The repository contains the experiments done in Table 1 of our paper.

## What is a TQS matrix?

A TQS matrix is a matrix whose rows and columns are split into blocks, one per node of a tree (a connected acyclic undirected graph with a chosen root). Node `i` owns the input block `x[i]` and the output block `b[i]`. The matrix is not stored densely. Instead, each node holds a **spinner**: a few small generator matrices.

| Generator     | Size                | Role                                                     |
| ------------- | ------------------- | -------------------------------------------------------- |
| `D`           | `m × n`             | diagonal block                                           |
| `inp[k]`      | `p_out[k] × n`      | local input → state sent to neighbour `k`                |
| `out[j]`      | `m × p_in[j]`       | state received from neighbour `j` → local output         |
| `trans[k, l]` | `p_out[k] × p_in[l]`| state received from `l` → state sent to `k` (`trans[k, k] = 0`) |

The state dimensions `ρ[(i, j)]` on the tree edges are bounded by the ranks of the matrix's Hankel blocks. The construction algorithm finds minimal state dimensions.

The same spinners on an arbitrary undirected graph (cycles allowed, bounce-back operators `trans[k, k] ≠ 0` allowed) form a **GIRS representation**. A TQS matrix is a GIRS representation on a tree without bounce-back operators.

## Types

| Type | Fields | Meaning |
| ---- | ------ | ------- |
| `UndirectedGraph` | `adjacency_list` | any undirected graph on integer node labels |
| `Tree` | `graph`, `root`, `levels`, `parent`, `children`, ... | a connected acyclic `UndirectedGraph` with a root |
| `BlockPartition` | `node_ordering`, `m`, `n`, `mrange`, `nrange` | one row block and one column block per node |
| `GraphPartitionedMatrix` | `mat`, `graph`, `partition` | a dense matrix on a graph |
| `GIRSRepresentation` | `spinners`, `graph`, `partition` | spinners on any graph |
| `TQSMatrix` | `spinners`, `tree`, `partition` | spinners on a tree, no bounce-back operators |

Every inner constructor takes exactly the fields and only checks them; constructors that compute fields are outer constructors.

## Files

The `TQSMatrices` module lives in [`src/`](src/). Every public function and type has a docstring (use `?name` in the REPL).

| File | Contents |
| ---- | -------- |
| [`src/TQSMatrices.jl`](src/TQSMatrices.jl) | Module entry point: module docstring and notation, exports, and `include`s of the files below. |
| [`src/containers.jl`](src/containers.jl) | `ZeroMatrix`, `IndexedVector`, `IndexedMatrix`. |
| [`src/utils.jl`](src/utils.jl) | `relative_error`. |
| [`src/partition.jl`](src/partition.jl) | `BlockPartition`, `construct_range_vec`. |
| [`src/graphs.jl`](src/graphs.jl) | `UndirectedGraph`, `is_connected_acyclic`, the rooted `Tree` type and `depth`. |
| [`src/graph_partitioned.jl`](src/graph_partitioned.jl) | `GraphPartitionedMatrix`, `get_block`, `get_hankelblock`. |
| [`src/spinner.jl`](src/spinner.jl) | `Spinner` and the checks `GIRS_is_consistent`, `has_no_bounce_back_operators`. |
| [`src/girs.jl`](src/girs.jl) | The `GIRSRepresentation` type and its constructors from spinners. |
| [`src/tqsmatrix.jl`](src/tqsmatrix.jl) | The `TQSMatrix` type, `TQSMatrix(G, root)` and `GIRSRepresentation(T)`. |
| [`src/matvec.jl`](src/matvec.jl) | `StateGraph`, the fast mat-vec `T * x`, and conversion to dense (`Matrix(T)`, `GraphPartitionedMatrix(T)`). |
| [`src/construction.jl`](src/construction.jl) | `lowrankapprox`, `HankelFactorization`, `TransGraph`, and the construction algorithm `TQSMatrix(A, root, tol, ρ_max)`. |
| [`test/runtests.jl`](test/runtests.jl) | Unit tests for the module. |
| `experiments/first-paper/Ga4.jl`, `Gb4.jl`, `Gc5.jl`, `Gd7.jl` | Table 1 experiments: build a random TQS matrix on the example graph G_a(4), G_b(4), G_c(5) or G_d(7), densify it, rebuild a TQS representation with the construction algorithm, and report the mean/std relative error over 10 trials. |
| [`experiments/first-paper/regularbinarytree.jl`](experiments/first-paper/regularbinarytree.jl) | Sec. 3.5: TQS construction for the inverses of sparse matrices on regular binary trees of growing depth. Checks that every state dimension is 1. |
| [`experiments/first-paper/shivexample.jl`](experiments/first-paper/shivexample.jl) | A variant of the binary-tree example that has some rank-2 Hankel blocks. |

## Requirements

Julia (1.9+) with the packages `IterTools` and `Statistics`. `LinearAlgebra`, `Random` and `Test` are standard libraries.

```julia
using Pkg; Pkg.add(["IterTools", "Statistics"])
```

## Usage

Scripts locate the module relative to their own path, so they can be run from any directory:

```sh
julia test/runtests.jl
julia experiments/first-paper/Gd7.jl
```

Minimal example (from this directory):

```julia
include("src/TQSMatrices.jl")
using .TQSMatrices

# Graph 1 -- 2 -- 3, block sizes m (rows) and n (columns) per node
nodes = [1, 2, 3]
m, n  = [4, 5, 3], [4, 5, 3]
adj   = Dict(1 => [2], 2 => [1, 3], 3 => [2])
A     = GraphPartitionedMatrix(rand(12, 12), nodes, m, n, adj)

# Construction algorithm: (TQS matrix rooted at node 3, state dimensions per edge)
T, ρ = TQSMatrix(A, 3, 1e-12, Inf)

x = rand(12)
b = T * x                          # fast matrix-vector product
relative_error(A.mat, Matrix(T))   # ≈ 1e-15
```

Spinners can also be built by hand: `G = GIRSRepresentation([spinner1, spinner2, ...])` collects them on their graph, and `TQSMatrix(G, root)` turns that into a TQS matrix (it checks that the graph is a tree and that there are no bounce-back operators). See the `experiments/first-paper/G*.jl` scripts for complete examples.
