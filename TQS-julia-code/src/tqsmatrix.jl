################
# TQS matrices #
################

"""
	TQSMatrix(G::GIRSRepresentation, root)
	TQSMatrix(G::GIRSRepresentation, tree::AbstractRootedTree)
	TQSMatrix(T::TQSMatrix, root)
	TQSMatrix(T::TQSMatrix, tree::AbstractRootedTree)
	TQSMatrix(A::GraphPartitionedMatrix, root, tol, ρ_max) -> (T, ρ)
	TQSMatrix(A::GraphPartitionedMatrix, tree::AbstractRootedTree, tol, ρ_max) -> (T, ρ)
	TQSMatrix{Scalar}(spinners, tree, partition)

A Tree Quasi-Separable matrix: one [`Spinner`](@ref) per node of a rooted tree,
without bounce-back operators. It has the same fields as a
[`GIRSRepresentation`](@ref), except that the graph is replaced by a rooted tree,
i.e. the graph is connected and acyclic and a root is chosen.

The type parameter `TreeType <: AbstractRootedTree` is the orientation:
[`RootOrientedTree`](@ref) (rounds counted down from the root) or
[`LeafOrientedTree`](@ref) (rounds counted up from the leaves). The aliases
[`RootOrientedTQSMatrix`](@ref) and [`LeafOrientedTQSMatrix`](@ref) name the two
cases. Neither the root nor the orientation changes the matrix; they only fix the
order in which the algorithms visit the nodes.

The inner constructor (last form) takes every field and only checks that the
spinners are consistent ([`GIRS_is_consistent`](@ref)), that `tree.graph` is the
graph of their neighbour lists, that `partition` matches their block sizes, and
that they have no bounce-back operators ([`has_no_bounce_back_operators`](@ref)).
That the graph is connected and acyclic is checked by the tree.

- `TQSMatrix(G, root)`: root the graph of the GIRS representation `G` at `root`,
  as a [`RootOrientedTree`](@ref).
- `TQSMatrix(G, tree)`: use the given rooted tree, which must be on the graph of
  `G`. For example `TQSMatrix(G, LeafOrientedTree(G.graph))`.
- `TQSMatrix(T, root)`, `TQSMatrix(T, tree)`: the same TQS matrix with a different
  root or orientation.
- `TQSMatrix(A, root, tol, ρ_max)`, `TQSMatrix(A, tree, tol, ρ_max)`: run the
  **construction algorithm** on a dense matrix (see the method docstring).

`GIRSRepresentation(T)` forgets the root again.

# Fields
- `spinners`: the spinner of each node, keyed by node id.
- `tree`: the rooted tree ([`AbstractRootedTree`](@ref)) of the spinners'
  neighbour lists.
- `partition`: the [`BlockPartition`](@ref) given by the spinners' block sizes.
"""
struct TQSMatrix{Scalar <: Number, TreeType <: AbstractRootedTree}
	spinners::IndexedVector{Spinner{Scalar}}
	tree::TreeType
	partition::BlockPartition
	function TQSMatrix{Scalar, TreeType}(
		spinners,
		tree,
		partition,
	) where {Scalar <: Number, TreeType <: AbstractRootedTree}
		@assert _check_spinners(Scalar, spinners, tree.graph, partition)
		@assert has_no_bounce_back_operators(spinners)
		new{Scalar, TreeType}(spinners, tree, partition)
	end
end
function TQSMatrix{Scalar}(spinners, tree::AbstractRootedTree, partition) where {Scalar <: Number}
	return TQSMatrix{Scalar, typeof(tree)}(spinners, tree, partition)
end

"""
	RootOrientedTQSMatrix{Scalar}

A [`TQSMatrix`](@ref) on a [`RootOrientedTree`](@ref).
"""
const RootOrientedTQSMatrix{Scalar} = TQSMatrix{Scalar, RootOrientedTree}

"""
	LeafOrientedTQSMatrix{Scalar}

A [`TQSMatrix`](@ref) on a [`LeafOrientedTree`](@ref).
"""
const LeafOrientedTQSMatrix{Scalar} = TQSMatrix{Scalar, LeafOrientedTree}

function TQSMatrix(G::GIRSRepresentation, tree::AbstractRootedTree)
	return TQSMatrix{eltype(G)}(G.spinners, tree, G.partition)
end
TQSMatrix(G::GIRSRepresentation, root::Integer) = TQSMatrix(G, RootOrientedTree(G.graph, root))
TQSMatrix(T::TQSMatrix, root::Integer) = TQSMatrix(GIRSRepresentation(T), root)
TQSMatrix(T::TQSMatrix, tree::AbstractRootedTree) = TQSMatrix(GIRSRepresentation(T), tree)
function GIRSRepresentation(T::TQSMatrix)
	return GIRSRepresentation{eltype(T)}(T.spinners, T.tree.graph, T.partition)
end

Base.eltype(::TQSMatrix{Scalar}) where {Scalar} = Scalar
Base.:size(T::TQSMatrix) = size(T.partition)
Base.:size(T::TQSMatrix, d) = size(T.partition, d)
