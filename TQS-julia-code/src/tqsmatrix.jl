################
# TQS matrices #
################

"""
	TQSMatrix(G::GIRSRepresentation, root)
	TQSMatrix(T::TQSMatrix, root)
	TQSMatrix(A::GraphPartitionedMatrix, root, tol, ρ_max) -> (T, ρ)
	TQSMatrix{Scalar}(spinners, tree, partition)

A Tree Quasi-Separable matrix: one [`Spinner`](@ref) per node of a
[`Tree`](@ref), without bounce-back operators. It has the same fields as a
[`GIRSRepresentation`](@ref), except that the graph is replaced by a tree, i.e.
the graph is connected and acyclic and a root is chosen. The root does not change
the matrix; it only fixes the order in which the mat-vec visits the nodes.

The inner constructor (last form) takes every field and only checks that the
spinners are consistent ([`GIRS_is_consistent`](@ref)), that `tree.graph` is the
graph of their neighbour lists, that `partition` matches their block sizes, and
that they have no bounce-back operators ([`has_no_bounce_back_operators`](@ref)).
That the graph is connected and acyclic is checked by [`Tree`](@ref).

- `TQSMatrix(G, root)`: root the graph of the GIRS representation `G` at `root`.
- `TQSMatrix(T, root)`: the same TQS matrix with a different root.
- `TQSMatrix(A, root, tol, ρ_max)`: run the **construction algorithm** on a
  dense matrix (see the method docstring).

`GIRSRepresentation(T)` forgets the root again.

# Fields
- `spinners`: the spinner of each node, keyed by node id.
- `tree`: the rooted [`Tree`](@ref) of the spinners' neighbour lists.
- `partition`: the [`BlockPartition`](@ref) given by the spinners' block sizes.
"""
struct TQSMatrix{Scalar <: Number}
	spinners::IndexedVector{Spinner{Scalar}}
	tree::Tree
	partition::BlockPartition
	function TQSMatrix{Scalar}(spinners, tree, partition) where {Scalar <: Number}
		@assert _check_spinners(Scalar, spinners, tree.graph, partition)
		@assert has_no_bounce_back_operators(spinners)
		new{Scalar}(spinners, tree, partition)
	end
end
function TQSMatrix(G::GIRSRepresentation, root::Integer)
	return TQSMatrix{eltype(G)}(G.spinners, Tree(G.graph, root), G.partition)
end
TQSMatrix(T::TQSMatrix, root::Integer) = TQSMatrix(GIRSRepresentation(T), root)
function GIRSRepresentation(T::TQSMatrix)
	return GIRSRepresentation{eltype(T)}(T.spinners, T.tree.graph, T.partition)
end

Base.eltype(T::TQSMatrix) = typeof(T).parameters[1]
Base.:size(T::TQSMatrix) = size(T.partition)
Base.:size(T::TQSMatrix, d) = size(T.partition, d)
