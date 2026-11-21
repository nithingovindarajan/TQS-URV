#######################
# GIRS representation #
#######################

"""
	GIRSRepresentation(spinners::Vector{<:Spinner})
	GIRSRepresentation(spinners::IndexedVector{<:Spinner}, node_ordering)
	GIRSRepresentation{Scalar}(trans, inp, out, D, node_ordering)
	GIRSRepresentation(T::TQSMatrix)
	GIRSRepresentation{Scalar}(spinners, graph, partition)

A collection of [`Spinner`](@ref)s, one per node of an arbitrary undirected
graph, that together describe a graph-partitioned matrix. The graph may have
cycles, and the spinners may have bounce-back operators. A GIRS representation on
a connected acyclic graph without bounce-back operators can be turned into a
[`TQSMatrix`](@ref) with `TQSMatrix(G, root)`.

The inner constructor (last form) takes every field and only checks that the
spinners are consistent ([`GIRS_is_consistent`](@ref)), that `graph` is the graph
of their neighbour lists, and that `partition` matches their block sizes. The
other forms compute `graph` and `partition` from the spinners:

- `GIRSRepresentation(spinners)`: a vector of spinners. The block order follows
  the order of the vector.
- `GIRSRepresentation(spinners, node_ordering)`: spinners keyed by id. The block
  order is given by `node_ordering`.
- `GIRSRepresentation{Scalar}(trans, inp, out, D, node_ordering)`: per-node
  generator dictionaries (`trans[k]`, `inp[k]`, ...). The construction algorithm
  uses this form.
- `GIRSRepresentation(T)`: forget the root of a TQS matrix.

# Fields
- `spinners`: the spinner of each node, keyed by node id.
- `graph`: the [`UndirectedGraph`](@ref) of the spinners' neighbour lists.
- `partition`: the [`BlockPartition`](@ref) given by the spinners' block sizes.
"""
struct GIRSRepresentation{Scalar <: Number}
	spinners::IndexedVector{Spinner{Scalar}}
	graph::UndirectedGraph
	partition::BlockPartition
	function GIRSRepresentation{Scalar}(spinners, graph, partition) where {Scalar <: Number}
		@assert _check_spinners(Scalar, spinners, graph, partition)
		new{Scalar}(spinners, graph, partition)
	end
end
function GIRSRepresentation{Scalar}(spinners, node_ordering) where {Scalar <: Number}
	graph = UndirectedGraph(Dict{Int, Vector{Int}}(s.id => s.neighbors for s ∈ values(spinners)))
	m = IndexedVector{Int}(s.id => s.m for s ∈ values(spinners))
	n = IndexedVector{Int}(s.id => s.n for s ∈ values(spinners))
	partition = BlockPartition(node_ordering, m, n)
	return GIRSRepresentation{Scalar}(spinners, graph, partition)
end
function GIRSRepresentation(spinners::AbstractDict, node_ordering)
	@assert !isempty(spinners)
	Scalar = eltype(first(values(spinners)))
	return GIRSRepresentation{Scalar}(spinners, node_ordering)
end
function GIRSRepresentation(spinners::AbstractVector)
	node_ordering = [s.id for s ∈ spinners]
	return GIRSRepresentation(IndexedVector{Spinner}(spinners), node_ordering)
end
function GIRSRepresentation{Scalar}(trans, inp, out, D, node_ordering) where {Scalar <: Number}
	spinners = Dict{Int, Spinner{Scalar}}(
		k => Spinner{Scalar}(k, trans[k], inp[k], out[k], D[k]) for k ∈ node_ordering
	)
	return GIRSRepresentation{Scalar}(spinners, node_ordering)
end

Base.eltype(G::GIRSRepresentation) = typeof(G).parameters[1]
Base.:size(G::GIRSRepresentation) = size(G.partition)
Base.:size(G::GIRSRepresentation, d) = size(G.partition, d)
has_no_bounce_back_operators(G::GIRSRepresentation) = has_no_bounce_back_operators(G.spinners)
