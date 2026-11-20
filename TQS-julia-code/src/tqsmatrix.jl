################
# TQS matrices #
################

"""
	TQSMatrix(spinners::Vector{<:Spinner})
	TQSMatrix(spinners::IndexedVector{Spinner}, node_ordering)
	TQSMatrix{Scalar}(trans, inp, out, D, node_ordering)
	TQSMatrix(A::GraphPartitionedMatrix, root, tol, ρ_max) -> (T, tree, ρ)
	TQSMatrix{Scalar}(spinners, node_ordering, adjacency_list, m, n, mrange, nrange)

A Tree Quasi-Separable matrix, stored as one [`Spinner`](@ref) per node of a tree.

The inner constructor (last form) takes every field and only checks them: the
spinners are consistent ([`GIRS_is_consistent`](@ref)), have no bounce-back
operators ([`GIRS_has_no_bounce_back_operators`](@ref)), and live on a connected
acyclic graph ([`is_connected_acyclic`](@ref)), and the cached fields agree with
the spinners. The other forms are convenience constructors that compute the
cached fields and call it.

- `TQSMatrix(spinners)`: use a vector of spinners. The block order of the
  matrix follows the order of the vector.
- `TQSMatrix(spinners, node_ordering)`: use spinners keyed by id. The block
  order is given by `node_ordering`.
- `TQSMatrix{Scalar}(trans, inp, out, D, node_ordering)`: use per-node
  generator dictionaries (`trans[k]`, `inp[k]`, ...). The construction algorithm uses this form.
- `TQSMatrix(A, root, tol, ρ_max)`: run the **construction algorithm** on a
  dense matrix (see the method docstring).

# Fields
- `spinners`: the spinner of each node, keyed by node id.
- `node_ordering`: node ids in block order.
- `adjacency_list`: the neighbours of each node (a cache of `spinners[k].neighbors`).
- `m`, `n`: block row/column size of each node (a cache of `spinners[k].m`/`.n`).
- `mrange`, `nrange`: row/column index range of each node (a cache;
  `mrange == construct_range_vec(m, node_ordering)`).

The number of nodes is `length(T.node_ordering)` and the size is `size(T)`.
"""
struct TQSMatrix{Scalar <: Number}
	# spinners
	spinners::IndexedVector{Spinner{Scalar}}
	# graph attributes
	node_ordering::Vector{Int}
	adjacency_list::IndexedVector{Vector{Int}}
	# block sizes
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	# mapping between nodes and matrix entries
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}

	function TQSMatrix{Scalar}(
		spinners,
		node_ordering,
		adjacency_list,
		m,
		n,
		mrange,
		nrange,
	) where {Scalar <: Number}

		# correctness of input
		@assert all(s -> eltype(s) == Scalar, values(spinners))
		@assert all([spinners[k].id == k for k ∈ keys(spinners)])
		@assert allunique(node_ordering)
		@assert Set(keys(spinners)) == Set(node_ordering)

		# checks spinners generate a valid TQS matrix
		@assert GIRS_is_consistent(spinners)
		@assert GIRS_has_no_bounce_back_operators(spinners)
		@assert is_connected_acyclic(adjacency_list)

		# cached fields agree with the spinners
		@assert adjacency_list == Dict(s.id => s.neighbors for s ∈ values(spinners))
		@assert m == Dict(s.id => s.m for s ∈ values(spinners))
		@assert n == Dict(s.id => s.n for s ∈ values(spinners))
		@assert mrange == construct_range_vec(m, node_ordering)
		@assert nrange == construct_range_vec(n, node_ordering)

		new{Scalar}(spinners, node_ordering, adjacency_list, m, n, mrange, nrange)
	end
end
function TQSMatrix{Scalar}(spinners, node_ordering) where {Scalar <: Number}
	adjacency_list = IndexedVector{Vector{Int}}(s.id => s.neighbors for s ∈ values(spinners))
	m = IndexedVector{Int}(s.id => s.m for s ∈ values(spinners))
	n = IndexedVector{Int}(s.id => s.n for s ∈ values(spinners))
	mrange = construct_range_vec(m, node_ordering)
	nrange = construct_range_vec(n, node_ordering)
	return TQSMatrix{Scalar}(spinners, node_ordering, adjacency_list, m, n, mrange, nrange)
end
function TQSMatrix(spinners, node_ordering)
	@assert !isempty(spinners)
	T = eltype(spinners[1])
	return TQSMatrix{T}(spinners, node_ordering)
end
function TQSMatrix(spinners)
	node_ordering = [s.id for s ∈ spinners]
	spinners = IndexedVector{Spinner}(spinners)
	return TQSMatrix(spinners, node_ordering)
end

Base.eltype(T::TQSMatrix) = typeof(T).parameters[1]
Base.:size(T::TQSMatrix) = (sum(values(T.m)), sum(values(T.n)))
Base.:size(T::TQSMatrix, d) = size(T)[d]
construct_tree(T::TQSMatrix, root) = construct_tree(T.adjacency_list, root)
