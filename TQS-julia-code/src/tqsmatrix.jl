################
# TQS matrices #
################

"""
	TQSMatrix(spinners::Vector{<:Spinner})
	TQSMatrix(spinners::IndexedVector{Spinner}, node_ordering)
	TQSMatrix{Scalar}(trans, inp, out, D, node_ordering)
	TQSMatrix(A::GraphPartitionedMatrix, root, tol, ρ_max) -> (T, tree, ρ)

A Tree Quasi-Separable matrix, stored as one [`Spinner`](@ref) per node of a tree.

The constructors check that the spinners are consistent
([`GIRS_is_consistent`](@ref)), have no bounce-back operators
([`GIRS_has_no_bounce_back_operators`](@ref)), and live on a tree
([`is_a_tree`](@ref)).

- `TQSMatrix(spinners)`: use a vector of spinners. The block order of the
  matrix follows the order of the vector.
- `TQSMatrix(spinners, node_ordering)`: use spinners keyed by id. The block
  order is given by `node_ordering`.
- `TQSMatrix{Scalar}(trans, inp, out, D, node_ordering)`: use per-node
  generator dictionaries (`trans[k]`, `inp[k]`, ...). The construction algorithm uses this form.
- `TQSMatrix(A, root, tol, ρ_max)`: run the **construction algorithm** on a
  dense matrix (see the method docstring).

# Fields
`spinners`, `node_ordering`, `K` (number of nodes), `adjacency_list`, the block
sizes `m`/`n`, the ranges `mrange`/`nrange`, and the total size `M × N`.
"""
struct TQSMatrix{Scalar <: Number}
	# spinners
	spinners::IndexedVector{Spinner{Scalar}}
	# graph attributes
	node_ordering::Vector{Int}
	K::Int
	adjacency_list::IndexedVector{Vector{Int}}
	# dimensions
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	M::Int
	N::Int

	function TQSMatrix{Scalar}(spinners, node_ordering) where {Scalar <: Number}

		# correctness of input
		@assert all(s -> eltype(s) == Scalar, values(spinners))
		@assert all([spinners[k].id == k for k ∈ keys(spinners)])
		@assert length(spinners) == length(node_ordering)
		@assert Set(keys(spinners)) == Set(node_ordering)

		# checks spinners generate a valid TQS matrix
		@assert GIRS_is_consistent(spinners)
		@assert GIRS_has_no_bounce_back_operators(spinners)
		adjacency_list = Dict(s.id => s.neighbors for s ∈ values(spinners))
		@assert is_a_tree(adjacency_list)

		K = length(node_ordering)
		m = Dict(s.id => s.m for s ∈ values(spinners))
		n = Dict(s.id => s.n for s ∈ values(spinners))
		mrange = construct_range_vec(m, node_ordering)
		nrange = construct_range_vec(n, node_ordering)
		M = sum(values(m))
		N = sum(values(n))

		new{Scalar}(spinners, node_ordering, K, adjacency_list, m, n, mrange, nrange, M, N)
	end
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
Base.:size(T::TQSMatrix) = (T.M, T.N)
construct_tree(T::TQSMatrix, root) = construct_tree(T.adjacency_list, root)
