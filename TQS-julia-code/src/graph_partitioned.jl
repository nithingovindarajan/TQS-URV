############################
# Graph Partitioned matrix #
############################

"""
	GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list)
	GraphPartitionedMatrix{Scalar}(mat, nodes, adjacency_list, m, n, mrange, nrange)

A dense matrix `mat` whose rows and columns are partitioned into blocks, one per
graph node. This is the input format for the TQS construction algorithm
`TQSMatrix(A, root, tol, ρ_max)`.

# Arguments (convenience constructor)
- `mat`: the `M × N` dense matrix.
- `nodes`: node labels, in the order their blocks appear in `mat`.
- `m`, `n`: block row/column sizes, given as vectors aligned with `nodes`.
  They must sum to `M` and `N`.
- `adjacency_list`: `Dict{Int, Vector{Int}}` giving the neighbours of each node.
  It must be symmetric and contain every node.

# Fields
- `mat`: the `M × N` dense matrix.
- `nodes`: node labels, in block order.
- `adjacency_list`: the neighbours of each node.
- `m`, `n`: block row/column size of each node (`IndexedVector{Int}`).
- `mrange`, `nrange`: row/column index range of each node in `mat` (a cache;
  `mrange == construct_range_vec(m, nodes)`).

The inner constructor takes every field and only checks that they are
consistent. The number of nodes is `length(A.nodes)` and the size is `size(A)`.

See also [`get_block`](@ref) and [`get_hankelblock`](@ref).
"""
struct GraphPartitionedMatrix{Scalar <: Number}
	mat::AbstractMatrix{Scalar}
	# graph attributes
	nodes::Vector{Int}
	adjacency_list::IndexedVector{Vector{Int}}
	# block sizes
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	# mapping between nodes and matrix entries
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	function GraphPartitionedMatrix{Scalar}(
		mat,
		nodes,
		adjacency_list,
		m,
		n,
		mrange,
		nrange,
	) where {Scalar <: Number}
		# graph
		@assert allunique(nodes)
		@assert Set(keys(adjacency_list)) == Set(nodes)
		for (node, neighbors) ∈ adjacency_list
			for neighbor ∈ neighbors
				@assert haskey(adjacency_list, neighbor)
				@assert node ∈ adjacency_list[neighbor]  # symmetric
			end
		end
		# block sizes
		@assert Set(keys(m)) == Set(keys(n)) == Set(nodes)
		@assert sum(values(m)) == size(mat, 1)
		@assert sum(values(n)) == size(mat, 2)
		# ranges
		@assert mrange == construct_range_vec(m, nodes)
		@assert nrange == construct_range_vec(n, nodes)

		new{Scalar}(mat, nodes, adjacency_list, m, n, mrange, nrange)
	end
end
function GraphPartitionedMatrix{Scalar}(
	mat,
	nodes,
	m::AbstractVector,
	n::AbstractVector,
	adjacency_list,
) where {Scalar <: Number}
	@assert length(m) == length(n) == length(nodes)
	m = IndexedVector{Int}(m, nodes)
	n = IndexedVector{Int}(n, nodes)
	mrange = construct_range_vec(m, nodes)
	nrange = construct_range_vec(n, nodes)
	return GraphPartitionedMatrix{Scalar}(mat, nodes, adjacency_list, m, n, mrange, nrange)
end
GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list) =
	GraphPartitionedMatrix{eltype(mat)}(mat, nodes, m, n, adjacency_list)
Base.:size(A::GraphPartitionedMatrix) = size(A.mat)
Base.:size(A::GraphPartitionedMatrix, d) = size(A.mat, d)
Base.:getindex(A::GraphPartitionedMatrix, i::Int, j::Int) = A.mat[i, j]
Base.eltype(A::GraphPartitionedMatrix) = typeof(A).parameters[1]
"""
	get_block(A::GraphPartitionedMatrix, i, j)
	get_block(A::GraphPartitionedMatrix, i)

Extract the submatrix with block rows `i` and block columns `j`. Both may be a
single node or a collection of nodes, and the blocks are stacked in the order
given. With one argument, return the diagonal block `A[i, i]`.
"""
function get_block(A::GraphPartitionedMatrix, i, j)
	rows = vcat([A.mrange[k] for k ∈ i]...)
	columns = vcat([A.nrange[k] for k ∈ j]...)
	return A.mat[rows, columns]
end
get_block(A::GraphPartitionedMatrix, i) = get_block(A, i, i)

"""
	get_hankelblock(A::GraphPartitionedMatrix, j)

Return the Hankel block for the node set `j`: block columns `j` and block rows
for every node *not* in `j`. In a TQS matrix, the rank of the Hankel block of a
subtree bounds the state dimension on the edge that leaves that subtree.
"""
function get_hankelblock(A::GraphPartitionedMatrix, j)
	i = filter(x -> !(x ∈ j), A.nodes)
	return get_block(A, i, j)
end


construct_tree(T::GraphPartitionedMatrix, root) = construct_tree(T.adjacency_list, root)
