############################
# Graph Partitioned matrix #
############################

"""
	GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list)

A dense matrix `mat` whose rows and columns are partitioned into blocks, one per
graph node. This is the input format for the TQS construction algorithm
`TQSMatrix(A, root, tol, ρ_max)`.

# Arguments
- `mat`: the `M × N` dense matrix.
- `nodes`: node labels, in the order their blocks appear in `mat`.
- `m`, `n`: block row/column sizes, given as vectors aligned with `nodes`.
  They must sum to `M` and `N`.
- `adjacency_list`: `Dict{Int, Vector{Int}}` giving the neighbours of each node.
  It must be symmetric and contain every node.

# Fields
In addition to the inputs, it stores `K` (number of nodes), the per-node sizes
`m`/`n` as `IndexedVector`s, and the index ranges `mrange`/`nrange`.

See also [`get_block`](@ref) and [`get_hankelblock`](@ref).
"""
struct GraphPartitionedMatrix{Scalar <: Number}
	mat::AbstractMatrix{Scalar}
	# graph attributes
	nodes::Vector{Int}
	K::Int
	adjacency_list::IndexedVector{Vector{Int}}
	# mapping between nodes and matrix entries
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	# dimensions
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	M::Int
	N::Int
	function GraphPartitionedMatrix{Scalar}(
		mat,
		nodes,
		m,
		n,
		adjacency_list,
	) where {Scalar <: Number}
		@assert length(m) == length(n) == length(nodes)
		@assert sum(values(m)) == size(mat, 1)
		@assert sum(values(n)) == size(mat, 2)
		@assert length(Set(nodes)) == length(nodes)
		@assert all([haskey(adjacency_list, node) for node ∈ nodes])
		@assert length(keys(adjacency_list)) == length(nodes)
		for neighbors ∈ values(adjacency_list)
			for n ∈ neighbors
				@assert haskey(adjacency_list, n)
			end
		end

		K = length(nodes)
		M = size(mat, 1)
		N = size(mat, 2)

		# m and n
		m = IndexedVector{Int}(m, nodes)
		n = IndexedVector{Int}(n, nodes)

		# construct ranges
		mrange = construct_range_vec(m, nodes)
		nrange = construct_range_vec(n, nodes)

		# construct 
		new{Scalar}(mat, nodes, K, adjacency_list, mrange, nrange, m, n, M, N)
	end
end
GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list) =
	GraphPartitionedMatrix{eltype(mat)}(mat, nodes, m, n, adjacency_list)
Base.:size(A::GraphPartitionedMatrix) = (A.M, A.N)
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
