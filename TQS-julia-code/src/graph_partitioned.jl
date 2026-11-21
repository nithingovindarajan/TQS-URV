############################
# Graph Partitioned matrix #
############################

"""
	GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list)
	GraphPartitionedMatrix(mat, graph::UndirectedGraph, partition::BlockPartition)

A dense matrix `mat` whose rows and columns are partitioned into blocks, one per
graph node. This is the input format for the TQS construction algorithm
`TQSMatrix(A, root, tol, ρ_max)`. The graph may be any undirected graph.

# Arguments (convenience constructor)
- `mat`: the `M × N` dense matrix.
- `nodes`: node labels, in the order their blocks appear in `mat`.
- `m`, `n`: block row/column sizes, given as vectors aligned with `nodes`.
  They must sum to `M` and `N`.
- `adjacency_list`: `Dict{Int, Vector{Int}}` giving the neighbours of each node.
  It must be symmetric and contain every node.

# Fields
- `mat`: the `M × N` dense matrix.
- `graph`: the [`UndirectedGraph`](@ref) on the nodes.
- `partition`: the [`BlockPartition`](@ref) of the rows and columns of `mat`.

The inner constructor (second form) takes every field and only checks that the
graph and the partition have the same nodes and that `size(mat) == size(partition)`.

See also [`get_block`](@ref) and [`get_hankelblock`](@ref).
"""
struct GraphPartitionedMatrix{Scalar <: Number}
	mat::AbstractMatrix{Scalar}
	graph::UndirectedGraph
	partition::BlockPartition
	function GraphPartitionedMatrix{Scalar}(mat, graph, partition) where {Scalar <: Number}
		@assert Set(keys(graph.adjacency_list)) == Set(partition.node_ordering)
		@assert size(mat) == size(partition)
		new{Scalar}(mat, graph, partition)
	end
end
function GraphPartitionedMatrix{Scalar}(
	mat,
	nodes,
	m::AbstractVector,
	n::AbstractVector,
	adjacency_list,
) where {Scalar <: Number}
	graph = UndirectedGraph(adjacency_list)
	partition = BlockPartition(nodes, m, n)
	return GraphPartitionedMatrix{Scalar}(mat, graph, partition)
end
GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list) =
	GraphPartitionedMatrix{eltype(mat)}(mat, nodes, m, n, adjacency_list)
GraphPartitionedMatrix(mat, graph::UndirectedGraph, partition::BlockPartition) =
	GraphPartitionedMatrix{eltype(mat)}(mat, graph, partition)
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
	rows = vcat([A.partition.mrange[k] for k ∈ i]...)
	columns = vcat([A.partition.nrange[k] for k ∈ j]...)
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
	i = filter(x -> !(x ∈ j), A.partition.node_ordering)
	return get_block(A, i, j)
end

