#########################################
# Check if vector of spinners is a tree #
#########################################

"""
	is_a_tree(adj_list)

Return `true` if the undirected graph given by the adjacency list
(`Dict{Int, Vector{Int}}`) is a tree, i.e. it is connected and has no cycles.
Uses a depth-first search.
"""
function is_a_tree(adj_list)
	# for an indirect graph to be a tree:
	# 1. no cycles
	# 2. fully connected

	visited = Dict(node => false for node ∈ keys(adj_list))
	start_node = first(keys(adj_list))

	# depth first search
	stack = [(start_node, -1)]
	while !isempty(stack)
		node, prev = pop!(stack)
		visited[node] = true

		for neighbor ∈ adj_list[node]
			if !visited[neighbor]
				push!(stack, (neighbor, node))
			elseif neighbor != prev
				return false  # Back edge, so it's not a tree
			end
		end
	end

	# Check if all nodes are visited
	return all(values(visited))
end


#########################
# Determine tree levels #
#########################

"""
	Tree

A rooted view of a tree graph, created by [`construct_tree`](@ref). Both the
mat-vec and the construction algorithm use it to schedule their upsweep
(leaves → root) and downsweep (root → leaves) stages.

# Fields
- `adjacency_list`: the underlying undirected graph.
- `root`: the chosen root node.
- `tree_depth`: number of levels minus one.
- `levels`: `levels[1] == [root]`, and `levels[l]` holds the nodes at distance `l - 1`.
- `parent[i]`: parent of `i` (`nothing` for the root).
- `children[i]`: children of `i`.
- `siblings[i]`: the other children of `parent[i]`.
- `descendants[i]`: the subtree rooted at `i`, including `i` itself.
- `descendants_complement[i]`: all nodes outside that subtree.
"""
struct Tree
	adjacency_list::IndexedVector{Vector{Int}}
	root::Int
	tree_depth::Int
	levels::Vector{Vector{Int}}
	parent::IndexedVector{Union{Int, Nothing}}
	children::IndexedVector{Vector{Int}}
	siblings::IndexedVector{Vector{Int}}
	descendants::IndexedVector{Vector{Int}}
	descendants_complement::IndexedVector{Vector{Int}}
end

"""
	construct_tree(adj_list, root)
	construct_tree(A::GraphPartitionedMatrix, root)
	construct_tree(T::TQSMatrix, root)

Root the tree described by `adj_list` (or by the graph of `A`/`T`) at `root`,
and return a [`Tree`](@ref) with its levels (from a breadth-first search) and its
parent/children/sibling/descendant relations.

The graph is assumed to be a tree. Check this first with [`is_a_tree`](@ref).
"""
function construct_tree(adj_list, root)
	# function assumes nodeset is a tree. Run first GIRS_is_tree

	@assert haskey(adj_list, root)
	not_yet_visited = Dict(node => true for node ∈ keys(adj_list))

	#breadth first search
	parent = Dict{Int, Union{Int, Nothing}}()
	parent[root] = nothing
	levels = [[root]]
	not_yet_visited[root] = false
	current_level = levels[1]
	k = 1
	while any(values(not_yet_visited))
		next_level = []
		for node ∈ current_level
			for neighbor ∈ adj_list[node]
				if not_yet_visited[neighbor]
					parent[neighbor] = node
					push!(next_level, neighbor)
					not_yet_visited[neighbor] = false
				end
			end
		end
		push!(levels, next_level)
		k += 1
		current_level = next_level
	end
	tree_depth = k - 1


	# determine children
	children = Dict{Int, Vector{Int}}(k => [] for k ∈ keys(adj_list))
	for k ∈ keys(parent)
		if !isnothing(parent[k])
			push!(children[parent[k]], k)
		end
	end


	# determine siblings
	siblings = Dict{Int, Vector{Int}}(k => [] for k ∈ keys(adj_list))
	for childrenset ∈ values(children)
		if !isempty(childrenset)
			for c ∈ childrenset
				siblings[c] = filter(x -> x != c, childrenset)
			end
		end
	end


	# determine descendant and descendants complement
	descendants = Dict{Int, Vector{Int}}(k => [k] for k ∈ keys(adj_list))
	for k ∈ keys(descendants)
		function recursively_add_children!(x)
			if !isempty(children[x])
				for l ∈ children[x]
					push!(descendants[k], l)
					recursively_add_children!(l)
				end
			end
		end
		recursively_add_children!(k)
	end
	nodes = keys(adj_list)
	descendants_complement =
		Dict{Int, Vector{Int}}(k => collect(setdiff(nodes, descendants[k])) for k ∈ nodes)

	return Tree(
		adj_list,
		root,
		tree_depth,
		levels,
		parent,
		children,
		siblings,
		descendants,
		descendants_complement,
	)
end
