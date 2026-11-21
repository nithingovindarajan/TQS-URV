####################
# Undirected graph #
####################

"""
	UndirectedGraph(adjacency_list)

An undirected graph on integer node labels, given by its adjacency list
(`Dict{Int, Vector{Int}}`): `adjacency_list[k]` holds the neighbours of node `k`.
The graph may have cycles and need not be connected.

The constructor takes the only field and checks that it describes an undirected
graph: neighbour lists have no duplicates, there are no self-loops, every
neighbour is itself a node, and the relation is symmetric (`j ∈ adjacency_list[i]`
iff `i ∈ adjacency_list[j]`).

Two graphs are `==` when they have the same nodes and the same neighbour sets;
the order inside a neighbour list does not matter.

See also [`is_connected_acyclic`](@ref) and [`Tree`](@ref).
"""
struct UndirectedGraph
	adjacency_list::IndexedVector{Vector{Int}}
	function UndirectedGraph(adjacency_list)
		for (node, neighbors) ∈ adjacency_list
			@assert allunique(neighbors)
			@assert node ∉ neighbors  # no self-loops
			for neighbor ∈ neighbors
				@assert haskey(adjacency_list, neighbor)
				@assert node ∈ adjacency_list[neighbor]  # symmetric
			end
		end
		new(adjacency_list)
	end
end
_neighbor_sets(G::UndirectedGraph) = Dict(k => Set(v) for (k, v) ∈ G.adjacency_list)
Base.:(==)(G1::UndirectedGraph, G2::UndirectedGraph) = _neighbor_sets(G1) == _neighbor_sets(G2)
Base.hash(G::UndirectedGraph, h::UInt) = hash(_neighbor_sets(G), hash(:UndirectedGraph, h))


"""
	is_connected_acyclic(G::UndirectedGraph)

Return `true` if `G` is connected and has no cycles. Such a graph becomes a
[`Tree`](@ref) once a root is picked. Uses a depth-first search.
"""
function is_connected_acyclic(G::UndirectedGraph)
	adj_list = G.adjacency_list
	isempty(adj_list) && return true

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
				return false  # Back edge, so there is a cycle
			end
		end
	end

	# Check if all nodes are visited
	return all(values(visited))
end


########
# Tree #
########

"""
	Tree(G::UndirectedGraph, root)
	Tree(graph, root, levels, parent, children, siblings, descendants, descendants_complement)

A connected acyclic undirected graph together with a chosen root, which orders
its nodes into levels, parents and children. Both the mat-vec and the
construction algorithm use it to schedule their upsweep (leaves → root) and
downsweep (root → leaves) stages.

- `Tree(G, root)`: root the graph `G` at `root`. The levels come from a
  breadth-first search, and the other fields are derived from the parents.
- The inner constructor (second form) takes every field and only checks that
  `graph` is connected and acyclic and that the other fields describe `graph`
  rooted at `root`.

# Fields
- `graph`: the underlying [`UndirectedGraph`](@ref).
- `root`: the chosen root node.
- `levels`: `levels[1] == [root]`, and `levels[l]` holds the nodes at distance `l - 1`.
- `parent[i]`: parent of `i` (`nothing` for the root).
- `children[i]`: children of `i`.
- `siblings[i]`: the other children of `parent[i]`.
- `descendants[i]`: the subtree rooted at `i`, including `i` itself.
- `descendants_complement[i]`: all nodes outside that subtree.

The depth of the tree is [`depth(tree)`](@ref depth).
"""
struct Tree
	graph::UndirectedGraph
	root::Int
	levels::Vector{Vector{Int}}
	parent::IndexedVector{Union{Int, Nothing}}
	children::IndexedVector{Vector{Int}}
	siblings::IndexedVector{Vector{Int}}
	descendants::IndexedVector{Vector{Int}}
	descendants_complement::IndexedVector{Vector{Int}}
	function Tree(
		graph,
		root,
		levels,
		parent,
		children,
		siblings,
		descendants,
		descendants_complement,
	)
		# the underlying graph
		@assert is_connected_acyclic(graph)
		adjacency_list = graph.adjacency_list
		nodes = Set(keys(adjacency_list))
		@assert root ∈ nodes
		for d ∈ (parent, children, siblings, descendants, descendants_complement)
			@assert Set(keys(d)) == nodes
		end

		# parent and children: every edge is a parent-child edge
		@assert isnothing(parent[root])
		for k ∈ nodes
			k == root || @assert parent[k] ∈ adjacency_list[k]
			@assert Set(children[k]) == Set(c for c ∈ nodes if c != root && parent[c] == k)
			@assert Set(adjacency_list[k]) ==
					union(Set(children[k]), k == root ? Set{Int}() : Set([parent[k]]))
		end

		# levels: levels[l] holds the nodes at distance l - 1 from the root
		@assert levels[1] == [root]
		@assert all(!isempty, levels)
		@assert sort(vcat(levels...)) == sort(collect(nodes))
		for l ∈ 2:length(levels), k ∈ levels[l]
			@assert parent[k] ∈ levels[l-1]
		end

		# siblings, descendants and their complements
		for k ∈ nodes
			@assert Set(siblings[k]) ==
					(k == root ? Set{Int}() : setdiff(Set(children[parent[k]]), [k]))
			@assert Set(descendants[k]) ==
					union(Set([k]), (Set(descendants[c]) for c ∈ children[k])...)
			@assert allunique(descendants[k])
			@assert Set(descendants_complement[k]) == setdiff(nodes, descendants[k])
		end

		new(
			graph,
			root,
			levels,
			parent,
			children,
			siblings,
			descendants,
			descendants_complement,
		)
	end
end

function Tree(G::UndirectedGraph, root)
	adj_list = G.adjacency_list
	@assert haskey(adj_list, root)

	# breadth first search (stops when a level is empty; if `G` is not connected
	# the unreached nodes make the inner constructor fail)
	not_yet_visited = Dict(node => true for node ∈ keys(adj_list))
	parent = Dict{Int, Union{Int, Nothing}}()
	parent[root] = nothing
	levels = [[root]]
	not_yet_visited[root] = false
	while true
		next_level = Int[]
		for node ∈ levels[end]
			for neighbor ∈ adj_list[node]
				if not_yet_visited[neighbor]
					parent[neighbor] = node
					push!(next_level, neighbor)
					not_yet_visited[neighbor] = false
				end
			end
		end
		isempty(next_level) && break
		push!(levels, next_level)
	end
	for node ∈ keys(adj_list)  # unreached nodes get a dummy parent, see above
		not_yet_visited[node] && (parent[node] = nothing)
	end

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
		for c ∈ childrenset
			siblings[c] = filter(x -> x != c, childrenset)
		end
	end

	# determine descendants and descendants complement
	descendants = Dict{Int, Vector{Int}}(k => [k] for k ∈ keys(adj_list))
	for k ∈ keys(descendants)
		function recursively_add_children!(x)
			for l ∈ children[x]
				push!(descendants[k], l)
				recursively_add_children!(l)
			end
		end
		recursively_add_children!(k)
	end
	nodes = keys(adj_list)
	descendants_complement =
		Dict{Int, Vector{Int}}(k => collect(setdiff(nodes, descendants[k])) for k ∈ nodes)

	return Tree(
		G,
		root,
		levels,
		parent,
		children,
		siblings,
		descendants,
		descendants_complement,
	)
end

"""
	depth(tree::Tree)

Number of levels of `tree` minus one (a single node has depth 0).
"""
depth(tree::Tree) = length(tree.levels) - 1
