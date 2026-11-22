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

See also [`is_connected_acyclic`](@ref), [`RootOrientedTree`](@ref) and
[`LeafOrientedTree`](@ref).
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
rooted tree ([`RootOrientedTree`](@ref), [`LeafOrientedTree`](@ref)) once a root
is picked. Uses a depth-first search.
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


#################################
# Onion peeling and the center #
#################################

"""
	peel_layers(G::UndirectedGraph) -> Vector{Vector{Int}}

Onion peeling of the connected acyclic graph `G`: repeatedly remove *all* current
leaves (nodes with at most one remaining neighbour) at once. `layers[k + 1]`
holds the nodes removed in round `k + 1`, the *k-leaves* of `G`. No root is
needed. The last layer is the [`center`](@ref) of `G`: a single node, or two
adjacent nodes. Each layer is sorted.
"""
function peel_layers(G::UndirectedGraph)
	@assert is_connected_acyclic(G)
	adj_list = G.adjacency_list
	@assert !isempty(adj_list)

	degree = Dict(k => length(v) for (k, v) ∈ adj_list)  # degree in the remaining graph
	removed = Set{Int}()
	layers = Vector{Vector{Int}}()
	layer = sort([k for (k, d) ∈ degree if d <= 1])
	while true
		push!(layers, layer)
		union!(removed, layer)
		length(removed) == length(adj_list) && break
		next_layer = Int[]
		for k ∈ layer, neighbor ∈ adj_list[k]
			neighbor ∈ removed && continue
			degree[neighbor] -= 1
			degree[neighbor] == 1 && push!(next_layer, neighbor)
		end
		layer = sort(next_layer)
	end
	return layers
end

"""
	center(G::UndirectedGraph) -> Vector{Int}

The center of the connected acyclic graph `G`: the last layer of
[`peel_layers`](@ref), either one node or two adjacent nodes (sorted). Rooting
`G` at a center node gives the smallest possible depth (the radius of `G`).
"""
center(G::UndirectedGraph) = peel_layers(G)[end]


################
# Rooted trees #
################

"""
	AbstractRootedTree

A connected acyclic [`UndirectedGraph`](@ref) together with a chosen root, which
fixes the parent, children, siblings and descendants of every node. Subtypes
differ only in how they group the nodes into *rounds*, the sets of nodes that the
upsweep (leaves → root) and downsweep (root → leaves) may process in parallel:

- [`RootOrientedTree`](@ref): rounds are levels, counted down from the root
  (depth). Any root is allowed.
- [`LeafOrientedTree`](@ref): rounds are leaf levels, counted up from the leaves
  (height). The root must be a [`center`](@ref) node, so that the leaf levels are
  the onion-peeling layers.

Every subtype has the fields `graph`, `root`, `parent`, `children`, `siblings`,
`descendants` and `descendants_complement`:
- `graph`: the underlying [`UndirectedGraph`](@ref).
- `root`: the chosen root node.
- `parent[i]`: parent of `i` (`nothing` for the root).
- `children[i]`: children of `i`.
- `siblings[i]`: the other children of `parent[i]`.
- `descendants[i]`: the subtree rooted at `i`, including `i` itself.
- `descendants_complement[i]`: all nodes outside that subtree.

The algorithms only use these fields and [`upsweep_rounds`](@ref), so they work
for every subtype, and the result does not depend on the subtype or the root.
"""
abstract type AbstractRootedTree end

"""
	upsweep_rounds(tree::AbstractRootedTree) -> Vector{Vector{Int}}

The rounds of the upsweep, in order, without the root. Every node comes after its
children. The downsweep uses the same rounds in reverse order, so every node comes
after its parent.
"""
function upsweep_rounds end

"""
	depth(tree::AbstractRootedTree)

Number of upsweep rounds: the length of the longest path from the root to a leaf
(a single node has depth 0). Equal for both orientations of the same rooted tree.
"""
depth(tree::AbstractRootedTree) = length(upsweep_rounds(tree))

# Parent, children, siblings and descendants of `G` rooted at `root`, and the
# levels from a breadth-first search (`levels[l]` = nodes at distance `l - 1`).
function _rooted_structure(G::UndirectedGraph, root)
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

	return (; levels, parent, children, siblings, descendants, descendants_complement)
end

# Check that `graph` is connected and acyclic and that the other arguments describe
# `graph` rooted at `root`. Shared by the inner constructors of the rooted trees.
function _check_rooted_structure(
	graph,
	root,
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

	# siblings, descendants and their complements
	for k ∈ nodes
		@assert Set(siblings[k]) ==
				(k == root ? Set{Int}() : setdiff(Set(children[parent[k]]), [k]))
		@assert Set(descendants[k]) ==
				union(Set([k]), (Set(descendants[c]) for c ∈ children[k])...)
		@assert allunique(descendants[k])
		@assert Set(descendants_complement[k]) == setdiff(nodes, descendants[k])
	end
	return true
end


######################
# Root-oriented tree #
######################

"""
	RootOrientedTree(G::UndirectedGraph, root)
	RootOrientedTree(graph, root, levels, parent, children, siblings, descendants, descendants_complement)

A rooted tree ([`AbstractRootedTree`](@ref)) whose rounds are the *levels*,
counted down from the root: `levels[1] == [root]`, and `levels[l]` holds the nodes
at distance `l - 1` from the root. The upsweep processes the deepest level first.
Any node may be the root.

- `RootOrientedTree(G, root)`: root the graph `G` at `root`. The levels come from
  a breadth-first search, and the other fields are derived from the parents.
- The inner constructor (second form) takes every field and only checks that
  `graph` is connected and acyclic and that the other fields describe `graph`
  rooted at `root`.
"""
struct RootOrientedTree <: AbstractRootedTree
	graph::UndirectedGraph
	root::Int
	levels::Vector{Vector{Int}}
	parent::IndexedVector{Union{Int, Nothing}}
	children::IndexedVector{Vector{Int}}
	siblings::IndexedVector{Vector{Int}}
	descendants::IndexedVector{Vector{Int}}
	descendants_complement::IndexedVector{Vector{Int}}
	function RootOrientedTree(
		graph,
		root,
		levels,
		parent,
		children,
		siblings,
		descendants,
		descendants_complement,
	)
		@assert _check_rooted_structure(
			graph, root, parent, children, siblings, descendants, descendants_complement,
		)

		# levels: levels[l] holds the nodes at distance l - 1 from the root
		@assert levels[1] == [root]
		@assert all(!isempty, levels)
		@assert sort(vcat(levels...)) == sort(collect(keys(graph.adjacency_list)))
		for l ∈ 2:length(levels), k ∈ levels[l]
			@assert parent[k] ∈ levels[l-1]
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

function RootOrientedTree(G::UndirectedGraph, root)
	s = _rooted_structure(G, root)
	return RootOrientedTree(
		G,
		root,
		s.levels,
		s.parent,
		s.children,
		s.siblings,
		s.descendants,
		s.descendants_complement,
	)
end

upsweep_rounds(tree::RootOrientedTree) = tree.levels[end:-1:2]


######################
# Leaf-oriented tree #
######################

"""
	LeafOrientedTree(G::UndirectedGraph)
	LeafOrientedTree(graph, root, leaf_levels, parent, children, siblings, descendants, descendants_complement)

A rooted tree ([`AbstractRootedTree`](@ref)) whose rounds are the *leaf levels*,
counted up from the leaves: `leaf_levels[k + 1]` holds the *k-leaves*, the nodes
whose longest path down to a leaf has length `k` (their height). The 0-leaves are
the actual leaves, the 1-leaves become leaves once those are pruned, and so on.
The upsweep processes the 0-leaves first; all leaves are pruned as early as
possible.

The root must be a [`center`](@ref) node of the graph. Then the leaf levels are
the onion-peeling layers of [`peel_layers`](@ref), except when the center has two
nodes: the other center node is then the only node in the second-to-last leaf
level, just before the root.

- `LeafOrientedTree(G)`: root `G` at its center; if the center has two nodes, at
  the one with the smaller label.
- The inner constructor (second form) takes every field and only checks that
  `graph` is connected and acyclic, that the other fields describe `graph` rooted
  at `root`, that `root` is a center node, and that every node is in the leaf
  level of its height.
"""
struct LeafOrientedTree <: AbstractRootedTree
	graph::UndirectedGraph
	root::Int
	leaf_levels::Vector{Vector{Int}}
	parent::IndexedVector{Union{Int, Nothing}}
	children::IndexedVector{Vector{Int}}
	siblings::IndexedVector{Vector{Int}}
	descendants::IndexedVector{Vector{Int}}
	descendants_complement::IndexedVector{Vector{Int}}
	function LeafOrientedTree(
		graph,
		root,
		leaf_levels,
		parent,
		children,
		siblings,
		descendants,
		descendants_complement,
	)
		@assert _check_rooted_structure(
			graph, root, parent, children, siblings, descendants, descendants_complement,
		)
		@assert root ∈ center(graph)

		# leaf levels: leaf_levels[k + 1] holds the nodes of height k
		@assert all(!isempty, leaf_levels)
		@assert sort(vcat(leaf_levels...)) == sort(collect(keys(graph.adjacency_list)))
		level = Dict(k => l for (l, ks) ∈ enumerate(leaf_levels) for k ∈ ks)
		for k ∈ keys(level)
			c = children[k]
			@assert level[k] == (isempty(c) ? 1 : 1 + maximum(level[w] for w ∈ c))
		end

		new(
			graph,
			root,
			leaf_levels,
			parent,
			children,
			siblings,
			descendants,
			descendants_complement,
		)
	end
end

function LeafOrientedTree(G::UndirectedGraph)
	root = minimum(center(G))  # checks that G is connected and acyclic
	s = _rooted_structure(G, root)

	# heights, from the deepest BFS level up: every child comes before its parent
	height = Dict{Int, Int}()
	for level ∈ Iterators.reverse(s.levels), k ∈ level
		c = s.children[k]
		height[k] = isempty(c) ? 0 : 1 + maximum(height[w] for w ∈ c)
	end
	leaf_levels = [sort([k for (k, h) ∈ height if h == l]) for l ∈ 0:height[root]]

	return LeafOrientedTree(
		G,
		root,
		leaf_levels,
		s.parent,
		s.children,
		s.siblings,
		s.descendants,
		s.descendants_complement,
	)
end

upsweep_rounds(tree::LeafOrientedTree) = tree.leaf_levels[1:end-1]
