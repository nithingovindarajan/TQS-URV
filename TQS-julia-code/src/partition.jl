###################
# Block partition #
###################

"""
	construct_range_vec(node_sizes::IndexedVector{Int}, nodes)
	construct_range_vec(node_sizes::Vector{Tuple{Int, Int}})

Lay out the blocks of `nodes` one after another, in the given order, and return
an `IndexedVector` that maps each node to its contiguous index range.

The first form takes the block sizes from `node_sizes[node]`. The second form
takes an ordered list of `(node, size)` pairs.

```julia
construct_range_vec(Dict(1 => 6, 2 => 5, 4 => 4), [1, 2, 4])
# Dict(1 => 1:6, 2 => 7:11, 4 => 12:15)
```
"""
function construct_range_vec(node_sizes::IndexedVector{Int}, nodes)
	ranges = IndexedVector{UnitRange}()
	off = 0
	for node ∈ nodes
		ranges[node] = (off+1):(off+node_sizes[node])
		off = off + node_sizes[node]
	end
	return ranges
end


function construct_range_vec(node_sizes::Vector{Tuple{Int, Int}})
	ranges = IndexedVector{UnitRange}()
	off = 0
	for k ∈ eachindex(node_sizes)
		ranges[node_sizes[k][1]] = (off+1):(off+node_sizes[k][2])
		off = off + node_sizes[k][2]
	end
	return ranges
end


"""
	BlockPartition(node_ordering, m::AbstractVector, n::AbstractVector)
	BlockPartition(node_ordering, m::AbstractDict, n::AbstractDict)
	BlockPartition(node_ordering, m, n, mrange, nrange)

How the rows and columns of a matrix are split into blocks, one per graph node.
Node `k` owns the `m[k]` rows `mrange[k]` and the `n[k]` columns `nrange[k]`, and
the blocks are laid out in the order `node_ordering`.

- `BlockPartition(node_ordering, m, n)`: block sizes given either as vectors
  aligned with `node_ordering` or as `IndexedVector`s keyed by node. The ranges
  are computed.
- The inner constructor (last form) takes every field and only checks them: the
  nodes are unique, `m` and `n` are non-negative and keyed by exactly those
  nodes, and `mrange == construct_range_vec(m, node_ordering)` (same for `n`).

# Fields
- `node_ordering`: node labels, in block order.
- `m`, `n`: block row/column size of each node (`IndexedVector{Int}`).
- `mrange`, `nrange`: row/column index range of each node (a cache).

`size(P)` is the size `(M, N)` of the partitioned matrix.
"""
struct BlockPartition
	node_ordering::Vector{Int}
	m::IndexedVector{Int}
	n::IndexedVector{Int}
	mrange::IndexedVector{UnitRange}
	nrange::IndexedVector{UnitRange}
	function BlockPartition(node_ordering, m, n, mrange, nrange)
		@assert allunique(node_ordering)
		@assert Set(keys(m)) == Set(keys(n)) == Set(node_ordering)
		@assert all(≥(0), values(m)) && all(≥(0), values(n))
		@assert mrange == construct_range_vec(m, node_ordering)
		@assert nrange == construct_range_vec(n, node_ordering)
		new(node_ordering, m, n, mrange, nrange)
	end
end
function BlockPartition(node_ordering, m::AbstractDict, n::AbstractDict)
	m = IndexedVector{Int}(m)
	n = IndexedVector{Int}(n)
	mrange = construct_range_vec(m, node_ordering)
	nrange = construct_range_vec(n, node_ordering)
	return BlockPartition(node_ordering, m, n, mrange, nrange)
end
function BlockPartition(node_ordering, m::AbstractVector, n::AbstractVector)
	@assert length(m) == length(n) == length(node_ordering)
	return BlockPartition(
		node_ordering,
		IndexedVector{Int}(m, node_ordering),
		IndexedVector{Int}(n, node_ordering),
	)
end
Base.:size(P::BlockPartition) = (sum(values(P.m); init = 0), sum(values(P.n); init = 0))
Base.:size(P::BlockPartition, d) = size(P)[d]
function Base.:(==)(P1::BlockPartition, P2::BlockPartition)
	return all(getfield(P1, f) == getfield(P2, f) for f ∈ fieldnames(BlockPartition))
end
function Base.hash(P::BlockPartition, h::UInt)
	return hash((P.node_ordering, P.m, P.n), hash(:BlockPartition, h))
end
