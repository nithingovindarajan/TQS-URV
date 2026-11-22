####################
# Sparse embedding #
####################

"""
	SparseEmbedding(G::GIRSRepresentation)
	SparseEmbedding(T::TQSMatrix)
	SparseEmbedding{Scalar}(Ξ, partition, lifted_partition, x_index, state_index)

The sparse linear system `Ξ θ = β` that embeds the (dense) system `T x = b` of a
GIRS representation with square blocks, by treating the states as unknowns too
(eq. (6.5) of the TQS paper, same notation). `x` solves `T x = b` iff
`θ` solves `Ξ θ = β` with `β = lift(E, b)` and `x = extract(E, θ)`.

For node `j` with neighbours `i_1, …, i_p` (in the order of its spinner):

- **Column block `j`**: `θ_j = [x_j; h[(i_1, j)]; …; h[(i_p, j)]]`, the input of
  `j` and the states arriving at `j`.
- **Row block `j`**: the output equation of `j` (`m_j` rows, right-hand side
  `b_j`), then the state equations of the edges `(i_1, j), …, (i_p, j)` (right-hand
  side `0`):

	D^j x_j + Σ_i Out^j[i] h[(i, j)]                           = b_j
	h[(i, j)] - Σ_{w ∈ N(i)} Trans^i[j, w] h[(w, i)] - Inp^i[j] x_i = 0

  The sum over `w` includes `w = j` (a bounce-back operator) when there is one.

So block `(j, i)` of `Ξ` is nonzero only if `i == j` or `(i, j)` is an edge: the
block sparsity pattern of `Ξ` is the graph. This holds for any GIRS
representation, with or without cycles and bounce-back operators.

Only square blocks are supported (`m_j == n_j` for every node). Then every
diagonal block of `Ξ` is square too, as block elimination needs, and the rows of
`Ξ` are laid out like its columns: the state equation of the edge `(i, j)` sits
in the rows `state_index[(i, j)]`.

The graph is not stored: its directed edges are the keys of `state_index`.

- `SparseEmbedding(G)`, `SparseEmbedding(T)`: build the embedding.
- The inner constructor (last form) takes every field and only checks that they
  fit together: square blocks, the index ranges tile every block with `x_j`
  first, every edge also has its reverse, and `Ξ` is zero outside the diagonal
  blocks and the blocks of the edges.

# Fields
- `Ξ`: the sparse matrix.
- `partition`: the [`BlockPartition`](@ref) of `T` (blocks of `b` and `x`).
- `lifted_partition`: the [`BlockPartition`](@ref) of `Ξ` (blocks of `β` and `θ`),
  in the same node order. Block `j` has size `m_j + Σ_i ρ[(i, j)]`.
- `x_index[j]`: the indices of `x_j` in `θ` (the start of block `j`).
- `state_index[(i, j)]`: the indices of the state `h[(i, j)]` in `θ`, one entry
  per directed edge `i → j`.

See also [`lift`](@ref), [`extract`](@ref).
"""
struct SparseEmbedding{Scalar <: Number}
	Ξ::SparseMatrixCSC{Scalar, Int}
	partition::BlockPartition
	lifted_partition::BlockPartition
	x_index::IndexedVector{UnitRange{Int}}
	state_index::Dict{Tuple{Int, Int}, UnitRange{Int}}
	function SparseEmbedding{Scalar}(
		Ξ,
		partition,
		lifted_partition,
		x_index,
		state_index,
	) where {Scalar <: Number}
		P, L = partition, lifted_partition
		nodes = P.node_ordering
		edges = keys(state_index)

		# square blocks, both partitions on the same nodes in the same order
		@assert all(P.m[j] == P.n[j] for j ∈ nodes)
		@assert L.node_ordering == nodes
		@assert all(L.m[j] == L.n[j] for j ∈ nodes)
		@assert size(Ξ) == size(L)
		@assert Set(keys(x_index)) == Set(nodes)

		# edges: between two different nodes, and in both directions
		@assert all(i != j && i ∈ nodes && j ∈ nodes && (j, i) ∈ edges for (i, j) ∈ edges)

		# θ_j = [x_j; incoming states]: the ranges tile block j, with x_j first
		for j ∈ nodes
			incoming = sort([state_index[e] for e ∈ edges if e[2] == j]; by = first)
			ranges = [x_index[j]; incoming]
			@assert length(x_index[j]) == P.n[j]
			@assert first(ranges[1]) == first(L.nrange[j])
			@assert all(first(ranges[t+1]) == last(ranges[t]) + 1 for t ∈ 1:length(ranges)-1)
			@assert last(ranges[end]) == last(L.nrange[j])
		end

		# Ξ is zero outside the diagonal blocks and the blocks of the edges
		row_node = _index_to_node(L.mrange)
		col_node = _index_to_node(L.nrange)
		rows, cols, _ = findnz(Ξ)
		for (r, c) ∈ zip(rows, cols)
			j, i = row_node[r], col_node[c]
			@assert i == j || (i, j) ∈ edges
		end

		new{Scalar}(Ξ, partition, lifted_partition, x_index, state_index)
	end
end

# node of every index, from the ranges of a partition
function _index_to_node(ranges)
	node = Vector{Int}(undef, sum(length, values(ranges); init = 0))
	for (k, r) ∈ ranges
		node[r] .= k
	end
	return node
end

function SparseEmbedding(G::GIRSRepresentation)
	Scalar = eltype(G)
	S, P = G.spinners, G.partition
	nodes = P.node_ordering
	@assert all(P.m[j] == P.n[j] for j ∈ nodes)   # square blocks only

	# lifted block sizes: append the incoming states to every node
	ρ_in(j) = sum(S[j].p_in[i] for i ∈ S[j].neighbors; init = 0)
	lifted_size = Dict(j => P.n[j] + ρ_in(j) for j ∈ nodes)
	L = BlockPartition(nodes, lifted_size, lifted_size)

	# where x_j and the states h[(i, j)] sit in θ. With square blocks the rows of Ξ
	# are laid out the same way: the output rows of j are x_index[j], and the
	# state equation of the edge (i, j) is in the rows state_index[(i, j)].
	x_index = Dict{Int, UnitRange{Int}}()
	state_index = Dict{Tuple{Int, Int}, UnitRange{Int}}()
	for j ∈ nodes
		c = first(L.nrange[j])
		x_index[j] = c:c+P.n[j]-1
		c += P.n[j]
		for i ∈ S[j].neighbors
			ρ = S[j].p_in[i]
			state_index[(i, j)] = c:c+ρ-1
			c += ρ
		end
	end

	# assemble Ξ from its dense blocks, in coordinate form
	I_, J_, V_ = Int[], Int[], Scalar[]
	function add_block!(rows, cols, block, sign = 1)
		for (b, c) ∈ enumerate(cols), (a, r) ∈ enumerate(rows)
			v = block[a, b]
			iszero(v) && continue
			push!(I_, r)
			push!(J_, c)
			push!(V_, sign * v)
		end
	end
	for j ∈ nodes
		s = S[j]
		output_rows = x_index[j]

		# output equation: D^j x_j + Σ_i Out^j[i] h[(i, j)] = b_j
		add_block!(output_rows, x_index[j], s.D)
		for i ∈ s.neighbors
			add_block!(output_rows, state_index[(i, j)], s.out[i])
		end

		# state equations of the incoming edges (i, j), computed at node i:
		# h[(i, j)] - Σ_w Trans^i[j, w] h[(w, i)] - Inp^i[j] x_i = 0
		for i ∈ s.neighbors
			rows = state_index[(i, j)]
			add_block!(rows, state_index[(i, j)], Matrix{Scalar}(I, length(rows), length(rows)))
			add_block!(rows, x_index[i], S[i].inp[j], -1)
			for w ∈ S[i].neighbors
				add_block!(rows, state_index[(w, i)], S[i].trans[j, w], -1)
			end
		end
	end
	Ξ = sparse(I_, J_, V_, size(L)...)

	return SparseEmbedding{Scalar}(Ξ, P, L, x_index, state_index)
end
SparseEmbedding(T::TQSMatrix) = SparseEmbedding(GIRSRepresentation(T))

Base.eltype(::SparseEmbedding{Scalar}) where {Scalar} = Scalar
Base.:size(E::SparseEmbedding) = size(E.Ξ)
Base.:size(E::SparseEmbedding, d) = size(E.Ξ, d)

"""
	lift(E::SparseEmbedding, b) -> β

The right-hand side of `Ξ θ = β`: `b_j` in the output rows of every node, zero in
the state rows.
"""
function lift(E::SparseEmbedding, b::AbstractVector)
	P = E.partition
	@assert length(b) == size(P, 1)
	β = zeros(promote_type(eltype(E), eltype(b)), size(E, 1))
	for j ∈ P.node_ordering
		β[E.x_index[j]] = b[P.mrange[j]]   # output rows of j (square blocks)
	end
	return β
end

"""
	extract(E::SparseEmbedding, θ) -> x

The solution `x` of `T x = b` from the solution `θ` of `Ξ θ = β`: the `x_j` parts.
"""
function extract(E::SparseEmbedding, θ::AbstractVector)
	P = E.partition
	@assert length(θ) == size(E.lifted_partition, 2)
	x = similar(θ, size(P, 2))
	for j ∈ P.node_ordering
		x[P.nrange[j]] = θ[E.x_index[j]]
	end
	return x
end
