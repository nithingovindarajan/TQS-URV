###########
# Spinner #
###########

"""
	Spinner{Scalar}(id, neighbors, trans, inp, out, D)
	Spinner{Scalar}(id, trans::IndexedMatrix, inp, out, D)
	Spinner{Scalar}(id, neighbors, trans, inp, out, D, m, n, p_in, p_out)

Generators of one node `id` in a TQS / GIRS representation. Say node `i` has
input `x_i` and output `b_i`, and `h[i][j]` is the state arriving from
neighbour `j`. The spinner defines

	b_i        = D * x_i + Σ_j out[j] * h[i][j]
	h[k][i]    = inp[k] * x_i + Σ_l trans[k, l] * h[i][l]     (state sent to k)

# Fields
- `id`: node label.
- `neighbors`: labels of adjacent nodes. They must be unique and must not include `id`.
- `D` (`m × n`): diagonal block of the matrix.
- `inp[k]` (`p_out[k] × n`): maps the local input to the outgoing state towards `k`.
- `out[j]` (`m × p_in[j]`): maps the incoming state from `j` to the local output.
- `trans[k, l]` (`p_out[k] × p_in[l]`): passes the state arriving from `l` on to `k`.
  For a TQS matrix, the "bounce-back" operators `trans[k, k]` must be [`ZeroMatrix`](@ref).
- `m`, `n`: output/input block sizes of the node.
- `p_in[j]`, `p_out[j]`: incoming/outgoing state dimensions per neighbour.

# Constructors
- `(id, neighbors, trans, inp, out, D)`: the most convenient form. `trans` is a
  plain `Matrix` of matrices, and `inp`/`out` are plain vectors, all ordered
  like `neighbors`. Sizes are inferred. This is the form the experiment scripts use.
- `(id, trans::IndexedMatrix, inp, out, D)`: neighbours are taken from
  `trans.labels`, and `inp`/`out` are `IndexedVector`s.
- The full inner constructor takes every field explicitly and checks that all
  dimensions agree.
"""
struct Spinner{Scalar <: Number}
	id::Int
	neighbors::Vector{Int}
	trans::IndexedMatrix{AbstractMatrix{Scalar}}
	inp::IndexedVector{AbstractMatrix{Scalar}}
	out::IndexedVector{AbstractMatrix{Scalar}}
	D::AbstractMatrix{Scalar}
	m::Int
	n::Int
	p_in::IndexedVector{Int}
	p_out::IndexedVector{Int}

	function Spinner{Scalar}(
		id,
		neighbors,
		trans,
		inp,
		out,
		D,
		m,
		n,
		p_in,
		p_out,
	) where {Scalar <: Number}

		# spinners cannot have its own node id as neighbor
		@assert all(x -> x != id, neighbors)
		# neighbor list is unique
		@assert length(Set(neighbors)) == length(neighbors)
		# trans, inp, and out, p_in, p_out have correct labels
		@assert trans.labels == neighbors
		@assert Set(keys(inp)) == Set(neighbors)
		@assert Set(keys(out)) == Set(neighbors)
		@assert Set(keys(p_in)) == Set(neighbors)
		@assert Set(keys(p_out)) == Set(neighbors)
		# dimensions of D coincide with m and n
		@assert size(D) == (m, n)
		# operator dimension checks
		@assert all([size(out[k]) == (m, p_in[k]) for k ∈ keys(out)])
		@assert all([size(inp[k]) == (p_out[k], n) for k ∈ keys(inp)])
		@assert all([
			size(trans[k, l]) == (p_out[k], p_in[l]) for k ∈ neighbors, l ∈ neighbors
		])

		new{Scalar}(id, neighbors, trans, inp, out, D, m, n, p_in, p_out)
	end

end
Base.:size(S::Spinner) = (S.m, S.n)
IndexedVector{Spinner}(S) = Dict(s.id => s for s ∈ S)
Base.eltype(x::Spinner) = typeof(x).parameters[1]


# alternative constructor 1
function Spinner{Scalar}(
	id::Int,
	trans,
	inp,
	out,
	D,
) where {Scalar <: Number}
	return Spinner{Scalar}(
		id,
		trans.labels,
		trans,
		inp,
		out,
		D,
		Base.:size(D, 1),
		Base.:size(D, 2),
		Dict(j => Base.:size(c, 2) for (j, c) ∈ out),
		Dict(j => Base.:size(b, 1) for (j, b) ∈ inp),
	)
end

# alternative constructor 2
function Spinner{Scalar}(
	id::Int,
	neighbors::Vector{Int},
	trans,
	inp,
	out,
	D,
) where {Scalar <: Number}
	return Spinner{Scalar}(
		id,
		neighbors,
		IndexedMatrix{AbstractMatrix{Scalar}}(trans, neighbors),
		IndexedVector{AbstractMatrix{Scalar}}(inp, neighbors),
		IndexedVector{AbstractMatrix{Scalar}}(out, neighbors),
		convert(AbstractMatrix{Scalar}, D),
		size(D, 1),
		size(D, 2),
		Dict(j => size(c, 2) for (j, c) ∈ zip(neighbors, out)),
		Dict(j => size(b, 1) for (j, b) ∈ zip(neighbors, inp)),
	)
end


#########################################################################
# Check if vector of spinners is a valid (infinite) GIRS representation #
#########################################################################

"""
	GIRS_is_consistent(nodeset)

Check that a collection of spinners (an `IndexedVector{Spinner}`, keyed by id)
forms a consistent GIRS representation:

1. Neighbour relations are symmetric: `j ∈ neighbors(i)` iff `i ∈ neighbors(j)`.
2. State dimensions match on every edge: `nodeset[i].p_out[j] == nodeset[j].p_in[i]`.

Returns `true`. An `AssertionError` is thrown on the first violation.
"""
function GIRS_is_consistent(nodeset)
	for node ∈ values(nodeset)
		for neighbor ∈ node.neighbors
			# node i is a neighbor of node j iff node j is a neighbor of node i
			@assert node.id ∈ nodeset[neighbor].neighbors
			# dimension consistency
			@assert node.p_out[neighbor] == nodeset[neighbor].p_in[node.id]
		end
	end
	return true
end


###############################################
# Check if all bounce back operators are null #
###############################################

"""
	GIRS_has_no_bounce_back_operators(nodeset)

Return `true` if every spinner's bounce-back operator `trans[i, i]` is a
[`ZeroMatrix`](@ref). A state that arrives from neighbour `i` must never be sent
straight back to `i`. Together with a tree graph, this is what makes a GIRS
representation a TQS matrix.
"""
function GIRS_has_no_bounce_back_operators(nodeset)
	for node ∈ values(nodeset)
		for i ∈ node.neighbors
			if !isa(node.trans[i, i], ZeroMatrix)
				return false
			end
		end
	end
	return true
end
