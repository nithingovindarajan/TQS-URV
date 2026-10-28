##########################
# low rank approximation #
##########################

"""
	lowrankapprox(A, tol, ρ_max) -> (X, Y, ρ)

Truncated-SVD factorization `A ≈ X * Y` with `X = U[:, 1:ρ] * Σ[1:ρ]` and
`Y = V[:, 1:ρ]'`. The rank `ρ` is the number of singular values with
`σ_k / σ_1 ≥ tol`, capped at `ρ_max` (pass `Inf` for no cap).
"""
function lowrankapprox(A, tol, ρ_max)
	U, S, V = svd(A)
	ρ = findfirst(x -> x < tol, S / S[1])
	if isnothing(ρ)
		ρ = length(S)
	else
		ρ -= 1
	end
	if ρ > ρ_max
		ρ = ρ_max
	end
	X = U[:, 1:ρ] * Diagonal(S[1:ρ])
	Y = V[:, 1:ρ]'
	return X, Y, ρ
end


########################
# Hankel factorization #
########################

"""
	HankelFactorization(X, Y, in_nodes, in_ranges, out_nodes, out_ranges)

Low-rank factorization `H ≈ X * Y` of the Hankel block for one directed edge,
`H = A[out_nodes, in_nodes]`. The construction algorithm stores one per edge and
reuses it when it processes neighbouring edges.

- `X`: rows split by `out_ranges` (relative to `out_nodes`). Use [`Xblock`](@ref).
- `Y`: columns split by `in_ranges` (relative to `in_nodes`). Use [`Yblock`](@ref).
"""
struct HankelFactorization
	X::AbstractMatrix
	Y::AbstractMatrix
	in_nodes::Vector{Int}
	in_ranges::IndexedVector{UnitRange}
	out_nodes::Vector{Int}
	out_ranges::IndexedVector{UnitRange}
end

"""
	Xblock(H::HankelFactorization, i)

Rows of `H.X` for the output nodes `i`, stacked in the order given.
"""
Xblock(H::HankelFactorization, i) = vcat([H.X[H.out_ranges[k], :] for k ∈ i]...)
"""
	Yblock(H::HankelFactorization, j)

Columns of `H.Y` for the input nodes `j`, concatenated in the order given.
"""
Yblock(H::HankelFactorization, j) = hcat([H.Y[:, H.in_ranges[k]] for k ∈ j]...)



####################
# TQS construction #
####################
"""
	TransGraph{Scalar}
	TransGraph{Scalar}(A::GraphPartitionedMatrix)

Per-node transition operators: `trans[k]` is the `IndexedMatrix` of node `k`,
indexed by its neighbours. The constructor allocates uninitialised entries for
the construction algorithm to fill in.
"""
TransGraph{Scalar <: Number} = Dict{Int, IndexedMatrix{Scalar}}
function TransGraph{Scalar}(T::GraphPartitionedMatrix) where {Scalar <: Number}
	transgraph = Dict{Int, IndexedMatrix{AbstractMatrix{Scalar}}}(
		k => IndexedMatrix{AbstractMatrix{Scalar}}(v) for (k, v) ∈ T.adjacency_list
	)
	return transgraph
end


# alternative constructor for TQS matrix
function TQSMatrix{T}(trans, inp, out, D, node_ordering) where {T <: Number}
	spinners = Dict()
	for k ∈ node_ordering
		spinners[k] = Spinner{T}(k, trans[k], inp[k], out[k], D[k])
	end
	return TQSMatrix{T}(spinners, node_ordering)
end


"""
	TQSMatrix(A::GraphPartitionedMatrix, root, tol, ρ_max) -> (T, tree, ρ)

**Construction algorithm.** Compute a (near-)minimal TQS representation of the
dense graph-partitioned matrix `A`, using `root` as the root of the tree.

Every directed edge `i → j` gets a low-rank factorization of a matrix `F`. `F` is
built from the Hankel block of the edge, reusing factorizations already computed
for neighbouring edges. Its rank becomes the state dimension `ρ[(i, j)]`, and its
factors give the generators.

1. **Upsweep** (leaves → root): for each child `i` with parent `j`, factor the map
   from the subtree of `i` to the rest of the graph. This sets `inp_i[j]`,
   `trans_i[j, w]` (for children `w`), and `out_j[i]`.
2. **Downsweep** (root → leaves): for each parent `j` and child `i`, factor the map
   from everything outside the subtree of `i` into that subtree. This sets
   `inp_j[i]`, `trans_j[i, ·]` (from siblings and the grandparent), and `out_i[j]`.
3. Set the bounce-back operators `trans_i[s, s]` to [`ZeroMatrix`](@ref).

# Arguments
- `tol`: relative singular-value threshold passed to [`lowrankapprox`](@ref).
  Use a tiny value (e.g. `1e-12`) for an exact (minimal) representation.
- `ρ_max`: maximum state dimension per edge (`Inf` for unlimited).

# Returns
- `T`: the resulting `TQSMatrix`.
- `tree`: the rooted [`Tree`](@ref), needed for `T * x` and `Matrix(T, tree)`.
- `ρ`: `Dict{Tuple{Int,Int},Int}` with the state dimension of each directed edge `(i, j)`.
"""
function TQSMatrix(T::GraphPartitionedMatrix, root::Int, tol, ρ_max)

	# construct tree
	@assert root ∈ T.nodes
	tree = construct_tree(T, root)

	# Initialize generators
	D = Dict{Int, Matrix{eltype(T)}}(node => get_block(T, node) for node ∈ T.nodes)
	trans = TransGraph{eltype(T)}(T)
	inp = Dict{Int, Dict{Int, Matrix{eltype(T)}}}(
		k => Dict{Int, Matrix{eltype(T)}}() for k ∈ T.nodes
	)
	out = Dict{Int, Dict{Int, Matrix{eltype(T)}}}(
		k => Dict{Int, Matrix{eltype(T)}}() for k ∈ T.nodes
	)

	# generator skeleton hankel graph
	H = Dict{Tuple{Int, Int}, HankelFactorization}()
	ρ = Dict{Tuple{Int, Int}, Int}()

	# Upsweep stage: from leaves to the root
	for l ∈ length(tree.levels):-1:2
		for i ∈ tree.levels[l]

			# parent and children
			j, w = tree.parent[i], tree.children[i]

			# out and in nodes of the Hankel block H[(i,j)]
			in_nodes = [vcat([H[(w_t, i)].in_nodes for w_t ∈ w]...); i]
			in_ranges = construct_range_vec(T.n, in_nodes)
			out_nodes = tree.descendants_complement[i]
			out_ranges = construct_range_vec(T.m, out_nodes)

			#construct F
			F = get_block(T, out_nodes, i)
			F_partition_sizes = [(i, T.n[i])]
			if !isempty(w)
				F = [hcat([Xblock(H[(w_t, i)], out_nodes) for w_t ∈ w]...) F]
				prepend!(F_partition_sizes, [(w_t, ρ[(w_t, i)]) for w_t ∈ w])
			end
			Frange = construct_range_vec(F_partition_sizes)

			# compute low-rank factorization of F
			X, Z, ρ[(i, j)] = lowrankapprox(F, tol, ρ_max)

			# Set inp, trans & out, and form Y in the process
			out[j][i] = X[out_ranges[j], :]
			Y = Array{eltype(T)}(undef, ρ[(i, j)], 0)
			for w_t ∈ w
				trans[i][j, w_t] = Z[:, Frange[w_t]]
				Y = [Y trans[i][j, w_t] * H[(w_t, i)].Y]
			end
			inp[i][j] = Z[:, Frange[i]]
			Y = [Y inp[i][j]]

			# form hankel factorization of the Hankel block H[(i,j)]
			H[(i, j)] = HankelFactorization(X, Y, in_nodes, in_ranges, out_nodes, out_ranges)
		end
	end

	# Downsweep stage: from root to the leaves
	for l ∈ 2:length(tree.levels)
		for i ∈ tree.levels[l]
			# parent and children
			j = tree.parent[i]
			k = tree.parent[j]
			v = tree.siblings[i]

			# out and in nodes of the Hankel block H[(j,i)]
			in_nodes = [isnothing(k) ? [] : tree.descendants_complement[j];
				vcat([tree.descendants[v_t] for v_t ∈ v]...);
				j]
			in_ranges = construct_range_vec(T.n, in_nodes)
			out_nodes = tree.descendants[i]
			out_ranges = construct_range_vec(T.m, out_nodes)

			#construct F
			F = get_block(T, out_nodes, j)
			F_partition_sizes = [(j, T.n[j])]
			if !isempty(v)
				F = [hcat([Xblock(H[(v_t, j)], out_nodes) for v_t ∈ v]...) F]
				prepend!(F_partition_sizes, [(v_t, ρ[(v_t, j)]) for v_t ∈ v])
			end
			if !isnothing(k)
				F = [Xblock(H[(k, j)], out_nodes) F]
				prepend!(F_partition_sizes, [(k, ρ[(k, j)])])
			end
			Frange = construct_range_vec(F_partition_sizes)

			# compute low rank factorization of F
			X, Z, ρ[(j, i)] = lowrankapprox(F, tol, ρ_max)

			# Set trans,inp, out, and form Y in the process
			out[i][j] = X[out_ranges[i], :]
			if isnothing(k)
				Y = Array{eltype(T)}(undef, ρ[(j, i)], 0)
			else
				trans[j][i, k] = Z[:, Frange[k]]
				Y = trans[j][i, k] * H[(k, j)].Y
			end
			for v_t ∈ v
				trans[j][i, v_t] = Z[:, Frange[v_t]]
				Y = [Y trans[j][i, v_t] * H[(v_t, j)].Y]
			end
			inp[j][i] = Z[:, Frange[j]]
			Y = [Y inp[j][i]]

			# form hankel factorization of the Hankel block H[(j,i)]
			H[(j, i)] = HankelFactorization(X, Y, in_nodes, in_ranges, out_nodes, out_ranges)
		end
	end

	# fill diagonals of trans with zero matrices
	for (i, tr) ∈ trans
		for s ∈ tr.labels
			tr[s, s] = ZeroMatrix{eltype(T)}(ρ[(i, s)], ρ[(s, i)])
		end
	end

	# construct TQS matrix (create alternative constructor!)
	T_TQS = TQSMatrix{eltype(T)}(trans, inp, out, D, T.nodes)

	return T_TQS, tree, ρ

end
