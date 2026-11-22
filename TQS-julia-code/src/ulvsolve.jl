#####################
# Implicit ULV solve #
#####################

# Working copy of one node's generators while the solver prunes the tree. The
# neighbours of the node in the current (pruned) tree are `keys(out)`.
mutable struct _ULVNode{Scalar <: Number}
	D::Matrix{Scalar}
	out::Dict{Int, Matrix{Scalar}}                 # out[j]: m × p_in[j]
	inp::Dict{Int, Matrix{Scalar}}                 # inp[j]: p_out[j] × n
	trans::Dict{Tuple{Int, Int}, Matrix{Scalar}}   # trans[(k, l)]: state from l on to k
	b::Vector{Scalar}                              # current right-hand side of the node
end

function _ULVNode{Scalar}(s::Spinner, b) where {Scalar <: Number}
	nb = s.neighbors
	return _ULVNode{Scalar}(
		Matrix{Scalar}(s.D),
		Dict(j => Matrix{Scalar}(s.out[j]) for j ∈ nb),
		Dict(j => Matrix{Scalar}(s.inp[j]) for j ∈ nb),
		Dict((k, l) => Matrix{Scalar}(s.trans[k, l]) for k ∈ nb, l ∈ nb if k != l),
		Vector{Scalar}(b),
	)
end

# What the back-substitution needs from the leaf operations of node `i`.
struct _ULVLeaf{Scalar <: Number}
	i::Int                 # the leaf
	p::Int                 # its parent, which absorbed it
	w::Matrix{Scalar}      # w_i' (the code stores w_i' = Z, so x_i = Z * [z_i; x̂_i])
	z::Vector{Scalar}      # z_i
	m_hat::Int             # size of x̂_i, the part of x_i stacked into the parent
end

"""
	ulvsolve(T::TQSMatrix, b::AbstractVector) -> x

Solve `T * x = b` with the implicit ULV solver of the notes
`notes/TQS_implicit_ULV_solve`, without forming `T` densely. All blocks must be
square (`m[k] == n[k]` for every node).

The solver works through the rounds of [`upsweep_rounds`](@ref)`(T.tree)`. When a
node's round comes, all its children have already been merged into it, so it is a
leaf of the current (pruned) tree. For every leaf `i` with parent `p` in the round:

1. **Leaf operations** (independent per leaf): unitary `q_i`, `w_i` with
   `q_i' * Q^i_p = [0; Q̂^i_p]` and `q_i' * D^i * w_i' = [D11 0; D21 D22]`, where
   `D11` is `k_i × k_i` lower triangular and `k_i = max(m_i - ρ_{p,i}, 0)`. Then
   `B^i_p * w_i' = [B̆ B̂]`, and `z_i = D11 \\ β_i` from `q_i' * b_i = [β_i; γ_i]`.
2. **Right-hand side**: subtract the contribution of the known `z_i` from every
   node, with one sweep up and down the current tree.
3. **Deflate and merge**: drop the solved rows and columns of `i`, and merge what
   is left (`D22`, `Q̂`, `B̂`) into the parent. The parent keeps all its other
   neighbours, so this works for both orientations: in a [`LeafOrientedTree`](@ref)
   a parent can still have unmerged children.

After the last round only the root is left; its block is solved densely. The
unknowns are then recovered in reverse order, `x_i = w_i' * [z_i; x̂_i]`.

The orientation (the type of `T.tree`) only changes how the nodes are grouped into
rounds; the solution is the same.
"""
function ulvsolve(T::TQSMatrix, b::AbstractVector)
	tree = T.tree
	partition = T.partition
	@assert size(T, 1) == length(b)
	@assert all(partition.m[k] == partition.n[k] for k ∈ partition.node_ordering)

	Scalar = promote_type(eltype(T), eltype(b))
	nodes = Dict(
		k => _ULVNode{Scalar}(T.spinners[k], b[partition.mrange[k]]) for
		k ∈ partition.node_ordering
	)
	rounds = upsweep_rounds(tree)
	leaves = _ULVLeaf{Scalar}[]   # in the order the leaves were merged

	for (r, group) ∈ enumerate(rounds)
		# 1. leaf operations, independent per leaf
		u = Dict{Int, Vector{Scalar}}()     # [z_i; 0] in the transformed coordinates
		ks = Dict{Int, Int}()
		ws = Dict{Int, Matrix{Scalar}}()
		for i ∈ group
			p = tree.parent[i]
			N = nodes[i]
			@assert collect(keys(N.out)) == [p]   # all children of i are merged
			ks[i], ws[i] = _leaf_operations!(N, p)
			u[i] = [N.D[1:ks[i], 1:ks[i]] \ N.b[1:ks[i]]; zeros(Scalar, size(N.D, 2) - ks[i])]
		end

		# 2. right-hand side: b ← b - T u over the current tree (nodes of this round
		#    and all later rounds, and the root)
		alive = vcat(rounds[r:end]..., [tree.root])
		y = _sweep(nodes, tree, alive, u)
		for k ∈ alive
			nodes[k].b .-= y[k]
		end

		# 3. deflate and merge into the parent
		for i ∈ group
			p = tree.parent[i]
			k = ks[i]
			N = nodes[i]
			z = u[i][1:k]
			D22 = N.D[k+1:end, k+1:end]
			Q̂ = N.out[p][k+1:end, :]
			B̂ = N.inp[p][:, k+1:end]
			b̂ = N.b[k+1:end]
			_merge!(nodes[p], i, D22, Q̂, B̂, b̂)
			push!(leaves, _ULVLeaf{Scalar}(i, p, ws[i], z, size(D22, 1)))
			delete!(nodes, i)
		end
	end

	# the root: one dense solve
	R = nodes[tree.root]
	@assert isempty(R.out)
	xs = Dict{Int, Vector{Scalar}}(tree.root => R.D \ R.b)

	# back-substitution, in reverse merge order: the parent's unknown stacks
	# [x̂_i; (what was there before i was merged)]
	for leaf ∈ Iterators.reverse(leaves)
		x̃p = xs[leaf.p]
		x̂ = x̃p[1:leaf.m_hat]
		xs[leaf.p] = x̃p[leaf.m_hat+1:end]
		xs[leaf.i] = leaf.w * [leaf.z; x̂]
	end

	x = Vector{Scalar}(undef, size(T, 2))
	for k ∈ partition.node_ordering
		x[partition.nrange[k]] = xs[k]
	end
	return x
end

# Leaf operations on the leaf `N` with parent `p`: transform its generators and
# right-hand side in place, and return k (number of compressed rows) and w_i' = Z.
function _leaf_operations!(N::_ULVNode{Scalar}, p) where {Scalar}
	m, n = size(N.D)
	ρ = size(N.out[p], 2)       # ρ_{p,i}: state arriving from the parent
	k = max(m - ρ, 0)
	k == 0 && return 0, Matrix{Scalar}(I, n, n)

	# left: q' * Q^i_p = [0; Q̂]. A full QR, Q^i_p = F.Q [R; 0], with the columns
	# of F.Q reordered so that the k columns orthogonal to Q^i_p come first.
	F = qr(N.out[p])
	Qfull = F.Q * Matrix{Scalar}(I, m, m)
	q = Qfull[:, [ρ+1:m; 1:ρ]]
	N.out[p] = q' * N.out[p]
	N.out[p][1:k, :] .= 0       # exactly zero (rounding)
	N.D = q' * N.D
	N.b = q' * N.b

	# right: the top k rows of q' D, times Z, are [D11 0] with D11 lower triangular.
	# A full QR of their adjoint, (q' D)[1:k, :]' = Z [R; 0], gives Z.
	G = qr(Matrix(N.D[1:k, :]'))
	Z = G.Q * Matrix{Scalar}(I, n, n)
	N.D = N.D * Z
	N.D[1:k, k+1:end] .= 0      # exactly zero (rounding)
	N.inp[p] = N.inp[p] * Z
	return k, Z
end

# Merge the deflated leaf i (D22, Q̂ = Q̂^i_p, B̂ = B̂^i_p, right-hand side b̂) into
# its parent P. The merged unknown is [x̂_i; x_P]; every other neighbour e of P
# (its parent and its unmerged children) keeps its edge and state dimension.
function _merge!(P::_ULVNode{Scalar}, i, D22, Q̂, B̂, b̂) where {Scalar}
	others = [e for e ∈ keys(P.out) if e != i]
	P.D = [D22 Q̂*P.inp[i]; P.out[i]*B̂ P.D]
	for e ∈ others
		P.out[e] = [Q̂ * P.trans[(i, e)]; P.out[e]]
		P.inp[e] = [P.trans[(e, i)] * B̂ P.inp[e]]
	end
	P.b = [b̂; P.b]
	delete!(P.out, i)
	delete!(P.inp, i)
	for key ∈ collect(keys(P.trans))
		i ∈ key && delete!(P.trans, key)
	end
	return P
end

# y = T u on the current tree (the nodes in `alive`, upsweep order, root last),
# where u is zero except at the keys of `u`. Same sweeps as the mat-vec.
function _sweep(nodes, tree, alive, u::Dict{Int, Vector{Scalar}}) where {Scalar}
	x(k) = get(u, k, nothing)
	h = Dict{Tuple{Int, Int}, Vector{Scalar}}()   # h[(j, i)]: state on edge j → i

	# upsweep: every node sends its state to its parent
	for i ∈ alive
		i == tree.root && continue
		j = tree.parent[i]
		N = nodes[i]
		s = isnothing(x(i)) ? zeros(Scalar, size(N.inp[j], 1)) : N.inp[j] * x(i)
		for w ∈ keys(N.out)
			w == j || (s += N.trans[(j, w)] * h[(w, i)])
		end
		h[(i, j)] = s
	end

	# downsweep: every node sends its state to its children
	for j ∈ Iterators.reverse(alive)
		N = nodes[j]
		for i ∈ keys(N.out)
			haskey(h, (j, i)) && continue          # i is j's parent: already sent
			s = isnothing(x(j)) ? zeros(Scalar, size(N.inp[i], 1)) : N.inp[i] * x(j)
			for l ∈ keys(N.out)
				l == i || (s += N.trans[(i, l)] * h[(l, j)])
			end
			h[(j, i)] = s
		end
	end

	# outputs
	y = Dict{Int, Vector{Scalar}}()
	for k ∈ alive
		N = nodes[k]
		yk = isnothing(x(k)) ? zeros(Scalar, size(N.D, 1)) : N.D * x(k)
		for l ∈ keys(N.out)
			yk += N.out[l] * h[(l, k)]
		end
		y[k] = yk
	end
	return y
end
