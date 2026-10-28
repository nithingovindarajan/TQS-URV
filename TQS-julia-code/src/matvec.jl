##############################
# TQS matrix vector multiply #
##############################

"""
	StateGraph{Scalar}
	StateGraph{Scalar}(T::TQSMatrix)

Storage for the state vectors used during a mat-vec. `h[i][j]` is the state on
the directed edge `j → i`: it is computed at node `j` and consumed at node `i`,
and has length `T.spinners[i].p_in[j]`. The constructor allocates all states as zeros.
"""
StateGraph{Scalar <: Number} = Dict{Int, Dict{Int, Vector{Scalar}}}
function StateGraph{Scalar}(T::TQSMatrix) where {Scalar <: Number}
	# creates a Stategraph with zero entries
	stategraph = StateGraph{Scalar}()
	for (node, neighbors) ∈ T.adjacency_list
		stategraph[node] = Dict(
			neighbor => zeros(Scalar, T.spinners[node].p_in[neighbor]) for
			neighbor ∈ neighbors
		)
	end
	return stategraph
end


"""
	*(T::TQSMatrix, x::Vector, tree::Tree)
	*(T::TQSMatrix, X::AbstractMatrix, tree::Tree)

Compute `b = T * x` without forming `T` densely. `tree` must come from
`construct_tree(T, root)`, and any root works. The computation has three stages:

1. **Diagonal**: `b[i] = D_i * x[i]` for every node.
2. **Upsweep** (leaves → root): every node `i` sends its parent `j` the state
   `h[j][i] = inp_i[j] * x[i] + Σ_{w ∈ children(i)} trans_i[j, w] * h[i][w]`,
   and `b[j]` gets `out_j[i] * h[j][i]` added to it.
3. **Downsweep** (root → leaves): every parent `j` sends child `i` the state built
   from `x[j]`, the states from `i`'s siblings, and the state from `j`'s own
   parent. Then `b[i]` gets `out_i[j] * h[i][j]` added to it.

The matrix form applies the vector product to every column of `X`.
"""
function Base.:*(T::TQSMatrix, x::Vector, tree::Tree)

	#check input correctness
	@assert T.N == length(x)

	# pre-allocate memory
	type_b = promote_type(eltype(T), eltype(x))
	b = Vector{type_b}(undef, T.M)
	h = StateGraph{type_b}(T)

	# diagonal stage
	for node ∈ eachindex(T.node_ordering)
		b[T.mrange[node]] = T.spinners[node].D * x[T.nrange[node]]
	end

	# Upsweep stage: from leaves to the root
	for l ∈ length(tree.levels):-1:2
		for i ∈ tree.levels[l]
			j = tree.parent[i]
			h[j][i] += T.spinners[i].inp[j] * x[T.nrange[i]]   #input contribution
			for w ∈ tree.children[i]                             #children state contribution
				h[j][i] += T.spinners[i].trans[j, w] * h[i][w]
			end
			b[T.mrange[j]] += T.spinners[j].out[i] * h[j][i]
		end
	end

	# Downsweep stage: from root to the leaves
	for l ∈ 2:length(tree.levels)
		for i ∈ tree.levels[l]
			j = tree.parent[i]
			h[i][j] += T.spinners[j].inp[i] * x[T.nrange[j]]    # input contribution
			for s ∈ tree.siblings[i]                              # sibling contribution
				h[i][j] += T.spinners[j].trans[i, s] * h[j][s]
			end
			k = tree.parent[j]
			if !isnothing(k)
				h[i][j] += T.spinners[j].trans[i, k] * h[j][k]  # grandparent contribution
			end
			b[T.mrange[i]] += T.spinners[i].out[j] * h[i][j]
		end
	end

	return b
end

function Base.:*(T::TQSMatrix, X::AbstractMatrix, tree::Tree)
	return mapslices(x -> *(T, x, tree), X; dims = 1)
end


#######################
# TQS to dense matrix #
#######################

"""
	Matrix(T::TQSMatrix, tree::Tree)

Form the dense `M × N` matrix by multiplying `T` with the identity. This is
meant for testing and error checks only.
"""
function Base.:Matrix(T::TQSMatrix, tree::Tree)
	return *(T, Matrix{eltype(T)}(I, T.N, T.N), tree)
end


"""
	GraphPartitionedMatrix(T::TQSMatrix, tree::Tree)

Densify `T` and wrap the result in a [`GraphPartitionedMatrix`](@ref) with the
same graph and block partitioning. Useful as input to the construction algorithm.
"""
function GraphPartitionedMatrix(T::TQSMatrix, tree::Tree)
	# convert to dense matrix
	Tdense = Matrix(T, tree)

	# equip dense matrix with graph
	TG = GraphPartitionedMatrix(
		Tdense,
		T.node_ordering,
		[T.m[k] for k ∈ T.node_ordering],
		[T.n[k] for k ∈ T.node_ordering],
		T.adjacency_list,
	)
	return TG
end
