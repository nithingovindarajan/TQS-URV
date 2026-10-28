#############
# utilities #
#############

"""
	relative_error(A, B)

Return the relative Frobenius-norm error `‖A - B‖ / ‖A‖`, where `A` is the
reference. Throws an `ArgumentError` if `A` and `B` differ in size.
"""
function relative_error(A::AbstractArray, B::AbstractArray)
	# Ensure matrices are of the same size
	if size(A) != size(B)
		throw(ArgumentError("Matrices A and B must have the same dimensions"))
	end

	# Compute the Frobenius norm of the difference (absolute error)
	abs_error = norm(A - B)

	# Compute the Frobenius norm of the reference matrix (for normalization)
	ref_norm = norm(A)

	# Compute the relative error
	rel_error = abs_error / ref_norm

	return rel_error
end


#############
# Utilities #
#############

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
