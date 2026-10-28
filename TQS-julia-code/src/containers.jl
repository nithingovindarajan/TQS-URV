##############
# ZeroMatrix #
##############

"""
	ZeroMatrix{Scalar}(m, n)
	ZeroMatrix(m, n)

Lightweight `m × n` matrix of zeros that stores only its size. It marks
generators that are structurally zero, most importantly the "bounce-back"
transition operators `trans[i, i]` of a spinner (see
[`GIRS_has_no_bounce_back_operators`](@ref)). The element type defaults to `Float64`.
"""
struct ZeroMatrix{Scalar <: Number} <: AbstractMatrix{Scalar}
	m::Int
	n::Int
end
ZeroMatrix(m, n) = ZeroMatrix{Float64}(m, n)
Base.:size(A::ZeroMatrix) = (A.m, A.n)
function Base.:getindex(A::ZeroMatrix, i::Int, j::Int)
	if 0 < i <= A.m && 0 < j <= A.n
		return convert(eltype(A), 0)
	else
		throw(BoundsError(A, (i, j)))
	end
end
Base.:Matrix(A::ZeroMatrix) = zeros(eltype(A), A.m, A.n)

#################
# IndexedVector #
#################

"""
	IndexedVector{T}
	IndexedVector{T}(array, labels)

A "vector" indexed by node labels rather than by `1:n`. It is just an alias for
`Dict{Int, T}`. The two-argument constructor pairs `labels[k] => array[k]`.
Calling `IndexedVector{T}(v)` on an existing `IndexedVector` converts its value type.

```julia
inp = IndexedVector{Matrix{Float64}}([B4, B5], [4, 5])   # inp[4] == B4
```
"""
IndexedVector{T <: Any} = Dict{Int, T}
IndexedVector{T}(array, labels) where {T <: Any} = Dict{Int, T}(zip(labels, array))
function IndexedVector{T}(v::IndexedVector) where {T <: Any}
	return Dict{Int, T}(key => val for (key, val) ∈ v)
end


#################
# IndexedMatrix #
#################

"""
	IndexedMatrix{T}(array, labels)
	IndexedMatrix{T}(labels)
	IndexedMatrix{T}(A::IndexedMatrix)

Square matrix whose rows and columns are both indexed by node labels.
`A[i, j]` looks up the positions of labels `i` and `j` in `labels` and returns
that entry of the underlying `array`. It stores a spinner's transition
operators, where `trans[k, l]` is itself a matrix.

- `IndexedMatrix{T}(array, labels)`: wrap the square `array` (converted to `Matrix{T}`).
- `IndexedMatrix{T}(labels)`: create an uninitialised matrix to fill in later.
- `IndexedMatrix{T}(A)`: recast an existing `IndexedMatrix` to element type `T`.
"""
struct IndexedMatrix{T <: Any}
	array::Matrix{T}
	labels::Vector{Int}
	index_map::Dict{Int, Int}
	N::Int
	function IndexedMatrix{T}(array, labels) where {T <: Any}
		@assert size(array, 1) == size(array, 2)
		@assert size(array, 1) == length(labels)
		new{T}(
			convert(Matrix{T}, array),
			labels,
			Dict(label => i for (i, label) ∈ enumerate(labels)),
			size(array, 1)::Int,
		)
	end
end
Base.:getindex(A::IndexedMatrix, i, j) = A.array[A.index_map[i], A.index_map[j]]
function Base.:setindex!(A::IndexedMatrix, val, i, j)
	A.array[A.index_map[i], A.index_map[j]] = val
end
function IndexedMatrix{T}(A::IndexedMatrix) where {T <: Any}     # to recast into different type parameter
	return IndexedMatrix{T}(A.array, A.labels)
end
function IndexedMatrix{T}(labels::Vector{Int}) where {T <: Any} # creates an undefined IndexedMatrix
	N = length(labels)
	return IndexedMatrix{T}(Array{T}(undef, N, N), labels)
end
Base.:size(A::IndexedMatrix) = (A.N, A.N)
Base.:size(A::IndexedMatrix, i) = size(A)[i]
