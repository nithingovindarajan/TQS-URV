##############################################
# Sparse solve via the embedding and UMFPACK #
##############################################

"""
	SparseEmbeddingLU(E::SparseEmbedding)

Sparse LU factorization of `E.Ξ` by UMFPACK (`lu` of `SparseArrays`), ordered by
METIS through UMFPACK itself (`UMFPACK_ORDERING_METIS`). UMFPACK runs with its
symmetric strategy: METIS orders the pattern of `Ξ + Ξᵀ`, the same order is used
for the rows and the columns, and diagonal pivots are preferred, but UMFPACK
pivots wherever a diagonal pivot is too small, so only `Ξ` itself must be
nonsingular. UMFPACK works in `Float64` or `ComplexF64`.

Solve with `F \\ β` (for `θ`) or [`sparsesolve`](@ref)`(F, b)` (for `x`); one
factorization serves any number of right-hand sides.

# Fields
- `embedding`: the [`SparseEmbedding`](@ref).
- `F`: the UMFPACK factorization.
"""
struct SparseEmbeddingLU{Scalar <: Number}
	embedding::SparseEmbedding{Scalar}
	F::SparseArrays.UMFPACK.UmfpackLU
end

function SparseEmbeddingLU(E::SparseEmbedding{Scalar}) where {Scalar}
	UT = Scalar <: Complex ? ComplexF64 : Float64   # the types UMFPACK supports
	LSS = SparseArrays.LibSuiteSparse
	control = SparseArrays.UMFPACK.get_umfpack_control(UT, Int)
	control[LSS.UMFPACK_STRATEGY+1] = LSS.UMFPACK_STRATEGY_SYMMETRIC
	control[LSS.UMFPACK_ORDERING+1] = LSS.UMFPACK_ORDERING_METIS
	F = lu(SparseMatrixCSC{UT, Int}(E.Ξ); control)
	return SparseEmbeddingLU{Scalar}(E, F)
end

Base.eltype(::SparseEmbeddingLU{Scalar}) where {Scalar} = Scalar

"""
	F \\ β

Solve `Ξ θ = β` with the factorization `F` of `Ξ`.
"""
function Base.:\(F::SparseEmbeddingLU, β::AbstractVector)
	@assert length(β) == size(F.embedding, 1)
	return F.F \ Vector{promote_type(eltype(F.F), eltype(β))}(β)
end

"""
	sparsesolve(G::GIRSRepresentation, b)
	sparsesolve(T::TQSMatrix, b)
	sparsesolve(E::SparseEmbedding, b)
	sparsesolve(F::SparseEmbeddingLU, b)

Solve `T x = b` through the sparse embedding (sec. 6.2 of the TQS paper): build
the [`SparseEmbedding`](@ref) `Ξ θ = β`, factor `Ξ` with UMFPACK in a METIS order
([`SparseEmbeddingLU`](@ref)), solve for `θ`, and read off `x`. Works for any
GIRS representation with square blocks, with or without cycles; for a TQS matrix
the root and orientation play no role.

To solve for several right-hand sides, factor once with
`F = SparseEmbeddingLU(SparseEmbedding(T))` and call `sparsesolve(F, b)`.
"""
sparsesolve(G::GIRSRepresentation, b::AbstractVector) = sparsesolve(SparseEmbedding(G), b)
sparsesolve(T::TQSMatrix, b::AbstractVector) = sparsesolve(SparseEmbedding(T), b)
sparsesolve(E::SparseEmbedding, b::AbstractVector) = sparsesolve(SparseEmbeddingLU(E), b)
sparsesolve(F::SparseEmbeddingLU, b::AbstractVector) =
	extract(F.embedding, F \ lift(F.embedding, b))
