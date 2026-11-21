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
