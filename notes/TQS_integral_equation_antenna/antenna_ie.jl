# Model problem for the notes TQS_integral_equation_antenna.tex.
#
# Antenna Γ ⊂ ℂ (axis-aligned segments), kernel K(z,w) = |z - w|^α, 0 < α < 1, and the
# first-kind equation
#     ∫_Γ |z - w|^α u(w) |dw| = f(z),   z ∈ Γ
# (the functions take a shift λ, A = λI + K; the notes use λ = 0).
# Midpoint rule on panels of length h, exact self-panel integral on the diagonal.
# The TQS tree is the skeleton of the antenna: one node per chunk of p panels.
# Preconditioner: M = T_ε (TQS compression of A at relative tolerance ε, or rank cap ρ_max),
# applied with ulvsolve. Solver: right-preconditioned GMRES (full, no restart).
#
# Run from the repository root:
#   julia --project=TQS-julia-code notes/TQS_integral_equation_antenna/antenna_ie.jl
using LinearAlgebra, Printf, Random
include(joinpath(@__DIR__, "..", "..", "TQS-julia-code", "src", "TQSMatrices.jl"))
using .TQSMatrices

const α = 0.5

# ---------------------------------------------------------------- geometry
# Mast 0 → 3.5i, crossbars at heights 1, 2, 3 with half-widths 1.5, 1.0, 0.5.
# A segment is (start, stop, parent segment or 0); every segment starts where its parent ends,
# except the base of the mast, whose start is a free end.
const SEGMENTS = [
	(0.0im, 1.0im, 0),               # 1 mast, base
	(1.0im, 2.0im, 1),               # 2 mast
	(2.0im, 3.0im, 2),               # 3 mast
	(3.0im, 3.5im, 3),               # 4 mast, top
	(1.0im, -1.5 + 1.0im, 1),        # 5 left arm, y = 1
	(1.0im, 1.5 + 1.0im, 1),         # 6 right arm, y = 1
	(2.0im, -1.0 + 2.0im, 2),        # 7 left arm, y = 2
	(2.0im, 1.0 + 2.0im, 2),         # 8 right arm, y = 2
	(3.0im, -0.5 + 3.0im, 3),        # 9 left arm, y = 3
	(3.0im, 0.5 + 3.0im, 3),         # 10 right arm, y = 3
]

"""
	antenna(P, p) -> (z, h, nodes, adj, blocksizes)

Panels of length h = 1/P, midpoints z (in node order), chunks of p panels per TQS node.
Chunks along a segment form a path; the first chunk of a segment is attached to the last
chunk of its parent segment (the chunk that ends at the junction).
"""
function antenna(P, p)
	h = 1 / P
	z = ComplexF64[]
	adj = Dict{Int, Vector{Int}}()
	lastchunk = Dict{Int, Int}()
	node = 0
	for (s, (a, b, parent)) in enumerate(SEGMENTS)
		len = abs(b - a); npan = round(Int, len * P)
		@assert npan % p == 0 "segment $s: $npan panels not divisible by p = $p"
		dir = (b - a) / len
		prev = parent == 0 ? 0 : lastchunk[parent]
		for c in 1:(npan ÷ p)
			node += 1
			adj[node] = Int[]
			if prev != 0
				push!(adj[node], prev); push!(adj[prev], node)
			end
			for k in 1:p
				push!(z, a + ((c - 1) * p + k - 0.5) * h * dir)
			end
			prev = node
		end
		lastchunk[s] = prev
	end
	nodes = collect(1:node)
	return z, h, nodes, adj, fill(p, node)
end

"Nyström matrix: λ I + h |z_i - z_j|^α off the diagonal, exact self-panel integral on it."
function kernelmatrix(z, h, λ)
	N = length(z)
	A = [h * abs(z[i] - z[j])^α for i in 1:N, j in 1:N]
	A[diagind(A)] .= λ + 2 * (h / 2)^(1 + α) / (1 + α)
	return A
end

# ---------------------------------------------------------------- GMRES (right preconditioned)
"""
	gmres(A, b, Minv; tol, maxit) -> (x, iterations, residual history)

Full GMRES (Arnoldi with modified Gram-Schmidt, Givens rotations) for A M⁻¹ y = b, x = M⁻¹ y.
Stops when ‖b - A x‖ ≤ tol ‖b‖. Real or complex; A only needs `A * v`.
"""
function gmres(A, b, Minv; tol = 1e-10, maxit = 400)
	S = promote_type(eltype(A), eltype(b), Float64)
	N = length(b); β = norm(b)
	V = zeros(S, N, maxit + 1); H = zeros(S, maxit + 1, maxit)
	cs = zeros(S, maxit); sn = zeros(S, maxit); g = zeros(S, maxit + 1); g[1] = β
	V[:, 1] = b / β
	hist = [1.0]; k = 0
	for j in 1:maxit
		k = j
		w = A * Minv(V[:, j])
		for i in 1:j
			H[i, j] = dot(V[:, i], w); w -= H[i, j] * V[:, i]
		end
		H[j+1, j] = norm(w); V[:, j+1] = w / H[j+1, j]
		for i in 1:j-1
			t = conj(cs[i]) * H[i, j] + conj(sn[i]) * H[i+1, j]
			H[i+1, j] = -sn[i] * H[i, j] + cs[i] * H[i+1, j]; H[i, j] = t
		end
		r = hypot(abs(H[j, j]), abs(H[j+1, j])); cs[j] = H[j, j] / r; sn[j] = H[j+1, j] / r
		H[j, j] = r; H[j+1, j] = 0
		g[j+1] = -sn[j] * g[j]; g[j] = conj(cs[j]) * g[j]
		push!(hist, abs(g[j+1]) / β)
		hist[end] ≤ tol && break
	end
	y = UpperTriangular(H[1:k, 1:k]) \ g[1:k]
	x = Minv(V[:, 1:k] * y)
	return x, k, hist
end

# ---------------------------------------------------------------- experiments
maxrank(ρ) = maximum(values(ρ))
function run(P, p, λ; tols = [1e-1, 1e-2, 1e-4, 1e-8], root = 1, verbose = true)
	z, h, nodes, adj, bs = antenna(P, p)
	A = kernelmatrix(z, h, λ)
	N = length(z)
	G = GraphPartitionedMatrix(A, nodes, bs, bs, adj)
	f = ones(N)
	ev = eigvals(Symmetric(A))
	verbose && @printf("\nP = %d, p = %d, N = %d, #nodes = %d, λ = %g, α = %g\n", P, p, N, length(nodes), λ, α)
	verbose && @printf("  eig(A): min %.3e, max %.3e, #negative %d, cond %.2e\n",
		ev[1], ev[end], count(<(0), ev), cond(A))
	rows = Any[]
	_, it0, _ = gmres(A, f, identity)
	push!(rows, ("none", "-", "-", it0))
	for ε in tols
		T, ρ = TQSMatrix(G, root, ε, Inf)
		relerr = opnorm(A - Matrix(T)) / opnorm(A)
		x, it, _ = gmres(A, f, v -> ulvsolve(T, v))
		res = norm(f - A * x) / norm(f)
		push!(rows, (@sprintf("TQS ε=%.0e", ε), string(maxrank(ρ)), @sprintf("%.1e", relerr), it))
		@assert res < 1e-9
	end
	if verbose
		@printf("  %-14s %6s %10s %6s\n", "precond", "max ρ", "‖A-T‖/‖A‖", "its")
		for r in rows
			@printf("  %-14s %6s %10s %6d\n", r...)
		end
	end
	return rows
end

# Hankel ranks of A (no preconditioning) as N grows: max ρ at fixed ε
function rankgrowth(Ps, p, λ, ε)
	println("\nmax state dimension vs N (ε = $ε, p = $p)")
	for P in Ps
		z, h, nodes, adj, bs = antenna(P, p)
		A = kernelmatrix(z, h, λ)
		G = GraphPartitionedMatrix(A, nodes, bs, bs, adj)
		_, ρ = TQSMatrix(G, 1, ε, Inf)
		@printf("  N = %5d  #nodes = %4d  max ρ = %d  mean ρ = %.2f\n",
			length(z), length(nodes), maxrank(ρ), sum(values(ρ)) / length(ρ))
	end
end

# residual histories for the convergence plot (one N, one λ)
function histories(P, p, λ, file)
	z, h, nodes, adj, bs = antenna(P, p)
	A = kernelmatrix(z, h, λ)
	G = GraphPartitionedMatrix(A, nodes, bs, bs, adj)
	f = ones(length(z))
	H = Dict("none" => gmres(A, f, identity)[3])
	for ε in (1e-2, 1e-4, 1e-8)
		T, _ = TQSMatrix(G, 1, ε, Inf)
		H[@sprintf("tqs%.0e", ε)] = gmres(A, f, v -> ulvsolve(T, v))[3]
	end
	keys_ = ["none", "tqs1e-02", "tqs1e-04", "tqs1e-08"]
	L = maximum(length(H[k]) for k in keys_)
	open(file, "w") do io
		println(io, "it ", join(keys_, " "))
		for i in 1:L
			println(io, i - 1, " ", join([i ≤ length(H[k]) ? @sprintf("%.3e", H[k][i]) : "nan" for k in keys_], " "))
		end
	end
	println("\nwrote ", file)
end

# eigenvalues of K (λ = 0) for several N. spectrum_<N>.dat columns:
#   k, μ_k (all eigenvalues, ascending), j, ν_j = -(j-th negative eigenvalue), ν_1 ≥ ν_2 ≥ ... (nan in the last row)
function spectrum(Ps, p, dir)
	for P in Ps
		z, h, _, _, _ = antenna(P, p)
		μ = eigvals(Symmetric(kernelmatrix(z, h, 0.0)))
		N = length(z)
		ν = sort(-μ[μ .< 0]; rev = true)
		open(joinpath(dir, "spectrum_$(N).dat"), "w") do io
			println(io, "k mu j nu")
			for k in 1:N
				@printf(io, "%d %.6e %d %s\n", k, μ[k], k, k ≤ length(ν) ? @sprintf("%.6e", ν[k]) : "nan")
			end
		end
		@printf("N = %4d: #positive = %d (μ_max = %.4f), ν_1 = %.4f, ν_min = %.3e, #(ν > 3, 1, 0.5, 0.1, 0.01) = %s\n",
			N, count(>(0), μ), μ[end], ν[1], ν[end], string([count(>(λ), ν) for λ in (3, 1, 0.5, 0.1, 0.01)]))
		js = 10:N÷4; c = [log.(js) ones(length(js))] \ log.(ν[js])
		@printf("         fitted ν_j ~ j^%.2f on j = 10..%d\n", c[1], N ÷ 4)
	end
end

"""
	tqsfloats(T) -> number of stored floats of the TQS matrix T

Per node: D (m × n), inp[k] (p_out[k] × n), out[j] (m × p_in[j]) and trans[k, l]
(p_out[k] × p_in[l]) for k ≠ l; the bounce-back operators trans[k, k] are zero and not stored.
"""
function tqsfloats(T)
	total = 0
	for s in values(T.spinners)
		total += s.m * s.n
		for k in s.neighbors
			total += s.p_out[k] * s.n + s.m * s.p_in[k]
			for l in s.neighbors
				k != l && (total += s.p_out[k] * s.p_in[l])
			end
		end
	end
	return total
end

# compression of A = λI + K by TQS at one (large) N: max/mean ρ, error, memory ratio
function compression(P, p, λ, tols)
	z, h, nodes, adj, bs = antenna(P, p)
	A = kernelmatrix(z, h, λ)
	N = length(z)
	G = GraphPartitionedMatrix(A, nodes, bs, bs, adj)
	@printf("\ncompression, N = %d, p = %d, #nodes = %d, λ = %g\n", N, p, length(nodes), λ)
	@printf("  %8s %6s %7s %12s %12s %8s\n", "ε", "max ρ", "mean ρ", "‖A-T‖F/‖A‖F", "TQS/dense", "time [s]")
	for ε in tols
		t0 = time(); T, ρ = TQSMatrix(G, 1, ε, Inf); t = time() - t0
		err = norm(A - Matrix(T)) / norm(A)
		@printf("  %8.0e %6d %7.2f %12.1e %12.4f %8.1f\n", ε, maximum(values(ρ)),
			sum(values(ρ)) / length(ρ), err, tqsfloats(T) / N^2, t)
	end
end

if abspath(PROGRAM_FILE) == @__FILE__
	for P in (32, 64, 128)
		run(P, 8, 0.0)
	end
	rankgrowth((32, 64, 128, 256), 8, 0.0, 1e-8)
	spectrum((32, 64, 128), 8, @__DIR__)
	compression(512, 8, 0.0, (1e-1, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12))
	histories(128, 8, 0.0, joinpath(@__DIR__, "gmres_history.dat"))
end
