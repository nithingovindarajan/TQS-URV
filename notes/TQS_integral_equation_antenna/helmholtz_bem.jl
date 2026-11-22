# Helmholtz scattering by the antenna Γ: Galerkin BEM for the single-layer equation
#     ∫_Γ G_k(x, y) σ(y) ds_y = -u^inc(x),  x ∈ Γ,   G_k(x, y) = (i/4) H_0^(1)(k |x - y|),
# with piecewise-constant σ on panels of length h = 1/P (test functions = basis functions):
#     V_ij = ∫_{Γ_i} ∫_{Γ_j} G_k(x, y) ds_y ds_x,     b_i = -∫_{Γ_i} u^inc(x) ds_x.
# Same geometry, panels, chunks and TQS tree as antenna_ie.jl. Preconditioner M = T_ε (TQS
# compression of V), applied with ulvsolve. GMRES mat-vec: dense V * v here (stand-in for an FMM).
#
# Quadrature (V is complex symmetric, only j ≥ i is computed):
#   self / touching panels: G_k = -log|x-y|/(2π) + R_k(|x-y|) with R_k smooth; the log part is
#     integrated exactly over Γ_j, the outer integral and the R_k part by NQ-point Gauss;
#   other panels with midpoint distance < 4h: NQ × NQ Gauss; far panels: 3 × 3 Gauss.
#
# Needs SpecialFunctions (besselh), taken from the default environment. Run from the repo root:
#   julia -t 8 --project=TQS-julia-code notes/TQS_integral_equation_antenna/helmholtz_bem.jl
include(joinpath(@__DIR__, "antenna_ie.jl"))   # antenna, SEGMENTS, gmres, tqsfloats, TQSMatrices
using SpecialFunctions

const EULERγ = Base.MathConstants.eulergamma
const DIR = 0.8 - 0.6im     # direction d of the incident plane wave (as a complex number)
const NQ = 8

"Gauss–Legendre nodes and weights on [0, 1] (Golub–Welsch)."
function gauss01(n)
	β = [k / sqrt(4k^2 - 1) for k in 1:n-1]
	E = eigen(SymTridiagonal(zeros(n), β))
	return (E.values .+ 1) ./ 2, E.vectors[1, :] .^ 2
end
const GQ = gauss01(NQ)
const G3 = gauss01(3)

"Panels in the node order of antenna(P, p): start points a and unit tangents t (length h = 1/P)."
function panels(P)
	h = 1 / P; a = ComplexF64[]; t = ComplexF64[]
	for (s0, s1, _) in SEGMENTS
		len = abs(s1 - s0); dir = (s1 - s0) / len
		for c in 1:round(Int, len * P)
			push!(a, s0 + (c - 1) * h * dir); push!(t, dir)
		end
	end
	return a, t, h
end

Gk(k, r) = 0.25im * besselh(0, 1, k * r)
"R_k(r) = G_k(r) + log(r)/(2π), smooth (C¹) at r = 0."
Rk(k, r) = r < 1e-10 ? 0.25im - (log(k / 2) + EULERγ) / (2π) : Gk(k, r) + log(r) / (2π)

"∫_0^h log|x - (a + s t)| ds, exact."
function loginner(x, a, t, h)
	w = (x - a) * conj(t); s0 = real(w); d = abs(imag(w))
	F(u) = d < 1e-14 ? (u == 0 ? 0.0 : u * log(abs(u)) - u) : u * log(hypot(u, d)) - u + d * atan(u / d)
	return F(h - s0) - F(-s0)
end

function entry(k, ai, ti, aj, tj, h)
	mi = ai + h / 2 * ti; mj = aj + h / 2 * tj
	touching = min(abs(ai - aj), abs(ai - aj - h * tj), abs(ai + h * ti - aj), abs(ai + h * ti - aj - h * tj)) < 1e-12
	(xs, ws) = GQ
	if touching
		v = 0.0im
		for (sx, wx) in zip(xs, ws)
			x = ai + sx * h * ti
			v += wx * h * (-loginner(x, aj, tj, h) / (2π))
			for (sy, wy) in zip(xs, ws)
				v += wx * wy * h^2 * Rk(k, abs(x - aj - sy * h * tj))
			end
		end
		return v
	end
	abs(mi - mj) ≥ 4h && ((xs, ws) = G3)
	v = 0.0im
	for (sx, wx) in zip(xs, ws), (sy, wy) in zip(xs, ws)
		v += wx * wy * Gk(k, abs(ai + sx * h * ti - aj - sy * h * tj))
	end
	return v * h^2
end

"Galerkin matrix V (complex symmetric) and right-hand side b for wavenumber k."
function bem(P, k)
	a, t, h = panels(P); N = length(a)
	V = zeros(ComplexF64, N, N)
	Threads.@threads for i in 1:N
		for j in i:N
			V[i, j] = entry(k, a[i], t[i], a[j], t[j], h)
		end
	end
	for i in 1:N, j in 1:i-1
		V[i, j] = V[j, i]
	end
	uinc(x) = cis(k * real(conj(DIR) * x))
	b = [-sum(w * h * uinc(a[i] + s * h * t[i]) for (s, w) in zip(GQ...)) for i in 1:N]
	return V, b, a, t, h
end

"Scattered field u^s(x) = ∫_Γ G_k(x, y) σ(y) ds_y (3-point Gauss per panel)."
scattered(x, k, σ, a, t, h) = sum(σ[j] * h * sum(w * Gk(k, abs(x - a[j] - s * h * t[j])) for (s, w) in zip(G3...)) for j in eachindex(σ))

# ---------------------------------------------------------------- experiments
const KS = (1.0, 40.0)   # large wavelength (2π ≈ 6.3) and small wavelength (2π/40 ≈ 0.157)

# compression at fixed N for each k: max/mean ρ, error, memory ratio
function compressionk(P, p, tols)
	_, _, nodes, adj, bs = antenna(P, p)
	for k in KS
		V, = bem(P, k); N = size(V, 1)
		G = GraphPartitionedMatrix(V, nodes, bs, bs, adj)
		@printf("\ncompression, k = %g, N = %d, p = %d, #nodes = %d\n", k, N, p, length(nodes))
		@printf("  %8s %6s %7s %12s %12s\n", "ε", "max ρ", "mean ρ", "‖V-T‖F/‖V‖F", "TQS/dense")
		for ε in tols
			T, ρ = TQSMatrix(G, 1, ε, Inf)
			@printf("  %8.0e %6d %7.2f %12.1e %12.4f\n", ε, maximum(values(ρ)), sum(values(ρ)) / length(ρ),
				norm(V - Matrix(T)) / norm(V), tqsfloats(T) / N^2)
		end
	end
end

# GMRES iterations with and without the TQS preconditioner, for each k and N
function gmresk(Ps, p, tols)
	for k in KS
		for P in Ps
			_, _, nodes, adj, bs = antenna(P, p)
			V, b, a, t, h = bem(P, k); N = length(b)
			G = GraphPartitionedMatrix(V, nodes, bs, bs, adj)
			@printf("\nk = %g, N = %d (%.1f panels per wavelength), cond(V) = %.2e\n", k, N, 2π / (k * h), cond(V))
			σ, it0, _ = gmres(V, b, identity)
			@printf("  %-10s %6s %10s %5s\n", "precond", "max ρ", "‖V-T‖/‖V‖", "its")
			@printf("  %-10s %6s %10s %5d\n", "none", "-", "-", it0)
			for ε in tols
				T, ρ = TQSMatrix(G, 1, ε, Inf)
				x, it, _ = gmres(V, b, v -> ulvsolve(T, v))
				@assert norm(b - V * x) / norm(b) < 1e-9
				@printf("  %-10s %6d %10.1e %5d\n", @sprintf("ε=%.0e", ε), maximum(values(ρ)), opnorm(V - Matrix(T)) / opnorm(V), it)
			end
			# sanity check of the BEM: scattered field at a fixed point
			x0 = 2.5 + 0.5im
			@printf("  u^s(%s) = %s\n", x0, scattered(x0, k, σ, a, t, h))
		end
	end
end

# residual histories (one N, each k) for the convergence plot
function historiesk(P, p, ε, file)
	_, _, nodes, adj, bs = antenna(P, p)
	cols = String[]; H = Vector{Float64}[]
	for k in KS
		V, b = bem(P, k)
		G = GraphPartitionedMatrix(V, nodes, bs, bs, adj)
		T, _ = TQSMatrix(G, 1, ε, Inf)
		push!(cols, @sprintf("none_k%d", k)); push!(H, gmres(V, b, identity)[3])
		push!(cols, @sprintf("tqs_k%d", k)); push!(H, gmres(V, b, v -> ulvsolve(T, v))[3])
	end
	L = maximum(length.(H))
	open(file, "w") do io
		println(io, "it ", join(cols, " "))
		for i in 1:L
			println(io, i - 1, " ", join([i ≤ length(h) ? @sprintf("%.3e", h[i]) : "nan" for h in H], " "))
		end
	end
	println("\nwrote ", file)
end

# |u^inc + u^s| on a grid, written as a PPM image (colour map CMAP on [0, UMAX], values clipped),
# for \addplot graphics in the notes. Convert with ImageMagick: magick file.ppm file.png
const UMAX = 2.5
const CMAP = [(68, 1, 84), (59, 82, 139), (33, 145, 140), (94, 201, 98), (253, 231, 37)]  # viridis anchors
function colour(u)
	s = clamp(u / UMAX, 0, 1) * (length(CMAP) - 1); i = min(floor(Int, s), length(CMAP) - 2); f = s - i
	return [round(UInt8, (1 - f) * CMAP[i+1][c] + f * CMAP[i+2][c]) for c in 1:3]
end
function fieldk(P, k, file; nx = 301, ny = 301)
	V, b, a, t, h = bem(P, k)
	σ = V \ b
	xs = range(-2.5, 2.5; length = nx); ys = range(-0.75, 4.25; length = ny)
	U = zeros(ny, nx)
	Threads.@threads for i in 1:ny
		for j in 1:nx
			x = xs[j] + ys[i] * im
			U[i, j] = abs(cis(k * real(conj(DIR) * x)) + scattered(x, k, σ, a, t, h))
		end
	end
	open(file, "w") do io
		write(io, "P6\n$nx $ny\n255\n")
		for i in ny:-1:1, j in 1:nx      # top row first
			write(io, colour(U[i, j]))
		end
	end
	println("wrote ", file)
end

if abspath(PROGRAM_FILE) == @__FILE__
	gmresk((64, 128, 256), 8, (1e-2, 1e-4, 1e-8))
	compressionk(256, 8, (1e-2, 1e-4, 1e-6, 1e-8, 1e-10))
	historiesk(128, 8, 1e-4, joinpath(@__DIR__, "helmholtz_gmres_history.dat"))
	for k in KS
		fieldk(128, k, joinpath(@__DIR__, @sprintf("helmholtz_field_k%d.ppm", k)))
	end
end
