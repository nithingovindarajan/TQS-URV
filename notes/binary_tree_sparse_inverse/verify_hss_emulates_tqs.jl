# Numerical check of the HSS-emulates-TQS theorems on random trees.
#
# Part 1: random scalar TQS matrices (rho_e = 1, dense random generators) on a tree. Compares the
# TQS storage on that tree with the HSS storage for the cluster tree of the general theorem
# (S(v) -> C_v, {v}; C_v split into balanced groups of child subtrees) and for the balanced
# bisection of the post-order.
#
# Part 2: inverses of random sparse matrices whose adjacency graph is the tree. HSS with the
# peel-one-child cluster tree (S(v) -> S(c_1), R_2(v); R_j(v) -> S(c_j), R_{j+1}(v)) versus TQS on
# the adjacency graph.
using LinearAlgebra, Random, Printf
Random.seed!(7)
numrank(A) = isempty(A) ? 0 : rank(A; rtol = 1e-9)

# random scalar TQS matrix (rho = 1, m = n = 1) on a tree given by adjacency lists
function tqsdense(adj)
	K = length(adj)
	D = randn(K); Inp = Dict{Tuple{Int,Int},Float64}(); Out = copy(Inp); Tr = Dict{NTuple{3,Int},Float64}()
	for v in 1:K, a in adj[v]
		Inp[(v, a)] = randn(); Out[(v, a)] = randn()
		for b in adj[v]; b != a && (Tr[(v, b, a)] = randn()); end
	end
	T = diagm(D)
	for j in 1:K
		stack = [(p, j, Inp[(j, p)]) for p in adj[j]]
		while !isempty(stack)
			a, from, h = pop!(stack)
			T[a, j] = Out[(a, from)] * h
			for q in adj[a]; q != from && push!(stack, (q, a, Tr[(a, q, from)] * h)); end
		end
	end
	T
end

function rooted(adj, r)
	K = length(adj); ch = [Int[] for _ in 1:K]; seen = falses(K); seen[r] = true; q = [r]
	while !isempty(q)
		v = popfirst!(q)
		for a in adj[v]; seen[a] || (seen[a] = true; push!(ch[v], a); push!(q, a)); end
	end
	ch
end
sub(ch, v) = [reduce(vcat, (sub(ch, c) for c in ch[v]); init = Int[]); v]

# cluster tree: S(v) -> C_v, {v};  C_v -> balanced halves of the children;  one child -> S(c)
function clS(ch, v)
	isempty(ch[v]) && return (set = [v], kids = ())
	(set = sub(ch, v), kids = (clC(ch, ch[v]), (set = [v], kids = ())))
end
function clC(ch, cs)
	length(cs) == 1 && return clS(ch, cs[1])
	h = length(cs) ÷ 2
	a, b = clC(ch, cs[1:h]), clC(ch, cs[h+1:end])
	(set = [a.set; b.set], kids = (a, b))
end
# balanced bisection of a given ordering
function clB(order)
	length(order) == 1 && return (set = order, kids = ())
	h = length(order) ÷ 2
	a, b = clB(order[1:h]), clB(order[h+1:end])
	(set = order, kids = (a, b))
end
function hss(T, node, rp, cp)
	K = size(T, 1); cnt = 0; mx = 0; rs = Tuple{Int,Int}[]
	for kid in node.kids
		out = setdiff(1:K, kid.set)
		r = numrank(T[kid.set, out]); c = numrank(T[out, kid.set])
		push!(rs, (r, c)); mx = max(mx, r, c); cnt += r * rp + c * cp
		if isempty(kid.kids); cnt += length(kid.set)^2 + length(kid.set) * (r + c)
		else n, m = hss(T, kid, r, c); cnt += n; mx = max(mx, m); end
	end
	cnt + rs[1][1] * rs[2][2] + rs[2][1] * rs[1][2], mx
end

function report(name, adj)
	K = length(adj); T = tqsdense(adj); ch = rooted(adj, 1)
	tqs = sum((length(a) + 1)^2 for a in adj)
	h1, m1 = hss(T, clS(ch, 1), 0, 0)
	h2, m2 = hss(T, clB(sub(ch, 1)), 0, 0)
	@printf "%-22s K=%4d maxdeg=%3d  TQS/K=%6.1f  HSS-subtree/K=%6.1f (max rank %2d)  HSS-bisect/K=%6.1f (max rank %2d)\n" name K maximum(length.(adj)) tqs/K h1/K m1 h2/K m2
end
addedge!(adj, a, b) = (push!(adj[a], b); push!(adj[b], a))
println("Part 1: TQS matrices on a tree")
for K in (100, 200, 400)
	adj = [Int[] for _ in 1:K]; for v in 2:K; addedge!(adj, v, rand(1:v-1)); end
	report("random recursive", adj)
	adj = [Int[] for _ in 1:K]; hubs = round(Int, sqrt(K))   # hubs on a path, leaves hanging off them
	for h in 2:hubs; addedge!(adj, h, h - 1); end
	for v in hubs+1:K; addedge!(adj, v, rand(1:hubs)); end
	report("hubs of degree ~sqrtK", adj)
	adj = [Int[] for _ in 1:K]; for v in 2:K; addedge!(adj, v, 1); end
	report("star", adj)
end

# Part 2: random sparse T with the tree as adjacency graph, HSS of T^{-1} on the peel-one-child cluster tree
function sparsetree(adj)
	K = length(adj)
	T = diagm(randn(K) .+ 2 * maximum(length.(adj)))
	for v in 1:K, a in adj[v]
		T[v, a] = randn()
	end
	return T
end
function clPeel(ch, v)
	isempty(ch[v]) && return (set = [v], kids = ())
	return clR(ch, v, 1)
end
# R_j(v) = S(c_j) ∪ ... ∪ S(c_d) ∪ {v}; R_1(v) = S(v)
function clR(ch, v, j)
	j > length(ch[v]) && return (set = [v], kids = ())
	a = clPeel(ch, ch[v][j])
	b = clR(ch, v, j + 1)
	return (set = [a.set; b.set], kids = (a, b))
end
function report2(name, adj)
	K = length(adj)
	Ti = inv(sparsetree(adj))
	ch = rooted(adj, 1)
	tqs = sum((length(a) + 1)^2 for a in adj)
	h, m = hss(Ti, clPeel(ch, 1), 0, 0)
	@printf "%-22s K=%4d maxdeg=%3d  TQS/K=%6.1f  HSS-peel/K=%6.2f (max rank %d)\n" name K maximum(length.(adj)) tqs/K h/K m
end
println()
println("Part 2: inverses of sparse matrices with a tree adjacency graph")
for K in (100, 200, 400)
	adj = [Int[] for _ in 1:K]; for v in 2:K; addedge!(adj, v, rand(1:v-1)); end
	report2("random recursive", adj)
	adj = [Int[] for _ in 1:K]; hubs = round(Int, sqrt(K))
	for h in 2:hubs; addedge!(adj, h, h - 1); end
	for v in hubs+1:K; addedge!(adj, v, rand(1:hubs)); end
	report2("hubs of degree ~sqrtK", adj)
	adj = [Int[] for _ in 1:K]; for v in 2:K; addedge!(adj, v, 1); end
	report2("star (arrowhead)", adj)
end
