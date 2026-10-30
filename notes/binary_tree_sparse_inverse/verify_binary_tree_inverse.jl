# Numerical check of the example in Section 3.5 of the TQS paper: the inverse of the
# sparse matrix T_k whose adjacency graph is the level-k regular binary tree.
#
# For k = 2, 3, ... it prints
#   - the largest rank of T_k^{-1} across a single tree edge (TQS on the adjacency graph),
#   - the largest SSS and HSS Hankel-block ranks of T_k^{-1},
#   - the number of parameters of the SSS, HSS and TQS representations (dense generators),
#     divided by K = 2^k - 1;
#   - the largest SSS and HSS ranks of T_k^{-1} in post-order and in level order;
#   - the largest rank and parameters/K of HSS with a subtree-aligned cluster tree.
using LinearAlgebra
using Random
using Printf

Random.seed!(1236)

# Post-ordered level-k binary tree on the nodes 1, ..., 2^k - 1. Returns the parent array
# (parent[root] = 0) and the root.
function binarytree(k)
	parent = zeros(Int, 2^k - 1)
	function build(k, offset)                 # returns (root, last node used)
		k == 1 && return offset + 1, offset + 1
		l, last = build(k - 1, offset)
		r, last = build(k - 1, last)
		root = last + 1
		parent[l] = root
		parent[r] = root
		return root, root
	end
	root, _ = build(k, 0)
	return parent, root
end

# Random T_k: d^i on the diagonal, b^i_{P(i)} at (P(i), i) and p^{P(i)}_i at (i, P(i)).
function sparsetreematrix(k)
	parent, root = binarytree(k)
	K = length(parent)
	T = diagm(randn(K) .+ 5)
	for i in 1:K
		parent[i] == 0 && continue
		T[parent[i], i] = randn()
		T[i, parent[i]] = randn()
	end
	return T, parent
end

numrank(A) = isempty(A) ? 0 : rank(A; rtol = 1e-8)

# nodes of the subtree rooted at node c
function subtree(parent, c)
	S = [c]
	for i in length(parent):-1:1                      # post-order: descendants of i precede i
		parent[i] != 0 && parent[i] in S && push!(S, i)
	end
	return sort(S)
end

# balanced bisection of lo:hi into the HSS cluster tree (leaves of size one)
function hss_count(Ti, lo, hi, rowrank_parent, colrank_parent)
	K = size(Ti, 1)
	n = hi - lo + 1
	n == 1 && return 1
	m = lo + n ÷ 2 - 1
	count = 0
	ranks = Tuple{Int, Int}[]
	for (a, b) in ((lo, m), (m + 1, hi))
		out = [1:a-1; b+1:K]
		r = numrank(Ti[a:b, out])
		c = numrank(Ti[out, a:b])
		push!(ranks, (r, c))
		count += r * rowrank_parent + c * colrank_parent           # translation operators
		count += a == b ? r + c : 0                                 # leaf bases
		count += hss_count(Ti, a, b, r, c)
	end
	count += ranks[1][1] * ranks[2][2] + ranks[2][1] * ranks[1][2]  # sibling couplings
	return count
end

function hss_maxrank(Ti, lo, hi)
	K = size(Ti, 1)
	n = hi - lo + 1
	n == 1 && return 0
	m = lo + n ÷ 2 - 1
	maximum(max(numrank(Ti[a:b, [1:a-1; b+1:K]]), numrank(Ti[[1:a-1; b+1:K], a:b]),
		hss_maxrank(Ti, a, b)) for (a, b) in ((lo, m), (m + 1, hi)))
end

@printf "%3s %6s %8s %7s %7s %8s %8s %8s\n" "k" "K" "edgerank" "SSSmax" "HSSmax" "SSS/K" "HSS/K" "TQS/K"
for k in 2:10
	T, parent = sparsetreematrix(k)
	K = size(T, 1)
	Ti = inv(T)

	# TQS on the adjacency graph: rank of T^{-1} across each tree edge
	edgerank = 0
	for c in 1:K
		parent[c] == 0 && continue
		A = subtree(parent, c)
		Ā = setdiff(1:K, A)
		edgerank = max(edgerank, numrank(Ti[Ā, A]), numrank(Ti[A, Ā]))
	end
	L = 2^(k - 1)                                    # leaves 2x2, interior 4x4, root 3x3
	tqs = 4 * L + 16 * (L - 2) + 9

	# SSS: ranks of the lower/upper Hankel blocks at the cuts i | i+1
	l = [0; [numrank(Ti[i+1:K, 1:i]) for i in 1:K-1]; 0]
	u = [0; [numrank(Ti[1:i, i+1:K]) for i in 1:K-1]; 0]
	sss = K + sum(l[i] + l[i+1] + l[i] * l[i+1] for i in 1:K) +
		  sum(u[i] + u[i+1] + u[i] * u[i+1] for i in 1:K)

	hss = hss_count(Ti, 1, K, 0, 0)

	@printf "%3d %6d %8d %7d %7d %8.1f %8.1f %8.1f\n" k K edgerank maximum(l) hss_maxrank(Ti, 1, K) sss / K hss / K tqs / K
end

# The SSS and HSS ranks depend on the ordering of the nodes: compare the post-order above
# with the level order (root first, then level by level, left to right).
function levelorder(parent, root)
	order = [root]
	i = 1
	while i <= length(order)
		append!(order, sort(findall(==(order[i]), parent)))
		i += 1
	end
	return order
end

println()
@printf "%3s %6s %16s %16s %16s %16s\n" "k" "K" "SSSmax post" "SSSmax level" "HSSmax post" "HSSmax level"
for k in 2:8
	T, parent = sparsetreematrix(k)
	K = size(T, 1)
	Ti = inv(T)
	π = levelorder(parent, K)
	Tl = Ti[π, π]
	sssmax(A) = maximum(max(numrank(A[i+1:end, 1:i]), numrank(A[1:i, i+1:end])) for i in 1:size(A, 1)-1)
	@printf "%3d %6d %16d %16d %16d %16d\n" k K sssmax(Ti) sssmax(Tl) hss_maxrank(Ti, 1, K) hss_maxrank(Tl, 1, K)
end

# HSS with a subtree-aligned cluster tree instead of the balanced bisection:
#   S(v) -> S(l(v)), R(v)   and   R(v) := S(r(v)) ∪ {v} -> S(r(v)), {v},
# where S(v) is the subtree of v. In post-order every cluster is an interval.
function children(parent)
	ch = [Int[] for _ in parent]
	for i in eachindex(parent)
		parent[i] != 0 && push!(ch[parent[i]], i)
	end
	return ch
end

subtreenodes(ch, v) = isempty(ch[v]) ? [v] : [subtreenodes(ch, ch[v][1]); subtreenodes(ch, ch[v][2]); v]

function clusterS(ch, v)
	isempty(ch[v]) && return (set = [v], kids = ())
	l, r = ch[v]
	R = (set = [subtreenodes(ch, r); v], kids = (clusterS(ch, r), (set = [v], kids = ())))
	return (set = subtreenodes(ch, v), kids = (clusterS(ch, l), R))
end

# parameters (dense generators) and largest rank of the HSS representation on a cluster tree
function hss_tree(Ti, node, rowrank_parent, colrank_parent)
	K = size(Ti, 1)
	count, maxrank = 0, 0
	ranks = Tuple{Int, Int}[]
	for kid in node.kids
		out = setdiff(1:K, kid.set)
		r = numrank(Ti[kid.set, out])
		c = numrank(Ti[out, kid.set])
		push!(ranks, (r, c))
		maxrank = max(maxrank, r, c)
		count += r * rowrank_parent + c * colrank_parent
		if isempty(kid.kids)
			count += length(kid.set)^2 + length(kid.set) * (r + c)
		else
			n, m = hss_tree(Ti, kid, r, c)
			count += n
			maxrank = max(maxrank, m)
		end
	end
	count += ranks[1][1] * ranks[2][2] + ranks[2][1] * ranks[1][2]
	return count, maxrank
end

println()
@printf "%3s %6s %16s %16s\n" "k" "K" "HSSmax subtree" "HSS/K subtree"
for k in 2:9
	T, parent = sparsetreematrix(k)
	K = size(T, 1)
	n, m = hss_tree(inv(T), clusterS(children(parent), K), 0, 0)
	@printf "%3d %6d %16d %16.2f\n" k K m n / K
end
