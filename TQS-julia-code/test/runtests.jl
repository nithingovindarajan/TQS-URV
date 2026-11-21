include(joinpath(@__DIR__, "..", "src", "TQSMatrices.jl"))
using .TQSMatrices
using LinearAlgebra
using Test


################################
# Test: low rank approximation #
###############################

X = rand(15, 5)
Y = rand(5, 15)
A = X * Y

tol = 1E-12
ρ_max = Inf
U, V, ρ = lowrankapprox(A, tol, ρ_max);

@test U * V ≈ A
@test ρ == 5

################################
# Test: construct range vector #
################################

node_sizes = Dict(4 => 4, 2 => 5, 1 => 6)
nodes = [1, 2, 4]
node_sizes2 = [(1, 6), (2, 5), (4, 4)]
@test construct_range_vec(node_sizes, nodes) == construct_range_vec(node_sizes2)


#######################
# Test: IndexedVector #
#######################

neighbors = [4, 5, 6]
inp = [rand(4, 2), rand(4, 2), nothing];
inp_indexed = IndexedVector{Union{Nothing, AbstractMatrix{Float64}}}(inp, neighbors)
B = rand(4, 2)
inp_indexed[6] = B[:, :]
B[1, 1] = 4
@test B[1, 1] != inp_indexed[6][1, 1]
@test typeof(IndexedVector{AbstractMatrix{Float64}}(inp_indexed)) ==
	  IndexedVector{AbstractMatrix{Float64}}

size(rand(4, 2))


#######################
# Test: IndexedMatrix #
#######################

neighbors = [4, 5, 6]
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = nothing, rand(3, 3), rand(3, 3);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(3, 3), ZeroMatrix(3, 3), rand(3, 3);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(3, 3), rand(3, 3), ZeroMatrix(3, 3);

trans_indexed = IndexedMatrix{Union{AbstractMatrix{Float64}, Nothing}}(trans, neighbors)
@test typeof(trans_indexed) == IndexedMatrix{Union{AbstractMatrix{Float64}, Nothing}}

# check if we can modify entries
B = rand(3, 3)
trans_indexed[4, 5] = B[:, :]
@test trans_indexed[4, 5] == B
B[1, 1] = 4
@test trans_indexed[4, 5][1, 1] != B[1, 1]

# fill-in unfilled entry
trans_indexed[4, 4] = rand(3, 3)
# "recast" into an indexedmatrix with type parameter AbstractMatrix{Float64} (so no union with Nothing)
@test typeof(IndexedMatrix{AbstractMatrix{Float64}}(trans_indexed)) ==
	  IndexedMatrix{AbstractMatrix{Float64}}

# create an undefined IndexedMatrix
IndexedMatrix{Matrix{Float64}}(neighbors)
IndexedMatrix{Matrix{ComplexF64}}(neighbors)


#################################################
# Test: constructing graph partitioned matrices #
#################################################

# Considered graph:
#         3 -- 1 -- 2 
#              |
#         4 -- 5 

nodes = [3, 1, 2, 4, 5]
m = [10, 12, 14, 16, 18]
n = [11, 13, 15, 17, 19]
M = sum(m)
N = sum(n)
adjacency_list = Dict{Int, Vector{Int}}()
adjacency_list[1] = [2, 3, 5]
adjacency_list[2] = [1]
adjacency_list[3] = [1]
adjacency_list[4] = [5]
adjacency_list[5] = [1, 4]
mat = randn(M, N)
tmp_m = [0; cumsum(m)]
tmp_n = [0; cumsum(n)]

A = GraphPartitionedMatrix(mat, nodes, m, n, adjacency_list)
@test eltype(A) == eltype(mat)
@test all([A.partition.m[nodes[k]] == m[k] for k in eachindex(nodes)])
@test all([A.partition.n[nodes[k]] == n[k] for k in eachindex(nodes)])
@test all([A.partition.mrange[nodes[k]] == tmp_m[k]+1:tmp_m[k+1] for k in eachindex(nodes)])
@test all([A.partition.nrange[nodes[k]] == tmp_n[k]+1:tmp_n[k+1] for k in eachindex(nodes)])
@test all([
	get_block(A, nodes[k], nodes[l]) == A.mat[tmp_m[k]+1:tmp_m[k+1], tmp_n[l]+1:tmp_n[l+1]]
	for k in eachindex(nodes), l in eachindex(nodes)
])
nodeset_in = rand(nodes)
nodeset_out = rand(nodes, 2)
@test get_block(A, nodeset_out, nodeset_in) == A.mat[
	vcat([A.partition.mrange[k] for k in nodeset_out]...),
	vcat([A.partition.nrange[k] for k in nodeset_in]...),
]
nodeset_in = rand(nodes, 2)
nodeset_out = rand(nodes)
@test get_block(A, nodeset_out, nodeset_in) == A.mat[
	vcat([A.partition.mrange[k] for k in nodeset_out]...),
	vcat([A.partition.nrange[k] for k in nodeset_in]...),
]

nodeset_in = rand(nodes, 2)
nodeset_out = rand(nodes, 2)
@test get_block(A, nodeset_out, nodeset_in) == A.mat[
	vcat([A.partition.mrange[k] for k in nodeset_out]...),
	vcat([A.partition.nrange[k] for k in nodeset_in]...),
]
nodeset_in = [2 3]
nodeset_out = [1, 4, 5]
@test get_hankelblock(A, nodeset_in) == A.mat[
	vcat([A.partition.mrange[k] for k in nodeset_out]...),
	vcat([A.partition.nrange[k] for k in nodeset_in]...),
]

# create a TransGraph
TransGraph{Float64}(A)
TransGraph{ComplexF64}(A)
####################################
# Test: construct spinner matrices #
####################################

# Considered graph:
#         3 -- 1 -- 2 
#              |
#         4 -- 6 -- 5

# input dimensions
n1 = 13
n2 = 15
n3 = 17
n4 = 19
n5 = 21
n6 = 23
# output dimensions
m1 = 12
m2 = 14
m3 = 16
m4 = 18
m5 = 20
m6 = 22
# state dimensions
p12 = 1
p21 = 2
p13 = 3
p31 = 4
p12 = 5
p21 = 6
p46 = 7
p64 = 8
p56 = 9
p65 = 10
p16 = 11
p61 = 12

id = 1;
neighbors = [2, 3, 6];
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p12, p21), rand(p12, p31), rand(p12, p61);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(p13, p21), ZeroMatrix(p13, p31), rand(p13, p61);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(p16, p21), rand(p16, p31), ZeroMatrix(p16, p61);
inp = [rand(p12, n1), rand(p13, n1), rand(p16, n1)];
out = [rand(m1, p21), rand(m1, p31), rand(m1, p61)];
D = rand(m1, n1);
node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node1.trans[2, 6] == trans[1, 3]
@test node1.inp[2] == inp[1]
@test node1.out[3] == out[2]
@test node1.D == D

# test alternative constructor
node1_alternative = Spinner{Float64}(id, node1.trans, node1.inp, node1.out, D);

id = 2;
neighbors = [1];
trans = Matrix(undef, 1, 1);
trans[1, 1] = ZeroMatrix(p21, p12)
inp = [rand(p21, n2)];
out = [rand(m2, p12)];
D = rand(m2, n2);
node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node2.trans[1, 1] == trans[1, 1]
@test node2.inp[1] == inp[1]
@test node2.out[1] == out[1]
@test node2.D == D

id = 3;
neighbors = [1];
trans = Matrix(undef, 1, 1);
trans[1, 1] = ZeroMatrix(p31, p13)
inp = [rand(p31, n3)];
out = [rand(m3, p13)];
D = rand(m3, n3);
node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node3.trans[1, 1] == trans[1, 1]
@test node3.inp[1] == inp[1]
@test node3.out[1] == out[1]
@test node3.D == D

id = 4;
neighbors = [6];
trans = Matrix(undef, 1, 1);
trans[1, 1] = ZeroMatrix(p46, p64)
inp = [rand(p46, n4)];
out = [rand(m4, p64)];
D = rand(m4, n4);
node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node4.trans[6, 6] == trans[1, 1]
@test node4.inp[6] == inp[1]
@test node4.out[6] == out[1]
@test node4.D == D

id = 5;
neighbors = [6];
trans = Matrix(undef, 1, 1);
trans[1, 1] = ZeroMatrix(p56, p65)
inp = [rand(p56, n5)];
out = [rand(m5, p65)];
D = rand(m5, n5);
node5 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node5.trans[6, 6] == trans[1, 1]
@test node5.inp[6] == inp[1]
@test node5.out[6] == out[1]
@test node5.D == D

id = 6;
neighbors = [1, 4, 5];
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p61, p16), rand(p61, p46), rand(p61, p56);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(p64, p16), ZeroMatrix(p64, p46), rand(p64, p56);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(p65, p16), rand(p65, p46), ZeroMatrix(p65, p56);
inp = [rand(p61, n6), rand(p64, n6), rand(p65, n6)];
out = [rand(m6, p16), rand(m6, p46), rand(m6, p56)];
D = rand(m6, n6);
node6 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node6.trans[5, 4] == trans[3, 2]
@test node6.inp[5] == inp[3]
@test node6.out[4] == out[2]
@test node6.D == D

nodeset_acyclic = IndexedVector{Spinner}([node1, node2, node3, node4, node5, node6])



# A non tree graph:
#              1 -- 2 -- 5
#              |    |
#              4 -- 3

# input dimensions
n1 = 13
n2 = 15
n3 = 17
n4 = 19
n5 = 21
# output dimensions
m1 = 14
m2 = 16
m3 = 18
m4 = 20
m5 = 22
# state dimensions
p12 = 1
p21 = 2
p14 = 3
p41 = 4
p25 = 5
p52 = 6
p14 = 7
p41 = 8
p23 = 9
p32 = 10
p43 = 11
p34 = 12


id = 1;
neighbors = [2, 4];
trans = Matrix(undef, 2, 2);
trans[1, 1], trans[1, 2] = ZeroMatrix(p12, p21), rand(p12, p41)
trans[2, 1], trans[2, 2] = rand(p14, p21), ZeroMatrix(p14, p41)
inp = [rand(p12, n1), rand(p14, n1)];
out = [rand(m1, p21), rand(m1, p41)];
D = rand(m1, n1);

node1 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node1.trans[2, 4] == trans[1, 2]
@test node1.inp[2] == inp[1]
@test node1.out[4] == out[2]
@test node1.D == D

id = 2;
neighbors = [1, 5, 3];
trans = Matrix(undef, 3, 3);
trans[1, 1], trans[1, 2], trans[1, 3] = ZeroMatrix(p21, p12), rand(p21, p52), rand(p21, p32);
trans[2, 1], trans[2, 2], trans[2, 3] = rand(p25, p12), ZeroMatrix(p25, p52), rand(p25, p32);
trans[3, 1], trans[3, 2], trans[3, 3] = rand(p23, p12), rand(p23, p52), ZeroMatrix(p23, p32);
inp = [rand(p21, m2), rand(p25, m2), rand(p23, m2)];
out = [rand(n2, p12), rand(n2, p52), rand(n2, p32)];
D = rand(n2, m2);
node2 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node2.trans[3, 1] == trans[3, 1]
@test node2.inp[5] == inp[2]
@test node2.out[1] == out[1]
@test node2.D == D

id = 3;
neighbors = [2, 4];
trans = Matrix(undef, 2, 2);
trans[1, 1], trans[1, 2] = ZeroMatrix(p32, p23), rand(p32, p43)
trans[2, 1], trans[2, 2] = rand(p34, p23), ZeroMatrix(p34, p43)
inp = [rand(p32, m3), rand(p34, m3)];
out = [rand(n3, p23), rand(n3, p43)];
D = rand(n3, m3);
node3 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node3.trans[2, 4] == trans[1, 2]
@test node3.inp[4] == inp[2]
@test node3.out[2] == out[1]
@test node3.D == D

id = 4;
neighbors = [1, 3];
trans = Matrix(undef, 2, 2);
trans[1, 1], trans[1, 2] = rand(p41, p14), rand(p41, p34)
trans[2, 1], trans[2, 2] = rand(p43, p14), ZeroMatrix(p43, p34)
inp = [rand(p41, m4), rand(p43, m4)];
out = [rand(n4, p14), rand(n4, p34)];
D = rand(n4, m4);
node4 = Spinner{Float64}(id, neighbors, trans, inp, out, D);


@test node4.trans[1, 3] == trans[1, 2]
@test node4.inp[3] == inp[2]
@test node4.out[1] == out[1]
@test node4.D == D

id = 5;
neighbors = [2];
trans = Matrix(undef, 1, 1);
trans[1, 1] = ZeroMatrix(p52, p25)
inp = [rand(p52, n5)];
out = [rand(m5, p25)];
D = rand(m5, n5);
node5 = Spinner{Float64}(id, neighbors, trans, inp, out, D);

@test node5.trans[2, 2] == trans[1, 1]
@test node5.inp[2] == inp[1]
@test node5.out[2] == out[1]
@test node5.D == D

nodeset_cyclic = IndexedVector{Spinner}([node1, node2, node3, node4, node5])

######################################################
# Test: Checking if vector of spinners is consistent #
######################################################

@test GIRS_is_consistent(nodeset_acyclic)
@test GIRS_is_consistent(nodeset_cyclic)

#####################################################################
# Test: Checking if vector of spinners has no bounce back operators #
#####################################################################

@test has_no_bounce_back_operators(nodeset_acyclic)
@test !has_no_bounce_back_operators(nodeset_cyclic)

#######################################
# Test: graph connected and acyclic   #
#######################################
adj_list_nodeset_acyclic = Dict(s.id => s.neighbors for s in values(nodeset_acyclic))
adj_list_nodeset_cyclic = Dict(s.id => s.neighbors for s in values(nodeset_cyclic))
graph_acyclic = UndirectedGraph(adj_list_nodeset_acyclic)
graph_cyclic = UndirectedGraph(adj_list_nodeset_cyclic)
@test is_connected_acyclic(graph_acyclic)
@test !is_connected_acyclic(graph_cyclic)

##################################
# Test: determine tree hierarchy #
##################################

# recall graph:
#         3 -- 1 -- 2 
#              |
#         4 -- 6 -- 5

tree = Tree(graph_acyclic, 1)
@test depth(tree) == 2
@test Set(tree.levels[1]) == Set([1])
@test Set(tree.levels[2]) == Set([3, 2, 6])
@test Set(tree.levels[3]) == Set([4, 5])
@test isnothing(tree.parent[1])
@test tree.parent[2] == 1
@test tree.parent[6] == 1
@test Set(tree.children[1]) == Set([3, 6, 2])
@test Set(tree.children[6]) == Set([4, 5])
@test Set(tree.children[2]) == Set([])
@test length(tree.siblings) == length(nodeset_acyclic)
@test Set(tree.siblings[3]) == Set([2, 6])
@test Set(tree.siblings[2]) == Set([3, 6])
@test Set(tree.siblings[6]) == Set([2, 3])
@test Set(tree.siblings[4]) == Set([5])
@test Set(tree.siblings[5]) == Set([4])
@test Set(tree.siblings[1]) == Set([])

@test Set(tree.descendants[1]) == Set([1, 2, 3, 4, 5, 6])
@test Set(tree.descendants[2]) == Set([2])
@test Set(tree.descendants[3]) == Set([3])
@test Set(tree.descendants[4]) == Set([4])
@test Set(tree.descendants[5]) == Set([5])
@test Set(tree.descendants[6]) == Set([4, 5, 6])

@test Set(tree.descendants_complement[1]) == Set([])
@test Set(tree.descendants_complement[2]) == Set([1, 3, 4, 5, 6])
@test Set(tree.descendants_complement[3]) == Set([1, 2, 4, 5, 6])
@test Set(tree.descendants_complement[4]) == Set([1, 3, 2, 5, 6])
@test Set(tree.descendants_complement[5]) == Set([1, 3, 4, 2, 6])
@test Set(tree.descendants_complement[6]) == Set([1, 2, 3])


tree = Tree(graph_acyclic, 2)
@test depth(tree) == 3
@test Set(tree.levels[1]) == Set([2])
@test Set(tree.levels[2]) == Set([1])
@test Set(tree.levels[3]) == Set([3, 6])
@test Set(tree.levels[4]) == Set([4, 5])
@test Set(tree.children[2]) == Set([1])
@test Set(tree.children[1]) == Set([3, 6])
@test Set(tree.children[6]) == Set([4, 5])
@test Set(tree.children[5]) == Set([])
@test length(tree.siblings) == length(nodeset_acyclic)
@test Set(tree.siblings[3]) == Set([6])
@test Set(tree.siblings[2]) == Set([])
@test Set(tree.siblings[6]) == Set([3])
@test Set(tree.siblings[4]) == Set([5])
@test Set(tree.siblings[5]) == Set([4])
@test Set(tree.siblings[1]) == Set([])

#########################
# Test: TQS matrix type #
#########################

# recall graph:
#         3 -- 1 -- 2 
#              |
#         4 -- 6 -- 5

node_ordering = [1, 2, 3, 4, 5, 6]
T = TQSMatrix(GIRSRepresentation(nodeset_acyclic, node_ordering), 1);

trans_acyclic = Dict{Int, IndexedMatrix{AbstractMatrix{eltype(T)}}}(
	k => node.trans for (k, node) in nodeset_acyclic
);
inp_acyclic = Dict{Int, IndexedVector{AbstractMatrix{eltype(T)}}}(
	k => node.inp for (k, node) in nodeset_acyclic
);
out_acyclic = Dict{Int, IndexedVector{AbstractMatrix{eltype(T)}}}(
	k => node.out for (k, node) in nodeset_acyclic
);
D_acyclic =
	Dict{Int, AbstractMatrix{eltype(T)}}(k => node.D for (k, node) in nodeset_acyclic);

G_alternative = GIRSRepresentation{eltype(T)}(
	trans_acyclic,
	inp_acyclic,
	out_acyclic,
	D_acyclic,
	[1, 2, 3, 4, 5, 6],
);
T_alternative = TQSMatrix(G_alternative, 2);


T11 = T.spinners[1].D                                                                   #                1 
T21 = T.spinners[2].out[1] * T.spinners[1].inp[2]                                   #           2 <- 1 
T31 = T.spinners[3].out[1] * T.spinners[1].inp[3]                                 #           3 <- 1
T41 = T.spinners[4].out[6] * T.spinners[6].trans[4, 1] * T.spinners[1].inp[6]           #      4 <- 6 <- 1 
T51 = T.spinners[5].out[6] * T.spinners[6].trans[5, 1] * T.spinners[1].inp[6]           #      5 <- 6 <- 1 
T61 = T.spinners[6].out[1] * T.spinners[1].inp[6]                   #           6 <- 1

T12 = T.spinners[1].out[2] * T.spinners[2].inp[1]                 #           1 <- 2
T22 = T.spinners[2].D    #                2
T32 = T.spinners[3].out[1] * T.spinners[1].trans[3, 2] * T.spinners[2].inp[1]                 #      3 <- 1 <- 2
T42 =
	T.spinners[4].out[6] *
	T.spinners[6].trans[4, 1] *
	T.spinners[1].trans[6, 2] *
	T.spinners[2].inp[1]   # 4 <- 6 <- 1 <- 2
T52 =
	T.spinners[5].out[6] *
	T.spinners[6].trans[5, 1] *
	T.spinners[1].trans[6, 2] *
	T.spinners[2].inp[1]   # 6 <- 1 <- 1 <- 2
T62 = T.spinners[6].out[1] * T.spinners[1].trans[6, 2] * T.spinners[2].inp[1]                 #      6 <- 1 <- 2


T13 = T.spinners[1].out[3] * T.spinners[3].inp[1]                 #           1 <- 3 
T23 = T.spinners[2].out[1] * T.spinners[1].trans[2, 3] * T.spinners[3].inp[1]                 #      2 <- 1 <- 3
T33 = T.spinners[3].D    #                3
T43 =
	T.spinners[4].out[6] *
	T.spinners[6].trans[4, 1] *
	T.spinners[1].trans[6, 3] *
	T.spinners[3].inp[1]                  # 4 <- 6 <- 1 <- 3  
T53 =
	T.spinners[5].out[6] *
	T.spinners[6].trans[5, 1] *
	T.spinners[1].trans[6, 3] *
	T.spinners[3].inp[1]                 # 5 <- 6 <- 1 <- 3  
T63 = T.spinners[6].out[1] * T.spinners[1].trans[6, 3] * T.spinners[3].inp[1]                    #      6 <- 1 <- 3

T14 = T.spinners[1].out[6] * T.spinners[6].trans[1, 4] * T.spinners[4].inp[6]                  #      1 <- 6 <- 4 
T24 =
	T.spinners[2].out[1] *
	T.spinners[1].trans[2, 6] *
	T.spinners[6].trans[1, 4] *
	T.spinners[4].inp[6]  # 2 <- 1 <- 6 <- 4 
T34 =
	T.spinners[3].out[1] *
	T.spinners[1].trans[3, 6] *
	T.spinners[6].trans[1, 4] *
	T.spinners[4].inp[6]                # 3 <- 1 <- 6 <- 4
T44 = T.spinners[4].D    #                4
T54 = T.spinners[5].out[6] * T.spinners[6].trans[5, 4] * T.spinners[4].inp[6]                 #      5 <- 6 <- 4
T64 = T.spinners[6].out[4] * T.spinners[4].inp[6]                  #           6 <- 4    

T15 = T.spinners[1].out[6] * T.spinners[6].trans[1, 5] * T.spinners[5].inp[6]                 #      1 <- 6 <- 5
T25 =
	T.spinners[2].out[1] *
	T.spinners[1].trans[2, 6] *
	T.spinners[6].trans[1, 5] *
	T.spinners[5].inp[6]                # 2 <- 1 <- 6 <- 5
T35 =
	T.spinners[3].out[1] *
	T.spinners[1].trans[3, 6] *
	T.spinners[6].trans[1, 5] *
	T.spinners[5].inp[6]                 # 3 <- 1 <- 6 <- 5 
T45 = T.spinners[4].out[6] * T.spinners[6].trans[4, 5] * T.spinners[5].inp[6]                  #      4 <- 6 <- 5 
T55 = T.spinners[5].D    #                5
T65 = T.spinners[6].out[5] * T.spinners[5].inp[6]                 #           6 <- 5

T16 = T.spinners[1].out[6] * T.spinners[6].inp[1]                 #           1 <- 6
T26 = T.spinners[2].out[1] * T.spinners[1].trans[2, 6] * T.spinners[6].inp[1]                 #      2 <- 1 <- 6
T36 = T.spinners[3].out[1] * T.spinners[1].trans[3, 6] * T.spinners[6].inp[1]                   #      3 <- 1 <- 6
T46 = T.spinners[4].out[6] * T.spinners[6].inp[4]                   #           4 <- 6
T56 = T.spinners[5].out[6] * T.spinners[6].inp[5]                    #           5 <- 6
T66 = T.spinners[6].D    #                6 


Tdense = [
	T11 T12 T13 T14 T15 T16
	T21 T22 T23 T24 T25 T26
	T31 T32 T33 T34 T35 T36
	T41 T42 T43 T44 T45 T46
	T51 T52 T53 T54 T55 T56
	T61 T62 T63 T64 T65 T66
];



TG = GraphPartitionedMatrix(
	Tdense,
	node_ordering,
	[T.partition.m[k] for k ∈ node_ordering],
	[T.partition.n[k] for k ∈ node_ordering],
	T.tree.graph.adjacency_list,
)

####################
# Test: Stategraph #
####################


# recall again graph:
#         3 -- 1 -- 2 
#              |
#         4 -- 6 -- 5

stategraph = StateGraph{Float64}(T);
@test all([Set(T.tree.graph.adjacency_list[k]) == Set(keys(stategraph[k])) for k in T.partition.node_ordering])
@test all([
	all([T.spinners[k].p_in[v_k] == length(v_v) for (v_k, v_v) in v]) for
	(k, v) in stategraph
])


############################################
# Test: TQS matrix vector and TQS to dense #
############################################

x = randn(size(T, 2))
@test T * x ≈ Tdense * x

X = randn(size(T, 2), 30)
@test T * X ≈ Tdense * X

@test Matrix(T) ≈ Tdense
@test Matrix(T_alternative) ≈ Tdense

[true for i ∈ 1:5, j ∈ 1:6]

#############################
# Test: HankelFactorization #
#############################

X = rand(10, 5)
Y = rand(5, 10)

in_nodes = [4, 8, 9]
in_ranges = IndexedVector{UnitRange}(4 => 1:3, 8 => 4:7, 9 => 8:10)
out_nodes = [3, 5, 1, 2]
out_ranges = IndexedVector{UnitRange}(3 => 1:3, 5 => 4:5, 1 => 6:6, 2 => 7:10)

H = HankelFactorization(X, Y, in_nodes, in_ranges, out_nodes, out_ranges)

@test Xblock(H, [1, 2]) == X[6:10, :]
@test Xblock(H, [2, 1]) == vcat(X[7:10, :], X[6:6, :])
@test Xblock(H, [3, 5, 1, 2]) == X

@test Yblock(H, [8, 9]) == Y[:, 4:10]
@test Yblock(H, [9, 8]) == [Y[:, 8:10] Y[:, 4:7]]
@test Yblock(H, [4, 8, 9]) == Y

############################################
# Test: Construction/realization algorithm #
############################################

root = 1
tol = 1E-12
ρ_max = Inf
TG_TQS, ρ = TQSMatrix(TG, root, tol, ρ_max);

@test Matrix(TG_TQS) ≈ Tdense

###########################################################
# Test: inner constructors take the fields and check them #
###########################################################

# rebuild `x` from its fields, optionally replacing field `name` by `val`
fields(x) = [getfield(x, f) for f ∈ fieldnames(typeof(x))]
rebuild(x, name = nothing, val = nothing) =
	typeof(x)([f == name ? val : getfield(x, f) for f ∈ fieldnames(typeof(x))]...)

# ZeroMatrix
@test_throws AssertionError ZeroMatrix(-1, 2)

# IndexedMatrix
IM = IndexedMatrix{Float64}(rand(3, 3), [7, 8, 9])
@test fields(rebuild(IM)) == fields(IM)
@test_throws AssertionError rebuild(IM, :index_map, Dict(7 => 2, 8 => 1, 9 => 3))
@test_throws AssertionError IndexedMatrix{Float64}(rand(3, 3), [7, 7, 9])

# GraphPartitionedMatrix
@test fields(rebuild(A)) == fields(A)
P = A.partition
@test_throws AssertionError rebuild(A, :partition, BlockPartition(P.node_ordering, P.n, P.m))
@test_throws AssertionError rebuild(A, :graph, UndirectedGraph(Dict(1 => [2], 2 => [1])))

# TQSMatrix
@test fields(rebuild(T)) == fields(T)
P = T.partition
@test_throws AssertionError rebuild(
	T,
	:partition,
	BlockPartition(P.node_ordering, IndexedVector{Int}(k => v + 1 for (k, v) ∈ P.m), P.n),
)
@test Matrix(rebuild(T, :tree, Tree(T.tree.graph, 4))) ≈ Matrix(T)  # any root works

# Tree
@test fields(rebuild(tree)) == fields(tree)
@test_throws AssertionError rebuild(tree, :root, first(tree.children[tree.root]))
@test_throws AssertionError rebuild(tree, :levels, reverse(tree.levels))

# HankelFactorization
@test fields(rebuild(H)) == fields(H)
@test_throws AssertionError rebuild(H, :in_nodes, reverse(H.in_nodes))
@test_throws AssertionError rebuild(H, :Y, H.Y[:, 2:end])


#########################
# Test: UndirectedGraph #
#########################

@test_throws AssertionError UndirectedGraph(Dict(1 => [2], 2 => Int[]))      # not symmetric
@test_throws AssertionError UndirectedGraph(Dict(1 => [1]))                  # self-loop
@test_throws AssertionError UndirectedGraph(Dict(1 => [2]))                  # unknown neighbour
@test_throws AssertionError UndirectedGraph(Dict(1 => [2, 2], 2 => [1]))     # duplicate neighbour
@test UndirectedGraph(Dict(1 => [2, 3], 2 => [1], 3 => [1])) ==
	  UndirectedGraph(Dict(1 => [3, 2], 2 => [1], 3 => [1]))                 # order does not matter
@test UndirectedGraph(Dict(1 => [2], 2 => [1], 3 => Int[])) != UndirectedGraph(Dict(1 => [2], 2 => [1]))
@test !is_connected_acyclic(UndirectedGraph(Dict(1 => [2], 2 => [1], 3 => Int[])))  # not connected

# a tree needs a connected acyclic graph
@test_throws AssertionError Tree(graph_cyclic, 1)
@test_throws AssertionError Tree(UndirectedGraph(Dict(1 => [2], 2 => [1], 3 => Int[])), 1)
@test depth(Tree(UndirectedGraph(Dict(7 => Int[])), 7)) == 0


########################
# Test: BlockPartition #
########################

P = BlockPartition([3, 1], [2, 4], [5, 6])
@test P.mrange[3] == 1:2 && P.mrange[1] == 3:6
@test P.nrange[3] == 1:5 && P.nrange[1] == 6:11
@test size(P) == (6, 11)
@test BlockPartition([3, 1], Dict(3 => 2, 1 => 4), Dict(3 => 5, 1 => 6)) == P
@test BlockPartition([1, 3], [4, 2], [6, 5]) != P                         # other block order
@test fields(rebuild(P)) == fields(P)
@test_throws AssertionError BlockPartition([3, 3], [2, 4], [5, 6])        # duplicate node
@test_throws AssertionError BlockPartition([3, 1], [-2, 4], [5, 6])       # negative size
@test_throws AssertionError rebuild(P, :mrange, P.nrange)                 # stale ranges


############################
# Test: GIRSRepresentation #
############################

# spinners on a graph with a cycle and with bounce-back operators are a valid
# GIRS representation, but not a TQS matrix
G_cyclic = GIRSRepresentation(nodeset_cyclic, [1, 2, 3, 4, 5])
@test G_cyclic.graph == graph_cyclic
@test !is_connected_acyclic(G_cyclic.graph)
@test !has_no_bounce_back_operators(G_cyclic)
@test size(G_cyclic) == (sum(s.m for s ∈ values(nodeset_cyclic)), sum(s.n for s ∈ values(nodeset_cyclic)))
@test fields(rebuild(G_cyclic)) == fields(G_cyclic)
@test_throws AssertionError rebuild(G_cyclic, :graph, graph_acyclic)
@test_throws AssertionError TQSMatrix(G_cyclic, 1)

# a bounce-back operator on a tree: a valid GIRS representation, not a TQS matrix
s1 = nodeset_acyclic[1]
s1_bounce = Spinner{Float64}(
	1,
	IndexedMatrix{AbstractMatrix{Float64}}(copy(s1.trans.array), s1.trans.labels),
	s1.inp,
	s1.out,
	s1.D,
)
s1_bounce.trans[2, 2] = rand(s1.p_out[2], s1.p_in[2])
G_bounce = GIRSRepresentation(merge(nodeset_acyclic, Dict(1 => s1_bounce)), node_ordering)
@test is_connected_acyclic(G_bounce.graph)
@test !has_no_bounce_back_operators(G_bounce)
@test_throws AssertionError TQSMatrix(G_bounce, 1)
@test has_no_bounce_back_operators(nodeset_acyclic)                       # original untouched

# TQS matrix ↔ GIRS representation
G = GIRSRepresentation(T)
@test G.spinners == T.spinners && G.graph == T.tree.graph && G.partition == T.partition
@test Matrix(TQSMatrix(G, 1)) ≈ Matrix(T)

# re-rooting does not change the matrix
T5 = TQSMatrix(T, 5)
@test T5.tree.root == 5 && depth(T5.tree) == 3
@test Matrix(T5) ≈ Matrix(T)
@test GraphPartitionedMatrix(T5).partition == T.partition
