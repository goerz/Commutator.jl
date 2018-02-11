using Commutator: commutator!

using BenchmarkTools

const SUITE = BenchmarkGroup()

for T ∈ [Complex128, Float64], β ∈ Any[0, 1, -1, rand(T)], N ∈ [4, 500]
    α, A, B, C = rand(T), rand(T, N, N), rand(T, N, N), rand(T, N, N)
    SUITE["dense_gemm"] = BenchmarkGroup(["commutator"])
    SUITE["dense_gemm"]["T=$T, β=$β, N=$N"] =
        @benchmarkable commutator!($α, $A, $B, $β, $C)
end
