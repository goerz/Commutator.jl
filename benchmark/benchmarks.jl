using Commutator: commutator!

using BenchmarkTools

SUITE = BenchmarkGroup()

# the benchmark suite names correspond to the names of the test sets

SUITE["numerical dense matrices"] = BenchmarkGroup(["commutator"])
for T ∈ [Complex128, Float64], β ∈ Any[0, 1, -1, rand(T)], N ∈ [4, 500]
    α, A, B, C = rand(T), rand(T, N, N), rand(T, N, N), rand(T, N, N)
    C2 = copy(C)
    SUITE["numerical dense matrices"]["T=$T, β=$β, N=$N - super"] =
        @benchmarkable invoke(commutator!, (Number, AbstractMatrix, AbstractMatrix, Number, AbstractMatrix), $α, $A, $B, $β, $C)
    SUITE["numerical dense matrices"]["T=$T, β=$β, N=$N"] =
        @benchmarkable commutator!($α, $A, $B, $β, $C)
end

#=
# If a cache of tuned parameters already exists, use it, otherwise, tune and cache
# the benchmark parameters. Reusing cached parameters is faster and more reliable
# than re-tuning `SUITE` every time the file is included.
paramspath = joinpath(dirname(@__FILE__), "params.json")

if isfile(paramspath)
    loadparams!(SUITE, BenchmarkTools.load(paramspath)[1], :evals);
else
    tune!(SUITE)
    BenchmarkTools.save(paramspath, params(SUITE));
end
=#
