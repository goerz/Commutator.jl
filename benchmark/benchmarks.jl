using Commutator: commutator!

using BenchmarkTools

SUITE = BenchmarkGroup()

function myrand(S, dims...)
    if S<:Int
        return rand(1:1000, dims...)
    else
        return rand(S, dims...)
    end
end


function mysprand(S, n, m, p)
    if S<:Int
        return rand(1:1000, n, m) .* sprand(Bool, n, m, p)
    else
        return sprand(S, n, m, p)
    end
end


# the benchmark suite names correspond to the names of the test sets

SUITE["numerical dense matrices"] = BenchmarkGroup(["commutator"])
for T ∈ [Complex128, Float64], β ∈ Any[0, 1, -1, rand(T)], N ∈ [4, 500]
    α, A, B, C = rand(T), rand(T, N, N), rand(T, N, N), rand(T, N, N)
    C2 = copy(C)
    SUITE["numerical dense matrices"]["T=$T, β=$β, N=$N - super"] =
        @benchmarkable invoke(commutator!, (Number, AbstractMatrix, AbstractMatrix, Number, AbstractMatrix), $α, $A, $B, $β, $C2)
    SUITE["numerical dense matrices"]["T=$T, β=$β, N=$N"] =
        @benchmarkable commutator!($α, $A, $B, $β, $C)
end


SUITE["sparse A"] = BenchmarkGroup(["commutator"])
for S ∈ (Complex128, Float64, Int), T ∈ (Complex128, Float64, Int)
    N = 10
    if S == T
        for β ∈ (0, 1, -1, myrand(S))
            α, A, B, C = (
                myrand(T), mysprand(S, N, N, 0.3), myrand(T, N, N),
                myrand(S, N, N))
            C2 = copy(C)
            SUITE["sparse A"]["T=$T, S=$S, β=$β - super"] =
                @benchmarkable invoke(commutator!, (Number, AbstractMatrix, AbstractMatrix, Number, AbstractMatrix), $α, $A, $B, $β, $C2)
            SUITE["sparse A"]["T=$T, S=$S, β=$β"] =
                @benchmarkable commutator!($α, $A, $B, $β, $C)
        end
    else
        for β ∈ (0, 1, -1, myrand(Complex128))
            α, A, B, C = (
                myrand(T), mysprand(S, N, N, 0.3), myrand(T, N, N),
                myrand(Complex128, N, N))
            C2 = copy(C)
            SUITE["sparse A"]["T=$T, S=$S, β=$β - super"] =
                @benchmarkable invoke(commutator!, (Number, AbstractMatrix, AbstractMatrix, Number, AbstractMatrix), $α, $A, $B, $β, $C2)
            SUITE["sparse A"]["T=$T, S=$S, β=$β"] =
                @benchmarkable commutator!($α, $A, $B, $β, $C)
        end
    end
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
