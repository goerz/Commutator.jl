using Commutator: commutator!
using Base.Test

using MicroLogging
using BenchmarkTools

configure_logging(min_level=:info)
#=configure_logging(min_level=:debug)=#

SUITE = BenchmarkGroup()
# Note: this file *defines* benchmarks, but does not run them.
# See benchmark/benchmarks.jl.


#==============================================================================
                                 SUPPORT
==============================================================================#


"""commutator function that we can compare against for correctness"""
function commutator(alpha, A, B, beta, C)
    C = beta * C + alpha * (A * B - B * A)
    return C  # not in place!
end


"""
Check the correctness of `commutator!(α, A, B, β, C)`. If
`enforce_noalloc=true`, also ensure that the call to `commutator!` causes zero
allocations.
"""
function check_commutator(α, A, B, β, C; enforce_noalloc=false)
    @debug "--------------------------------"
    C = copy(C)
    N = size(C, 1)
    mat_show = repr
    if N > 5
        mat_show = summary
    end
    @debug "check_commutator"
    @debug "alpha = $(repr(α))"
    @debug "A = $(mat_show(A))"
    @debug "B = $(mat_show(B))"
    @debug "beta = $(repr(β))"
    @debug "C = $(mat_show(C))"
    C_expect = commutator(α, A, B, β, C)
    w = which(commutator!,
              (typeof(α), typeof(A), typeof(B), typeof(β), typeof(C)))
    w = replace(repr(w), r"commutator!\(.*?\)", "commutator!(…)")
    @info " → $w"
    __, t, bytes, gctime, m = @timed commutator!(α, A, B, β, C)
    @info ("$t seconds (allocations: $bytes bytes, $(m.malloc)M " *
            "$(m.realloc)R $(m.poolalloc)P $(m.bigalloc)B)")
    δ = norm(C - C_expect)
    result = true
    if enforce_noalloc
        if bytes > 0
            result = false
            @warn "commutator! allocated >0 bytes"
        end
    end
    limit = 10 * eps(norm(C_expect))
    # My limit is somewhat heuristic, I'm not sure if it's more or less strict
    # than the builtin ≉. For added safety, require both.
    if C ≉ C_expect && abs(δ) > limit
        @warn "C  (out) = $(mat_show(C))"
        @warn "C_expect = $(mat_show(C_expect))"
        @warn "δ = $δ > $limit"
        result = false
    else
        @debug "δ = $δ < $limit"
    end
    @debug "--------------------------------"
    return result
end


"""Run the commutator function in order to eliminate JIT-overhead"""
function warmup(α, A, B, β, C)
    C = copy(C)
    commutator!(α, A, B, β, C)
end

function warmup(α, A, B, β, C, invoke_types)
    C = copy(C)
    invoke(commutator!, invoke_types, α, A, B, β, C)
end


"""Custom rand function that returns small values for integers"""
function myrand(S, dims...)
    if S<:Int
        return rand(1:1000, dims...)
    else
        return rand(S, dims...)
    end
end


"""Custom sprand function that returns small values for integers"""
function mysprand(S, n, m, p)
    if S<:Int
        return rand(1:1000, n, m) .* sprand(Bool, n, m, p)
    else
        return sprand(S, n, m, p)
    end
end


"""String representation of a number of two-digit precision (for labels)"""
reprnum(v) = repr(v)
reprnum(v::Float64) = @sprintf("%.2f", v)
reprnum(v::Complex128) =  @sprintf("%.2f%+.2fim", real(v), imag(v))

"""String representation of a type (for labels)"""
function reprtype(t)
    if t == Complex128
        return "Complex128"
    else
        return repr(t)
    end
end

#==============================================================================
                                  TESTS
==============================================================================#


#=
#---------------------------------------
N = 4
S = Complex128
T = Int
β = 0
α = myrand(T)
A, B, C = (
    mysprand(S, N, N, 0.3), myrand(T, N, N), myrand(Complex128, N, N))
commutator!(α, A, B, β, C)
@test check_commutator(α, A, B, β, C)
#-----------------------------------------
=#


@testset "commutator!" begin


SUITE["NDM"] = BenchmarkGroup(["commutator"])
@testset "numerical dense matrices; T=$T, β=$β, N=$N" for
        T ∈ (Complex128, Float64),
        β ∈ (0, 1, -1, rand(T)),
        N ∈ (4, 500)
    α, A, B, C = rand(T), rand(T, N, N), rand(T, N, N), rand(T, N, N)
    C2 = copy(C); C3 = copy(C)
    t_super = (Number, AbstractMatrix, AbstractMatrix, Number, AbstractMatrix)
    t_args = (typeof(α), typeof(A), typeof(B), typeof(β), typeof(C))
    @test check_commutator(α, A, B, β, C; enforce_noalloc=true)
    label = "T=$T, β=$(reprnum(β)), N=$N"
    SUITE["NDM"]["$label - super"] =
        @benchmarkable(
            invoke(commutator!, $t_super, $α, $A, $B, $β, Cb),
            setup=(Cb = copy($C)))
    warmup(α, A, B, β, C, t_args)
    SUITE["NDM"]["$label - invoke"] =
        @benchmarkable(
            invoke(commutator!, $t_args, $α, $A, $B, $β, Cb),
            setup=(Cb = copy($C)))
    warmup(α, A, B, β, C)
    SUITE["NDM"]["$label - direct"] =
        @benchmarkable(
            commutator!($α, $A, $B, $β, Cb),
            setup=(Cb = copy($C)))
end


SUITE["NDMMT"] = BenchmarkGroup(["commutator"])
@testset "numerical dense matrices (mixed types); S = $S, T=$T, β=$β" for
        S ∈ (Complex128, Float64, Int),
        T ∈ (Complex128, Float64, Int),
        β ∈ (myrand(T), myrand(S))
    # these generally cannot be mapped to gemm! and may require allocation of
    # temporary storage
    N = 4
    α, A, B, C = (
        myrand(T), myrand(S, N, N), myrand(T, N, N), myrand(Complex128, N, N))
    t_super = (Number, AbstractMatrix, AbstractMatrix, Number, AbstractMatrix)
    t_args = (typeof(α), typeof(A), typeof(B), typeof(β), typeof(C))
    @test check_commutator(α, A, B, β, C)
    label = "S=$(reprtype(S)), T=$(reprtype(T)), β=$(reprnum(β)), N=$N"
    warmup(α, A, B, β, C, t_super)
    SUITE["NDMMT"]["$label - super"] =
        @benchmarkable(
            invoke(commutator!, $t_super, $α, $A, $B, $β, Cb),
            setup=(Cb = copy($C)))
    warmup(α, A, B, β, C, t_args)
    SUITE["NDMMT"]["$label - invoke"] =
        @benchmarkable(
            invoke(commutator!, $t_args, $α, $A, $B, $β, Cb),
            setup=(Cb = copy($C)))
    warmup(α, A, B, β, C)
    SUITE["NDMMT"]["$label - direct"] =
        @benchmarkable(
            commutator!($α, $A, $B, $β, Cb),
            setup=(Cb = copy($C)))
end


SUITE["SPA"] = BenchmarkGroup(["commutator"])
@testset "sparse A; S = $S, T=$T" for
        S ∈ (Complex128, Float64, Int),
        T ∈ (Complex128, Float64, Int)
    N = 10
    if S == T
        for β ∈ (0, 1, -1, myrand(S))
            α, A, B, C = (
                myrand(T), mysprand(S, N, N, 0.3), myrand(T, N, N),
                myrand(S, N, N))
            t_super = (Number, AbstractMatrix, AbstractMatrix, Number,
                       AbstractMatrix)
            t_args = (typeof(α), typeof(A), typeof(B), typeof(β), typeof(C))
            @test check_commutator(α, A, B, β, C; enforce_noalloc=true)
            label = "S=$(reprtype(S)), T=$(reprtype(T)), β=$(reprnum(β)), N=$N"
            warmup(α, A, B, β, C, t_super)
            SUITE["SPA"]["$label - super"] =
                @benchmarkable(
                    invoke(commutator!, $t_super, $α, $A, $B, $β, Cb),
                    setup=(Cb = copy($C)))
            warmup(α, A, B, β, C, t_args)
            SUITE["SPA"]["$label - invoke"] =
                @benchmarkable(
                    invoke(commutator!, $t_args, $α, $A, $B, $β, Cb),
                    setup=(Cb = copy($C)))
            warmup(α, A, B, β, C)
            SUITE["SPA"]["$label - direct"] =
                @benchmarkable(
                    commutator!($α, $A, $B, $β, Cb),
                    setup=(Cb = copy($C)))
        end
    else  # S ≠ T
        for β ∈ (0, 1, -1, myrand(Complex128))
            α, A, B, C = (
                myrand(T), mysprand(S, N, N, 0.3), myrand(T, N, N),
                myrand(Complex128, N, N))
            C2 = copy(C); C3 = copy(C)
            t_super = (Number, AbstractMatrix, AbstractMatrix, Number,
                       AbstractMatrix)
            t_args = (typeof(α), typeof(A), typeof(B), typeof(β), typeof(C))
            @test check_commutator(α, A, B, β, C; enforce_noalloc=true)
            label = "S=$(reprtype(S)), T=$(reprtype(T)), β=$(reprnum(β)), N=$N"
            warmup(α, A, B, β, C, t_super)
            SUITE["SPA"]["$label - super"] =
                @benchmarkable(
                    invoke(commutator!, $t_super, $α, $A, $B, $β, Cb),
                    setup=(Cb = copy($C)))
            warmup(α, A, B, β, C, t_args)
            SUITE["SPA"]["$label - invoke"] =
                @benchmarkable(
                    invoke(commutator!, $t_args, $α, $A, $B, $β, Cb),
                    setup=(Cb = copy($C)))
            warmup(α, A, B, β, C)
            SUITE["SPA"]["$label - direct"] =
                @benchmarkable(
                    commutator!($α, $A, $B, $β, Cb),
                    setup=(Cb = copy($C)))
        end
    end
end


end
