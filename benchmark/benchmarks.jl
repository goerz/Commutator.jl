#=
The benchmarks are defined alongside the tests. In order to generate the
benchmark, run the following in the REPL

    julia> result = PkgBenchmark.benchmarkpkg("Commutator")
    julia>  PkgBenchmark.export_markdown("benchmark.md", result)

=#

include(joinpath(dirname(@__FILE__), "..", "test", "runtests.jl"))
