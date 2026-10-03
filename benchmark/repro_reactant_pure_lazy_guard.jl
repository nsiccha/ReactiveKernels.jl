# Backend-only boundary: pure live branches survive MLIR but stock default CPU
# XLA speculates their inactive arithmetic. Numerical agreement is insufficient.
using Reactant, Enzyme, Test
Reactant.set_default_backend("cpu")

function pure_lazy_guard(x, scale)
    value = zero(x)
    Reactant.@trace if scale > 0
        value = log(scale) + x / scale
    else
        value = -2 * one(x)
    end
    return value
end

pure_lazy_gradient(x, scale) = Enzyme.gradient(
    Enzyme.Reverse, Enzyme.Const(pure_lazy_guard), x, scale)

function lazy_executable_inventory(hlo::AbstractString)
    instructions = collect(eachmatch(
        r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = .*?\s+([A-Za-z][A-Za-z0-9_-]*)\(", hlo))
    assignments = collect(eachmatch(r"(?m)^\s*(?:ROOT\s+)?%?[\w.-]+ = ", hlo))
    @test length(instructions) == length(assignments)
    counts = Dict{String,Int}()
    for m in instructions
        op = m.captures[1]
        counts[op] = get(counts, op, 0) + 1
    end
    return counts
end

lazy_host(x::Tuple) = map(lazy_host, x)
lazy_host(x::Reactant.AbstractConcreteArray) = Array(x)
lazy_host(x) = Reactant.to_number(x)
lazy_traced(x::AbstractArray) = Reactant.to_rarray(x)
lazy_traced(x::Number) = Reactant.to_rarray(x; track_numbers=true)

function check_pure_lazy_guard()
    @testset "pure lazy guard: default executable boundary" begin
        x, scale = lazy_traced.((0.5, 2.0))
        primal = Reactant.@compile pure_lazy_guard(x, scale)
        reverse = Reactant.@compile pure_lazy_gradient(x, scale)
        for s in (2.0, 0.7, -1.0, 0.0, NaN)
            rs = lazy_traced(s)
            @test lazy_host(primal(x, rs)) ≈ pure_lazy_guard(0.5, s)
            @test all(isapprox.(lazy_host(reverse(x, rs)), pure_lazy_gradient(0.5, s)))
            @test isequal(lazy_host(rs), s)
        end
        for compiled in (primal, reverse)
            hlo = repr(only(Reactant.XLA.get_hlo_modules(compiled.exec)))
            counts = lazy_executable_inventory(hlo)
            println("default executable inventory: ", counts)
            # Stock 0.2.290: zero conditionals despite correct values/gradients.
            @test_broken get(counts, "conditional", 0) > 0
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    check_pure_lazy_guard()
end
