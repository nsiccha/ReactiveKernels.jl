using ReactiveKernels, Reactant, Test
import Enzyme

# An authored `for` inside a recipe keeps its iteration under Reactant: the
# loop body is emitted once inside one `stablehlo.while` region, whatever the
# (data-derived) trip count (docs/src/constraints.md).

@kernel running_sum_loop(x::Vector{Float64}, scale::Float64) = begin
    prefix::Vector{Float64} = let
        n = length(x)
        out = zero(x)
        acc = zero(scale)
        for i in 1:n
            acc = acc + scale * x[i]
            out[i] = acc
        end
        out
    end
    total::Float64 = sum(prefix)
    return total
end

_traced(v) = v isa AbstractArray ? Reactant.to_rarray(v) :
             Reactant.to_rarray(v; track_numbers = true)
_host(v) = v isa Reactant.AbstractConcreteArray ? Array(v) : Reactant.to_number(v)

@testset "an authored recipe loop lowers to one retained while region" begin
    k = prepare(running_sum_loop; want = :total)
    sizes = Int[]
    for n in (8, 32)
        x = collect(range(-1.0, 1.0; length = n))
        hlo = repr(Reactant.@code_hlo optimize = false k(_traced(x), _traced(0.5)))
        @test count("stablehlo.while", hlo) == 1
        push!(sizes, count("\n", hlo))
        compiled = Reactant.@compile k(_traced(x), _traced(0.5))
        @test _host(compiled(_traced(x), _traced(0.5))) ≈ k(x, 0.5)
    end
    # Only tensor shapes differ: the body is not replicated per iteration.
    @test sizes[1] == sizes[2]
end

@testset "the retained loop differentiates like the native loop" begin
    k = prepare(running_sum_loop; want = :total)
    x = [0.5, -1.0, 2.0, 0.25]
    native_gradient(v) = Enzyme.gradient(Enzyme.Reverse, w -> k(w, 0.5), v)
    gradient(v) = Enzyme.gradient(Enzyme.Reverse, w -> k(w, 0.5), v)
    compiled = Reactant.@compile gradient(_traced(x))
    @test _host(only(compiled(_traced(x)))) ≈ only(native_gradient(x))
end

@testset "the pointwise loop output is available" begin
    k = prepare(running_sum_loop; want = :prefix)
    x = [0.5, -1.0, 2.0, 0.25]
    compiled = Reactant.@compile k(_traced(x), _traced(0.5))
    @test _host(compiled(_traced(x), _traced(0.5))) ≈ k(x, 0.5)
end
