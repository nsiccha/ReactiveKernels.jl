module TGIControlFlowTests
using ReactiveKernels, ReactiveKernelsPPL, Reactant, Test
import Enzyme

const readscalar = ReactiveKernels._tensorized_getindex
interval(q) = tgi_interval_logprob(readscalar(q, 1), readscalar(q, 2))
interval_gradient(q) = only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(interval), q))

@testset "TGI interval branches stay lazy in primal and reverse" begin
    q = Reactant.to_rarray([0.0, 1.0])
    hlo = repr(Reactant.@code_hlo optimize=false interval(q))
    @test count("stablehlo.if", hlo) == 2
    compiled = Reactant.@compile interval(q)
    compiled_gradient = Reactant.@compile interval_gradient(q)
    for values in ([0.0, 1.0], [1.0, 2.0], [-2.0, -1.0],
            [1.0, 1.0], [2.0, 1.0], [31.0, 40.0])
        expected = interval(values)
        native_gradient = interval_gradient(values)
        rq = Reactant.to_rarray(values)
        actual = Float64(compiled(rq))
        @test actual == expected || isapprox(actual, expected; rtol=1e-12)
        @test Array(compiled_gradient(rq)) ≈ native_gradient rtol=1e-10 atol=1e-12
        if values[2] <= values[1] || values[1] >= 30
            @test expected == -Inf
            @test native_gradient == zeros(2)
        end
    end
end

@testset "direct nadir retains data-size independent structure" begin
    counts = Int[]
    for n in (3, 9)
        # Keep this gradient oracle away from ties with the baseline minimum.
        x = collect(range(0.31, -0.51; length=n))
        rx = Reactant.to_rarray(x)
        hlo = repr(Reactant.@code_hlo optimize=false tgi_running_nadir(rx))
        @test count("stablehlo.while", hlo) == 1
        push!(counts, length(collect(eachmatch(r"stablehlo\.\w+", hlo))))
        println("DIRECT_NADIR_HLO rows=", n, " ops=", last(counts))
        compiled = Reactant.@compile tgi_running_nadir(rx)
        @test Array(compiled(rx)) == tgi_running_nadir(x)
        loss = x -> sum(tgi_running_nadir(x))
        gradient = x -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(loss), x))
        compiled_gradient = Reactant.@compile gradient(rx)
        @test Array(compiled_gradient(rx)) == gradient(x)
    end
    @test all(==(first(counts)), counts)
    empty = Reactant.to_rarray(Float64[])
    compiled = Reactant.@compile tgi_running_nadir(empty)
    @test isempty(Array(compiled(empty)))
end

@testset "traced views retain the same nadir boundary" begin
    f(x) = tgi_running_nadir(view(x, 2:length(x)))
    x = [8.0, 0.3, -0.4, -0.1]
    rx = Reactant.to_rarray(x)
    hlo = repr(Reactant.@code_hlo optimize=false f(rx))
    @test count("stablehlo.while", hlo) == 1
    compiled = Reactant.@compile f(rx)
    @test Array(compiled(rx)) == f(x)
    loss = x -> sum(f(x))
    gradient = x -> only(Enzyme.gradient(Enzyme.Reverse, Enzyme.Const(loss), x))
    compiled_gradient = Reactant.@compile gradient(rx)
    @test Array(compiled_gradient(rx)) == gradient(x)
end
end
