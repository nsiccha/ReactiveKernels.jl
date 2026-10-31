module GraphConstructorTests
using ReactiveKernels, Test

@kernel piece(x) = begin
    shifted = x .+ 1
    return shifted
end
graph_source(tag::Symbol) = tag === :piece ? piece : error("unknown graph")
function ReactiveKernels._kernel_resolve_graph_constructor(
        ::typeof(graph_source), arguments, mod::Module)
    length(arguments) == 1 && only(arguments) isa QuoteNode || return nothing
    graph_source(only(arguments).value)
end

@kernel model(x) = begin
    y = graph_source(:piece)(x)
    return sum(y)
end

const ordinary_calls = Ref(0)
ordinary_source() = (ordinary_calls[] += 1; identity)
@kernel ordinary(x) = begin
    return ordinary_source()(x)
end

@testset "opt-in static graph constructors compose; other calls stay runtime" begin
    @test ordinary_calls[] == 0
    @test prepare(ordinary)(2.0) == 2.0
    @test ordinary_calls[] == 1
    f = prepare(model)
    @test f([2.0,3.0]) == 7.0
    @test f(Float64[]) == 0.0
    @test !occursin("graph_source", string(code_expr(f)))
    @test any(r -> occursin(".+", string(r.source)), plan(model).recipes)
end
end
