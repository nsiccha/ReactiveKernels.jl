using ReactiveKernels, Reactant, Test
import Enzyme
Reactant.set_default_backend("cpu")
module AbstractPlateFixtures
using ReactiveKernels
# Widen inference without changing the scalar mathematics. This makes the
# fallback reproducible independently of compiler caches and prior models.
Base.@noinline number(x)::Number = Base.inferencebarrier(x)
@kernel values(x::Vector{Float64}) = begin
    pointwise = plate(x) do xi
        number(xi)
    end
    total::Float64 = sum(pointwise)
end
@kernel guarded(x::Vector{Float64}, w::Vector{Float64}) = begin
    pointwise = plate(x, w) do xi, wi
        value = xi > 0 ? log(xi) * wi : 0.0
        number(value)
    end
    total::Float64 = sum(pointwise)
end
end
function _abstract_plate_inventory(text)
    counts = Dict{String,Int}()
    for m in eachmatch(r"\b(?:stablehlo|chlo|func|arith|enzyme|scf|tensor|cf|math|linalg|memref)\.\w+", text)
        counts[m.match] = get(counts, m.match, 0) + 1
    end
    counts
end
@testset "abstract scalar inference in a plate" begin
    @test Base.promote_op(AbstractPlateFixtures.number, Reactant.TracedRNumber{Float64}) === Number
    for spec in (AbstractPlateFixtures.values,)
        kernel = prepare(spec; want=:total)
        collected = prepare(spec; want=:pointwise)
        grad(v)=Enzyme.gradient(Enzyme.Reverse,kernel,v)
        inventories = []
        for n in (1,4,0,19,37)
            x = [isodd(i) ? 0.2i : -0.3i for i in 1:n]
            original = copy(x)
            rx = Reactant.to_rarray(x)
            @test kernel(x) ≈ sum(x)
            compiled = Reactant.@compile kernel(rx)
            pointwise = Reactant.@compile collected(rx)
            @test Float64(compiled(rx)) ≈ sum(x)
            @test Array(pointwise(rx)) ≈ x
            @test x == original
            n == 0 && continue # Released XLA cannot export empty gradients.
            reverse = Reactant.@compile grad(rx)
            @test Array(only(reverse(rx))) ≈ ones(n)
            @test only(grad(x)) ≈ ones(n)
            shifted = x .+ 0.17
            @test Float64(compiled(Reactant.to_rarray(shifted))) ≈ sum(shifted)
            @test Array(rx) == original
            n < 19 && continue
            push!(inventories, (_abstract_plate_inventory(repr(Reactant.@code_hlo optimize=false kernel(rx))),
                _abstract_plate_inventory(repr(Reactant.@code_hlo kernel(rx))),
                _abstract_plate_inventory(repr(Reactant.@code_hlo optimize=false grad(rx))),
                _abstract_plate_inventory(repr(Reactant.@code_hlo grad(rx)))))
        end
        @test all(==(first(inventories)),inventories)
    end
end
@testset "abstract scalar fallback retains lazy branches" begin
    kernel = prepare(AbstractPlateFixtures.guarded; want=:total)
    grad(v,w)=Enzyme.gradient(Enzyme.Reverse,kernel,v,w)
    inventories=[]
    for n in (19,37)
        x=[isodd(i) ? 0.2i : -0.3i for i in 1:n]
        w=[1.0+0.1i for i in 1:n]
        rx,rw=Reactant.to_rarray(x),Reactant.to_rarray(w)
        expected=sum(xi > 0 ? log(xi)*wi : 0.0 for (xi,wi) in zip(x,w))
        gexpected=([xi > 0 ? wi/xi : 0.0 for (xi,wi) in zip(x,w)],
            [xi > 0 ? log(xi) : 0.0 for xi in x])
        @test kernel(x,w) ≈ expected
        primal=Reactant.@compile kernel(rx,rw)
        @test Float64(primal(rx,rw)) ≈ expected
        reverse=Reactant.@compile grad(rx,rw)
        gradient=reverse(rx,rw)
        for i in 1:2
            @test Array(gradient[i]) ≈ gexpected[i]
            @test grad(x,w)[i] ≈ gexpected[i]
        end
        raw=repr(Reactant.@code_hlo optimize=false kernel(rx,rw))
        @test occursin("stablehlo.if",raw)
        @test !occursin("stablehlo.select",raw)
        push!(inventories,(_abstract_plate_inventory(raw),_abstract_plate_inventory(repr(Reactant.@code_hlo kernel(rx,rw))),
            _abstract_plate_inventory(repr(Reactant.@code_hlo optimize=false grad(rx,rw))),
            _abstract_plate_inventory(repr(Reactant.@code_hlo grad(rx,rw)))))
    end
    @test inventories[1] == inventories[2]
end
