module PlateReverseSourceDispatchTests
using ReactiveKernels, DifferentiationInterface, Enzyme, Test
const RK = ReactiveKernels

composed_term(x, scale) = log1p(x * scale)

@traceable function traced_term(x::Real, scale::Real)
    if x > 0
        return composed_term(x, scale)
    else
        return sin(x * scale)
    end
end
traced_term(x::Float32, scale::Float64) = x + 2scale

struct DispatchScope{M} end
select_op(::DispatchScope, ::Val{:term}) = traced_term
forward(tag, scope, xs...) = select_op(scope, tag)(xs...)

supplied_term(x, scale) = traced_term(x, scale)
function scalar_source(f, types)
    f === supplied_term && types === Tuple{Float64,Float64} || return nothing
    scope = Expr(:call, Expr(:curly, GlobalRef(@__MODULE__, :DispatchScope), @__MODULE__))
    tag = Expr(:call, Expr(:curly, Val, QuoteNode(:term)))
    TraceableSource((:x, :scale), :(forward($tag, $scope, x, scale)), @__MODULE__)
end
register_source_provider!(scalar_source)

@kernel from_source(xs::Vector, scale::Float64) = begin
    terms = plate(xs) do x
        supplied_term(x, scale)
    end
    result = sum(terms)
    return result
end

@testset "registered scalar sources respect concrete cell dispatch and lazy branches" begin
    scale = 0.8
    for flag in (false, true), T in (Float64, Float32)
        xs = T[-2, 0.5, 1]
        k, gradient = with_plate_reverse(flag) do
            k = prepare(from_source; exemplars = (xs, scale))
            g = prepare_ad(k, AutoEnzyme(; mode = Enzyme.Reverse), xs, scale; active = :scale)
            k, g
        end
        value, ds = ad_value_and_gradient(gradient, xs, scale)
        expected = sum(x -> supplied_term(x, scale), xs)
        derivative = T === Float32 ? 2length(xs) :
            sum(x -> x > 0 ? x / (1 + x * scale) : x * cos(x * scale), xs)
        @test value ≈ expected
        @test only(ds) ≈ derivative
        if flag
            @test last(only(plate_reverse_report(k, 2))) == "rule"
            rules = RK._PlateSumRule[]
            # Rules are installed for a preparation, which collects their marks.
            marks = RK._PlateRuleMark[]
            body = RK._pr_collecting_marks(marks) do
                with_plate_reverse(() -> RK._ad_inline_plate_programs(k, (2,)).ast, true)
            end
            @test length(marks) == 1
            # A second preparation of the kernel runs the same compiled
            # program, so the program holds nothing of one preparation: its
            # mark has no state.
            again = RK._pr_collecting_marks(RK._PlateRuleMark[]) do
                with_plate_reverse(() -> RK._ad_inline_plate_programs(k, (2,)), true)
            end
            first_program = RK._pr_collecting_marks(RK._PlateRuleMark[]) do
                with_plate_reverse(() -> RK._ad_inline_plate_programs(k, (2,)), true)
            end
            @test typeof(again.f) === typeof(first_program.f)
            sites = Any[]
            RK._index_run_visit(again.ast) do ex
                Meta.isexpr(ex, :call) && ex.args[1] isa RK._PlateSumRule && push!(sites, ex.args[end])
            end
            @test sites == [RK._PlateRuleMark()]
            @test isbitstype(RK._PlateRuleMark)
            # Outside a preparation the plates stay inline.
            inline = with_plate_reverse(() -> RK._ad_inline_plate_programs(k, (2,)).ast, true)
            unverified = Ref(0)
            RK._index_run_visit(inline) do ex
                Meta.isexpr(ex, :call) && ex.args[1] isa RK._PlateSumRule && (unverified[] += 1)
            end
            @test unverified[] == 0
            RK._index_run_visit(body) do ex
                Meta.isexpr(ex, :call) && ex.args[1] isa RK._PlateSumRule && push!(rules, ex.args[1])
            end
            rule = only(rules)
            args = (xs, axes(xs), scale, 0.0, supplied_term)
            target = RK._pr_rule_target(rule, args)
            # No one-pass schedule applies to this dense plate.
            @test target === rule.program
        end
    end
end
const lookups = Ref(0)
function counted_lookup(tag)
    lookups[] += 1
    return (+)
end
counted_forward(tag, xs...) = counted_lookup(tag)(xs...)

struct CountedTag
    function CountedTag()
        lookups[] += 1
        new()
    end
end

@testset "source preparation keeps observable dispatch and constructor calls" begin
    @test isnothing(RK._pr_forward_source(counted_forward, (Val{:add}, Float64, Float64)))
    constructor = Expr(:call, GlobalRef(@__MODULE__, :CountedTag))
    @test RK._pr_source_constants(constructor) == constructor
    @test lookups[] == 0
end

end
